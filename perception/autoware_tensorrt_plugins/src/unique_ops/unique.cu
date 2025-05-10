// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/unique_ops/unique.hpp"

#include <cub/cub.cuh>

#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

std::int64_t unique(
  const std::int64_t * input, std::int64_t * unique, std::int64_t * inverse_indices,
  std::int64_t * unique_counts, void * workspace, std::size_t num_input_elements,
  std::size_t unique_workspace_size, cudaStream_t stream)
{
  auto policy = thrust::cuda::par.on(stream);

  thrust::device_ptr<std::int64_t> idx_ptr(reinterpret_cast<std::int64_t *>(workspace));

  thrust::sequence(policy, idx_ptr, idx_ptr + num_input_elements + 1, 0);

  std::int64_t * sorted_input = unique;
  std::int64_t * sorted_idx = thrust::raw_pointer_cast(idx_ptr) + 2 * num_input_elements + 1;
  std::int64_t * inv_loc_ptr = thrust::raw_pointer_cast(idx_ptr) + 3 * num_input_elements + 1;

  void * sort_workspace_ptr =
    reinterpret_cast<void *>(thrust::raw_pointer_cast(idx_ptr) + 4 * num_input_elements + 1);

  auto sort_workspace_size =
    unique_workspace_size - (4 * num_input_elements + 1) * sizeof(std::int64_t);

  auto result = cub::DeviceRadixSort::SortPairs(
    sort_workspace_ptr, sort_workspace_size, input,
    sorted_input,  // or nullptr if you don't need sorted keys
    thrust::raw_pointer_cast(idx_ptr), sorted_idx, num_input_elements, 0, 64, stream);

  auto equal = [] __device__(const std::int64_t a, const std::int64_t b) { return a == b; };

  auto not_equal = [] __device__(const std::int64_t a, const std::int64_t b) { return a != b; };

  thrust::adjacent_difference(
    policy, sorted_input, sorted_input + num_input_elements, inv_loc_ptr, not_equal);

  cudaMemsetAsync(inv_loc_ptr, 0, sizeof(int64_t), stream);

  thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_input_elements, inv_loc_ptr);
  thrust::scatter(
    policy, inv_loc_ptr, inv_loc_ptr + num_input_elements, sorted_idx, inverse_indices);

  std::int64_t num_out;

  std::int64_t * range_ptr = idx_ptr.get();
  num_out =
    thrust::unique_by_key(policy, sorted_input, sorted_input + num_input_elements, range_ptr, equal)
      .first -
    sorted_input;

  cudaMemcpyAsync(
    range_ptr + num_out * sizeof(int64_t), &num_input_elements, sizeof(std::int64_t),
    cudaMemcpyHostToDevice, stream);

  thrust::adjacent_difference(policy, range_ptr + 1, range_ptr + num_out + 1, unique_counts);

  return num_out;
}

std::size_t get_unique_workspace_size(std::size_t num_elements)
{
  size_t temp_size = 0;

  // void*  d_temp   = nullptr;
  int64_t * d_keys_in = nullptr;   // input keys, size N
  int64_t * d_keys_out = nullptr;  // (optional) output keys, size N (can be nullptr)
  int64_t * d_idx_in = nullptr;    // existing device buffer for indices, size N
  int64_t * d_idx_out = nullptr;   // output indices, size N

  cub::DeviceRadixSort::SortPairs(
    /* d_temp_storage */ nullptr,
    /* temp_storage_bytes */ temp_size,
    /* d_keys_in */ d_keys_in,
    /* d_keys_out */ d_keys_out,  // or nullptr if you don't need sorted keys
    /* d_values_in */ d_idx_in,
    /* d_values_out */ d_idx_out,
    /* num_items */ num_elements,
    /* begin_bit */ 0,
    /* end_bit */ 64,
    /* stream */ 0);

  return temp_size + (4 * num_elements + 1) * sizeof(std::int64_t);
}
