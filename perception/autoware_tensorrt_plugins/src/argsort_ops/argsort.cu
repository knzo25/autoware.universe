// Copyright 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "autoware/argsort_ops/argsort.hpp"

#include <cub/cub.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

cudaError_t argsort(
  const std::int64_t * input_d, std::int64_t * output_d, void * workspace, std::size_t num_elements,
  std::size_t argsort_workspace_size, cudaStream_t stream)
{
  // use the next int64_t aligned address after workspace + argsort_workspace_size
  int workspace_offset = (argsort_workspace_size + sizeof(std::int64_t) - 1) / sizeof(std::int64_t);
  thrust::device_ptr<std::int64_t> idx_ptr(
    reinterpret_cast<std::int64_t *>(workspace + workspace_offset * sizeof(std::int64_t)));

  // fill [idx_ptr, idx_ptr + N) with 0,1,2,… on your stream
  thrust::sequence(
    thrust::cuda::par.on(stream), idx_ptr, idx_ptr + num_elements,
    0  // start value
  );

  /* std::vector<std::int64_t> idx_host(num_elements);
  cudaMemcpyAsync(
      idx_host.data(),
      idx_ptr.get(),
      num_elements * sizeof(std::int64_t),
      cudaMemcpyDeviceToHost,
      stream
  );
  cudaStreamSynchronize(stream); */

  int64_t * d_keys_out = thrust::raw_pointer_cast(idx_ptr) + num_elements;
  /* cudaMalloc(
      reinterpret_cast<void **>(&d_keys_out),
      num_elements * sizeof(std::int64_t)
  ); */

  // sort the indices based on the input values
  auto result = cub::DeviceRadixSort::SortPairs(
    /* d_temp_storage */ workspace,
    /* temp_storage_bytes */ argsort_workspace_size,
    /* d_keys_in */ input_d,
    /* d_keys_out */ d_keys_out,  // or nullptr if you don't need sorted keys
    /* d_values_in */ thrust::raw_pointer_cast(idx_ptr),
    /* d_values_out */ output_d,
    /* num_items */ num_elements,
    /* begin_bit */ 0,
    /* end_bit */ 64,
    /* stream */ stream);

  /* cudaStreamSynchronize(stream);

  std::vector<std::int64_t> keys_host(num_elements);
  cudaMemcpyAsync(
      keys_host.data(),
      d_keys_out,
      num_elements * sizeof(std::int64_t),
      cudaMemcpyDeviceToHost,
      stream
  );
  cudaStreamSynchronize(stream);
  cudaFree(d_keys_out);

  std::vector<std::int64_t> sorted_idx(num_elements);
  cudaMemcpyAsync(
      sorted_idx.data(),
      output_d,
      num_elements * sizeof(std::int64_t),
      cudaMemcpyDeviceToHost,
      stream
  );
  cudaStreamSynchronize(stream); */

  return result;
}

std::size_t get_argsort_workspace_size(std::size_t num_elements)
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

  return temp_size;
}
