

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
  // auto policy = thrust::cuda::par(allocator).on(stream);
  auto policy = thrust::cuda::par.on(stream);

  // use the next int64_t aligned address after workspace + argsort_workspace_size
  thrust::device_ptr<std::int64_t> idx_ptr(reinterpret_cast<std::int64_t *>(workspace));

  // fill [idx_ptr, idx_ptr + N) with 0,1,2,… on your stream
  thrust::sequence(
    policy, idx_ptr, idx_ptr + num_input_elements + 1,
    0  // start value
  );

  int64_t * sorted_input = unique;  // thrust::raw_pointer_cast(idx_ptr) + num_input_elements + 1;
  int64_t * sorted_idx = thrust::raw_pointer_cast(idx_ptr) + 2 * num_input_elements + 1;
  int64_t * inv_loc_ptr = thrust::raw_pointer_cast(idx_ptr) + 3 * num_input_elements + 1;
  /* cudaMalloc(
      reinterpret_cast<void **>(&d_keys_out),
      num_elements * sizeof(std::int64_t)
  ); */

  void * sort_workspace_ptr =
    reinterpret_cast<void *>(thrust::raw_pointer_cast(idx_ptr) + 4 * num_input_elements + 1);

  auto sort_workspace_size =
    unique_workspace_size - (4 * num_input_elements + 1) * sizeof(std::int64_t);

  // sort the indices based on the input values
  auto result = cub::DeviceRadixSort::SortPairs(
    /* d_temp_storage */ sort_workspace_ptr,
    /* temp_storage_bytes */ sort_workspace_size,
    /* d_keys_in */ input,
    /* d_keys_out */ sorted_input,  // or nullptr if you don't need sorted keys
    /* d_values_in */ thrust::raw_pointer_cast(idx_ptr),
    /* d_values_out */ sorted_idx,
    /* num_items */ num_input_elements,
    /* begin_bit */ 0,
    /* end_bit */ 64,
    /* stream */ stream);

  /* std::vector<std::int64_t> sorted_input_host(num_input_elements);
  cudaMemcpyAsync(
    sorted_input_host.data(), sorted_input, num_input_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<std::int64_t> sorted_idx_host(num_input_elements);
  cudaMemcpyAsync(
    sorted_idx_host.data(), sorted_idx, num_input_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); */

  /////////////

  auto equal = [] __device__(const int64_t a, const int64_t b) { return a == b; };

  auto not_equal = [] __device__(const int64_t a, const int64_t b) { return a != b; };

  /* const int64_t *sorted_indices_ptr = sorted_indices.const_data_ptr<int64_t>();
  Tensor inv_loc = at::empty({num_inp}, options);
  inverse_indices = at::empty({num_inp}, options);
  int64_t* inv_loc_ptr = inv_loc.mutable_data_ptr<int64_t>();
  int64_t* inverse_indices_ptr = inverse_indices.mutable_data_ptr<int64_t>(); */

  // Replace by kernel
  thrust::adjacent_difference(
    policy, sorted_input, sorted_input + num_input_elements, inv_loc_ptr, not_equal);
  // inv_loc[0] = 0;
  cudaMemsetAsync(inv_loc_ptr, 0, sizeof(int64_t), stream);

  thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_input_elements, inv_loc_ptr);
  thrust::scatter(
    policy, inv_loc_ptr, inv_loc_ptr + num_input_elements, sorted_idx, inverse_indices);

  /* std::vector<std::int64_t> inv_loc_host(num_input_elements);
  cudaMemcpyAsync(
    inv_loc_host.data(), inv_loc_ptr, num_input_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::vector<std::int64_t> inverse_indices_host(num_input_elements);
  cudaMemcpyAsync(
    inverse_indices_host.data(), inverse_indices, num_input_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); */

  // unique and count
  // Tensor counts = at::empty({0}, options);
  int64_t num_out;

  int64_t * range_ptr = idx_ptr.get();
  num_out =
    thrust::unique_by_key(policy, sorted_input, sorted_input + num_input_elements, range_ptr, equal)
      .first -
    sorted_input;
  // range[num_out] = num_input_elements;
  cudaMemcpyAsync(
    range_ptr + num_out * sizeof(int64_t), &num_input_elements, sizeof(int64_t),
    cudaMemcpyHostToDevice, stream);

  // counts.resize_(num_out);
  // int64_t* counts_ptr = counts.mutable_data_ptr<int64_t>();
  thrust::adjacent_difference(policy, range_ptr + 1, range_ptr + num_out + 1, unique_counts);

  /* std::vector<std::int64_t> unique_counts_host(num_out);
  cudaMemcpyAsync(
    unique_counts_host.data(), unique_counts, num_out * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<std::int64_t> unique_host(num_out);
  cudaMemcpyAsync(
    unique_host.data(), sorted_input, num_out * sizeof(std::int64_t), cudaMemcpyDeviceToHost,
    stream); */
  cudaStreamSynchronize(stream);

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
