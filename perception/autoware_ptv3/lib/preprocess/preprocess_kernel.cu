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
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "autoware/ptv3/preprocess/point_type.hpp"
#include "autoware/ptv3/preprocess/preprocess_kernel.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

// output a std::int64_t vector to a txt file
#include <fstream>
#include <sstream>
#include <string>

template <typename T>
void write_vector_to_file(const std::string & filename, const std::vector<T> & vec)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    for (const auto & value : vec) {
      file << value << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

namespace autoware::ptv3
{

PreprocessCuda::PreprocessCuda(
  const PTv3Config & config, cudaStream_t stream, bool allocate_buffers)
: stream_(stream), config_(config)
{
  points_d_ = autoware::cuda_utils::make_unique<float[]>(
    config_.cloud_capacity_ * config_.num_point_feature_size_);
  cropped_points_d_ = autoware::cuda_utils::make_unique<float[]>(
    config_.cloud_capacity_ * config_.num_point_feature_size_);
  crop_mask_d_ = autoware::cuda_utils::make_unique<std::uint32_t[]>(config_.cloud_capacity_);
  crop_indices_d_ = autoware::cuda_utils::make_unique<std::uint32_t[]>(config_.cloud_capacity_);

  hashes_d_ = autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_);
  sorted_hashes_d_ = autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_);

  hash_indexes_d_ = autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_ + 1);
  sorted_hash_indexes_d_ =
    autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_ + 1);

  unique_mask_d_ = autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_);
  unique_indices_d_ = autoware::cuda_utils::make_unique<std::uint64_t[]>(config_.cloud_capacity_);

  // TODO(knzo25): delete this
  voxelization_coords_d_ =
    autoware::cuda_utils::make_unique<std::int32_t[]>(config_.cloud_capacity_ * 3);

  auto policy = thrust::cuda::par.on(stream_);
  thrust::device_ptr<std::uint64_t> idx_ptr(hash_indexes_d_.get());

  thrust::sequence(
    policy, idx_ptr, idx_ptr + config_.cloud_capacity_ + 1,
    0  // start value
  );

  // void*  d_temp   = nullptr;
  std::uint64_t * d_keys_in = nullptr;   // input keys, size N
  std::uint64_t * d_keys_out = nullptr;  // (optional) output keys, size N (can be nullptr)
  std::uint64_t * d_idx_in = nullptr;    // existing device buffer for indices, size N
  std::uint64_t * d_idx_out = nullptr;   // output indices, size N

  cub::DeviceRadixSort::SortPairs(
    /* d_temp_storage */ nullptr,
    /* temp_storage_bytes */ sort_workspace_size_,
    /* d_keys_in */ d_keys_in,
    /* d_keys_out */ d_keys_out,  // or nullptr if you don't need sorted keys
    /* d_values_in */ d_idx_in,
    /* d_values_out */ d_idx_out,
    /* num_items */ config_.cloud_capacity_,
    /* begin_bit */ 0,
    /* end_bit */ 64,
    /* stream */ 0);

  sort_workspace_d_ = autoware::cuda_utils::make_unique<std::uint8_t[]>(sort_workspace_size_);

  cudaStreamSynchronize(stream_);
}

__global__ void extractPointsKernel(
  const InputPointType * __restrict__ input_points, std::size_t points_size,
  float4 * __restrict__ output_points)
{
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= points_size) {
    return;
  }

  const InputPointType & input_point = input_points[idx];
  float4 & output_point = output_points[idx];
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.w = static_cast<float>(input_point.intensity) / 255.f;
}

__global__ void cropKernel(
  float4 * __restrict__ points, std::uint32_t * __restrict__ mask, int num_points, float min_x,
  float min_y, float min_z, float max_x, float max_y, float max_z)
{
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }
  const float & x = points[idx].x;
  const float & y = points[idx].y;
  const float & z = points[idx].z;

  mask[idx] = x >= min_x && x <= max_x && y >= min_y && y <= max_y && z >= min_z && z <= max_z;
}

template <typename scalar_t, typename mask_t>
__global__ void extractIndicesKernel(
  scalar_t * __restrict__ input_data, mask_t * __restrict__ masks, mask_t * __restrict__ indices,
  scalar_t * __restrict__ output_data, int num_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points && masks[idx] == 1) {
    output_data[indices[idx] - 1] = input_data[idx];
  }
}

template <typename scalar_t, typename mask_t>
__global__ void extractIndicesKernel(
  scalar_t * __restrict__ input_data, mask_t * __restrict__ masks, mask_t * __restrict__ indices1,
  mask_t * __restrict__ indices2, scalar_t * __restrict__ output_data, int num_points)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points && masks[idx] == 1) {
    output_data[indices1[idx] - 1] = input_data[indices2[idx]];
  }
}

__global__ void voxelizationHashKernel(
  const float4 * __restrict__ points, int3 * __restrict__ coords,
  std::uint64_t * __restrict__ hashes, int num_points, float voxel_size_x, float voxel_size_y,
  float voxel_size_z, std::int32_t min_x, std::int32_t min_y, std::int32_t min_z)
{
  // FNV64-1A
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  const float4 & point = points[idx];
  std::int32_t x = static_cast<std::int32_t>(std::floor(point.x / voxel_size_x));
  const std::int32_t y = static_cast<std::int32_t>(std::floor(point.y / voxel_size_y));
  const std::int32_t z = static_cast<std::int32_t>(std::floor(point.z / voxel_size_z));

  /* if (static_cast<std::uint64_t>(x - min_x) == 1536) {
    printf("voxelizationHashKernel: point.x: %f point.y: %f point.z: %f x: %d y: %d z: %d min_x: %d
  min_y: %d min_z: %d\n", point.x, point.y, point.z, x, y, z, min_x, min_y, min_z); x = 1535;
  } */

  int3 & coord = coords[idx];

  std::uint64_t hash = 14695981039346656037LL;
  hash *= 1099511628211LL;
  hash ^= static_cast<std::uint64_t>(x - min_x);
  hash *= 1099511628211LL;
  hash ^= static_cast<std::uint64_t>(y - min_y);
  hash *= 1099511628211LL;
  hash ^= static_cast<std::uint64_t>(z - min_z);

  hashes[idx] = hash;

  coord = make_int3(
    static_cast<std::int32_t>(x - min_x), static_cast<std::int32_t>(y - min_y),
    static_cast<std::int32_t>(z - min_z));
}

__global__ void computeGridCoordsKernel(
  const float4 * __restrict__ points, longlong3 * __restrict__ coords, int num_points,
  float voxel_size_x, float voxel_size_y, float voxel_size_z, std::int32_t min_x,
  std::int32_t min_y, std::int32_t min_z)
{
  static_assert(sizeof(longlong3) == sizeof(std::uint64_t) * 3, "longlong3 must be 24 bytes");
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  const float4 & point = points[idx];
  std::int32_t x = static_cast<std::int32_t>(std::floor(point.x / voxel_size_x));
  const std::int32_t y = static_cast<std::int32_t>(std::floor(point.y / voxel_size_y));
  const std::int32_t z = static_cast<std::int32_t>(std::floor(point.z / voxel_size_z));

  /* if (static_cast<std::uint64_t>(x - min_x) == 1536) {
    x = 1535 + min_x;
  } */

  if (idx > num_points - 10) {
    printf(
      "computeGridCoordsKernel: point.x: %f point.y: %f point.z: %f x: %d y: %d z: %d min_x: %d "
      "min_y: %d min_z: %d\n",
      point.x, point.y, point.z, x, y, z, min_x, min_y, min_z);
    // x = 1535;
  }

  coords[idx] = make_longlong3(
    static_cast<std::int64_t>(x - min_x), static_cast<std::int64_t>(y - min_y),
    static_cast<std::int64_t>(z - min_z));
}

/**
 *     def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

 */

__global__ void serializationHashKernel(
  const longlong3 * __restrict__ coords, std::int64_t * __restrict__ hashes, int num_points,
  int depth)
{
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }

  std::int64_t key1 = 0;
  std::int64_t key2 = 0;

  const std::int64_t & x = coords[idx].x;
  const std::int64_t & y = coords[idx].y;
  const std::int64_t & z = coords[idx].z;

  for (int i = 0; i < depth; ++i) {
    std::int64_t mask = 1 << i;
    key1 |= ((x & mask) << (2 * i + 2));
    key1 |= ((y & mask) << (2 * i + 1));
    key1 |= ((z & mask) << (2 * i + 0));

    key2 |= ((y & mask) << (2 * i + 2));
    key2 |= ((x & mask) << (2 * i + 1));
    key2 |= ((z & mask) << (2 * i + 0));
  }

  hashes[idx] = key1;
  hashes[idx + num_points] = key2;
}

std::size_t PreprocessCuda::generateFeatures(
  const InputPointType * input_data, unsigned int num_points, float * voxel_features,
  std::int64_t * voxel_coords, std::int64_t * voxel_hashes, std::uint64_t * precomputed_hashes)
{
  auto policy = thrust::cuda::par.on(stream_);

  const auto num_blocks = divup(num_points, config_.threads_per_block_);
  extractPointsKernel<<<num_blocks, config_.threads_per_block_, 0, stream_>>>(
    input_data, num_points, reinterpret_cast<float4 *>(points_d_.get()));

  cropKernel<<<num_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<float4 *>(points_d_.get()), crop_mask_d_.get(), num_points,
    config_.min_x_range_, config_.min_y_range_, config_.min_z_range_, config_.max_x_range_,
    config_.max_y_range_, config_.max_z_range_);

  thrust::inclusive_scan(
    policy, crop_mask_d_.get(), crop_mask_d_.get() + num_points, crop_indices_d_.get());

  std::uint32_t num_cropped_points;

  cudaMemcpyAsync(
    &num_cropped_points, crop_indices_d_.get() + num_points - 1, sizeof(std::uint32_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Test results so far start
  std::vector<std::uint32_t> crop_mask_host(num_points);
  cudaMemcpyAsync(
    crop_mask_host.data(), crop_mask_d_.get(), num_points * sizeof(std::uint32_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint32_t> crop_indices_host(num_points);
  cudaMemcpyAsync(
    crop_indices_host.data(), crop_indices_d_.get(), num_points * sizeof(std::uint32_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Test results so far end

  if (num_cropped_points == 0) {
    return 0;
  }

  const auto num_cropped_blocks = divup(num_cropped_points, config_.threads_per_block_);

  extractIndicesKernel<<<num_cropped_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<float4 *>(points_d_.get()), crop_mask_d_.get(), crop_indices_d_.get(),
    reinterpret_cast<float4 *>(cropped_points_d_.get()), num_points);

  auto min_op = [] __host__ __device__(const float4 & a, const float4 & b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
  };

  auto max_op = [] __host__ __device__(const float4 & a, const float4 & b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
  };

  float4 min_value = make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  min_value = thrust::reduce(
    policy, reinterpret_cast<float4 *>(cropped_points_d_.get()),
    reinterpret_cast<float4 *>(cropped_points_d_.get()) + num_cropped_points, min_value, min_op);

  float4 max_value = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  max_value = thrust::reduce(
    policy, reinterpret_cast<float4 *>(cropped_points_d_.get()),
    reinterpret_cast<float4 *>(cropped_points_d_.get()) + num_cropped_points, max_value, max_op);

  std::int32_t min_x = static_cast<std::int32_t>(std::floor(min_value.x / config_.voxel_x_size_));
  std::int32_t min_y = static_cast<std::int32_t>(std::floor(min_value.y / config_.voxel_y_size_));
  std::int32_t min_z = static_cast<std::int32_t>(std::floor(min_value.z / config_.voxel_z_size_));

  // Compute voxel hashes over the cropped points

  voxelizationHashKernel<<<num_cropped_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<float4 *>(cropped_points_d_.get()),
    reinterpret_cast<int3 *>(voxelization_coords_d_.get()), hashes_d_.get(), num_cropped_points,
    config_.voxel_x_size_, config_.voxel_y_size_, config_.voxel_z_size_, min_x, min_y, min_z);

  // This is to test using external hashes
  /* cudaMemcpyAsync(
    hashes_d_.get(), precomputed_hashes, num_cropped_points * sizeof(std::uint64_t),
    cudaMemcpyHostToDevice, stream_);
  cudaStreamSynchronize(stream_); */

  // sort the indices based on the input values
  auto result = cub::DeviceRadixSort::SortPairs(
    /* d_temp_storage */ reinterpret_cast<void *>(sort_workspace_d_.get()),
    /* temp_storage_bytes */ sort_workspace_size_,
    /* d_keys_in */ hashes_d_.get(),
    /* d_keys_out */ sorted_hashes_d_.get(),  // or nullptr if you don't need sorted keys
    /* d_values_in */ hash_indexes_d_.get(),
    /* d_values_out */ sorted_hash_indexes_d_.get(),
    /* num_items */ num_cropped_points,
    /* begin_bit */ 0,
    /* end_bit */ 64,
    /* stream */ stream_);

  auto not_equal = [] __device__(const uint64_t a, const uint64_t b) { return a != b; };

  // Replace by kernel
  /* auto sorted_hashes_ptr = sorted_hashes_d_.get();
  thrust::transform(
    policy,
    thrust::make_counting_iterator(1),
    thrust::make_counting_iterator(num_cropped_points),
    unique_mask_d_.get() + 1,
    [sorted_hashes_ptr] __device__ (int i) {
      return sorted_hashes_ptr[i] != sorted_hashes_ptr[i - 1];
    }
  ); */

  thrust::adjacent_difference(
    policy, sorted_hashes_d_.get(), sorted_hashes_d_.get() + num_cropped_points,
    unique_mask_d_.get(), not_equal);

  std::uint64_t one = 1;
  cudaMemcpyAsync(
    unique_mask_d_.get(), &one, sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream_);

  thrust::inclusive_scan(
    policy, unique_mask_d_.get(), unique_mask_d_.get() + num_cropped_points,
    unique_indices_d_.get());

  std::uint64_t num_unique_points;

  cudaMemcpyAsync(
    &num_unique_points, unique_indices_d_.get() + num_cropped_points - 1, sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Test results so far start
  std::vector<std::uint64_t> unique_mask_host(num_cropped_points);
  cudaMemcpyAsync(
    unique_mask_host.data(), unique_mask_d_.get(), num_cropped_points * sizeof(std::uint64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint64_t> unique_indices_host(num_cropped_points);
  cudaMemcpyAsync(
    unique_indices_host.data(), unique_indices_d_.get(), num_cropped_points * sizeof(std::uint64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint64_t> sorted_hashes_host(num_cropped_points);
  cudaMemcpyAsync(
    sorted_hashes_host.data(), sorted_hashes_d_.get(), num_cropped_points * sizeof(std::uint64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint64_t> sorted_hash_indexes_host(num_cropped_points + 1);
  cudaMemcpyAsync(
    sorted_hash_indexes_host.data(), sorted_hash_indexes_d_.get(),
    (num_cropped_points + 1) * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint64_t> hashes_host(num_cropped_points);
  cudaMemcpyAsync(
    hashes_host.data(), hashes_d_.get(), num_cropped_points * sizeof(std::uint64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::uint64_t> hash_indexes_host(num_cropped_points + 1);
  cudaMemcpyAsync(
    hash_indexes_host.data(), hash_indexes_d_.get(),
    (num_cropped_points + 1) * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<std::int32_t> raw_voxel_coords_host(num_cropped_points * 3);
  cudaMemcpyAsync(
    raw_voxel_coords_host.data(), voxelization_coords_d_.get(),
    num_cropped_points * sizeof(std::int32_t) * 3, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  write_vector_to_file("preprocessing_hashes.txt", hashes_host);
  write_vector_to_file("preprocessing_hash_indexes.txt", hash_indexes_host);
  write_vector_to_file("preprocessing_sorted_hashes.txt", sorted_hashes_host);
  write_vector_to_file("preprocessing_sorted_hash_indexes.txt", sorted_hash_indexes_host);
  write_vector_to_file("preprocessing_raw_coords.txt", raw_voxel_coords_host);

  // Extract features and grid_coords

  // apply sorted_hash_indexes_d_ to unique_indices_d
  /* thrust::gather(
    thrust::device,
    thrust::device_pointer_cast(sorted_hash_indexes_d_.get()),
    thrust::device_pointer_cast(sorted_hash_indexes_d_.get() + N),
    thrust::device_pointer_cast(unique_vals),
    thrust::device_pointer_cast(temp)
  ); */

  extractIndicesKernel<<<num_cropped_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<float4 *>(cropped_points_d_.get()), unique_mask_d_.get(),
    unique_indices_d_.get(), sorted_hash_indexes_d_.get(),
    reinterpret_cast<float4 *>(voxel_features), num_cropped_points);

  cudaStreamSynchronize(stream_);

  // one-pass reduction
  cudaStreamSynchronize(stream_);

  computeGridCoordsKernel<<<num_cropped_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<float4 *>(voxel_features), reinterpret_cast<longlong3 *>(voxel_coords),
    num_unique_points, config_.voxel_x_size_, config_.voxel_y_size_, config_.voxel_z_size_, min_x,
    min_y, min_z);

  serializationHashKernel<<<num_cropped_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<longlong3 *>(voxel_coords), voxel_hashes, num_unique_points, 11);

  // Test results so far end
  std::vector<std::uint64_t> voxel_hashes_host(2 * num_unique_points);
  cudaMemcpyAsync(
    voxel_hashes_host.data(), voxel_hashes, 2 * num_unique_points * sizeof(std::uint64_t),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
  std::vector<std::int64_t> voxel_coords_host(num_unique_points * 3);
  cudaMemcpyAsync(
    voxel_coords_host.data(), voxel_coords, num_unique_points * sizeof(std::int64_t) * 3,
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  std::vector<float> voxel_features_host(num_unique_points * config_.num_point_feature_size_);
  cudaMemcpyAsync(
    voxel_features_host.data(), voxel_features,
    num_unique_points * sizeof(float) * config_.num_point_feature_size_, cudaMemcpyDeviceToHost,
    stream_);
  cudaStreamSynchronize(stream_);

  write_vector_to_file("preprocessing_unique_coords.txt", voxel_coords_host);
  write_vector_to_file("preprocessing_unique_feats.txt", voxel_features_host);
  write_vector_to_file("preprocessing_unique_code.txt", voxel_hashes_host);

  return num_unique_points;
}

void PreprocessCuda::computeSerializationCodes(
  std::int64_t * voxel_hashes, std::int64_t * voxel_coords, int num_points)
{
  auto num_blocks = divup(num_points, config_.threads_per_block_);
  serializationHashKernel<<<num_blocks, config_.threads_per_block_, 0, stream_>>>(
    reinterpret_cast<longlong3 *>(voxel_coords), voxel_hashes, num_points, 11);
}

}  // namespace autoware::ptv3
