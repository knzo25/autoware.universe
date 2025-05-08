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

#ifndef AUTOWARE__PTV3__PREPROCESS__PREPROCESS_KERNEL_HPP_
#define AUTOWARE__PTV3__PREPROCESS__PREPROCESS_KERNEL_HPP_

#include "autoware/ptv3/preprocess/point_type.hpp"
#include "autoware/ptv3/ptv3_config.hpp"
#include "autoware/ptv3/utils.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/cuda_utils/cuda_unique_ptr.hpp>

#include <cuda_runtime_api.h>
#include <tensorview/tensor.h>

#include <cstddef>
#include <cstdint>

namespace autoware::ptv3
{

class PreprocessCuda
{
public:
  PreprocessCuda(const PTv3Config & config, cudaStream_t stream, bool allocate_buffers);

  /* cudaError_t generateSweepPoints_launch(
    const InputPointType * input_data, std::size_t points_size, float time_lag,
    const float * transform, float * output_points);

  std::size_t generateVoxels(
    const float * points, unsigned int points_size, float * voxel_features,
    std::int32_t * voxel_coords, std::int32_t * num_points_per_voxel); */

  std::size_t generateFeatures(
    const InputPointType * input_data, unsigned int num_points, float * voxel_features,
    std::int64_t * voxel_coords, std::int64_t * voxel_hashes, std::uint64_t * precomputed_hashes);

  void computeSerializationCodes(
    std::int64_t * voxel_hashes, std::int64_t * voxel_coords, int num_points);

private:
  PTv3Config config_;
  cudaStream_t stream_;

  autoware::cuda_utils::CudaUniquePtr<float[]> points_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<float[]> cropped_points_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<std::uint32_t[]> crop_mask_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<std::uint32_t[]> crop_indices_d_{nullptr};

  // Consider deleting this
  autoware::cuda_utils::CudaUniquePtr<std::int32_t[]> voxelization_coords_d_{nullptr};

  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> hashes_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> sorted_hashes_d_{nullptr};

  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> hash_indexes_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> sorted_hash_indexes_d_{nullptr};

  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> unique_mask_d_{nullptr};
  autoware::cuda_utils::CudaUniquePtr<std::uint64_t[]> unique_indices_d_{nullptr};

  autoware::cuda_utils::CudaUniquePtr<std::uint8_t[]> sort_workspace_d_{nullptr};
  std::size_t sort_workspace_size_{0};
};
}  // namespace autoware::ptv3

#endif  // AUTOWARE__PTV3__PREPROCESS__PREPROCESS_KERNEL_HPP_
