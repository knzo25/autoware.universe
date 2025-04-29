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

#ifndef AUTOWARE__PTV3__POSTPROCESS__POSTPROCESS_KERNEL_HPP_
#define AUTOWARE__PTV3__POSTPROCESS__POSTPROCESS_KERNEL_HPP_

#include "autoware/ptv3/ptv3_config.hpp"
#include "autoware/ptv3/utils.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

namespace autoware::ptv3
{

class PostprocessCuda
{
public:
  explicit PostprocessCuda(const PTv3Config & config, cudaStream_t stream);

private:
  PTv3Config config_;
  cudaStream_t stream_;
};

}  // namespace autoware::ptv3

#endif  // AUTOWARE__PTV3__POSTPROCESS__POSTPROCESS_KERNEL_HPP_
