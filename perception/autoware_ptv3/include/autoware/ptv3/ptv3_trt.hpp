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

#ifndef AUTOWARE__PTV3__PTV3_TRT_HPP_
#define AUTOWARE__PTV3__PTV3_TRT_HPP_

#include "autoware/ptv3/postprocess/postprocess_kernel.hpp"
#include "autoware/ptv3/preprocess/preprocess_kernel.hpp"
#include "autoware/ptv3/utils.hpp"
#include "autoware/ptv3/visibility_control.hpp"

#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware/cuda_utils/cuda_unique_ptr.hpp>
#include <autoware/tensorrt_common/tensorrt_common.hpp>
#include <autoware/universe_utils/system/stop_watch.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::ptv3
{

using autoware::cuda_utils::CudaUniquePtr;

class NetworkParam
{
public:
  NetworkParam(std::string onnx_path, std::string engine_path, std::string trt_precision)
  : onnx_path_(std::move(onnx_path)),
    engine_path_(std::move(engine_path)),
    trt_precision_(std::move(trt_precision))
  {
  }

  std::string onnx_path() const { return onnx_path_; }
  std::string engine_path() const { return engine_path_; }
  std::string trt_precision() const { return trt_precision_; }

private:
  std::string onnx_path_;
  std::string engine_path_;
  std::string trt_precision_;
};

class PTV3_PUBLIC PTv3TRT
{
public:
  explicit PTv3TRT(const tensorrt_common::TrtCommonConfig & trt_config, const PTv3Config & config);
  virtual ~PTv3TRT();

  bool fake_segment(sensor_msgs::msg::PointCloud2 & out_msg);

  bool segment(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg,
    sensor_msgs::msg::PointCloud2 & out_msg, std::unordered_map<std::string, double> & proc_timing);

protected:
  void initPtr();
  void initTrt(const tensorrt_common::TrtCommonConfig & trt_config);

  bool preProcess(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pc_msg);

  bool inference();

  bool postProcess(sensor_msgs::msg::PointCloud2 & out_msg);

  std::unique_ptr<autoware::tensorrt_common::TrtCommon> network_trt_ptr_{nullptr};
  // std::unique_ptr<VoxelGenerator> vg_ptr_{nullptr};
  std::unique_ptr<autoware::universe_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};
  std::unique_ptr<PreprocessCuda> pre_ptr_{nullptr};
  std::unique_ptr<PostprocessCuda> post_ptr_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<cudaStream_t> camera_streams_{};

  PTv3Config config_;
  std::vector<int> roi_start_y_vector_;

  // pre-process inputs

  unsigned int voxel_features_size_{0};
  unsigned int voxel_coords_size_{0};

  // lidar buffers
  CudaUniquePtr<float[]> points_d_{nullptr};
  CudaUniquePtr<float[]> coord_d_{nullptr};
  CudaUniquePtr<std::int64_t[]> grid_coord_d_{nullptr};
  CudaUniquePtr<std::int64_t> offset_d_{nullptr};
  CudaUniquePtr<float[]> feat_d_{nullptr};
  CudaUniquePtr<std::int64_t[]> serialized_code_d_{nullptr};
  CudaUniquePtr<std::int64_t[]> serialized_order_d_{nullptr};
  CudaUniquePtr<std::int64_t[]> serialized_inverse_d_{nullptr};

  CudaUniquePtr<std::int64_t[]> label_pred_output_d_{nullptr};
};

}  // namespace autoware::ptv3

#endif  // AUTOWARE__PTV3__PTV3_TRT_HPP_
