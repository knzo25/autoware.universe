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

#include "autoware/ptv3/ptv3_trt.hpp"

#include "autoware/ptv3/preprocess/point_type.hpp"
#include "autoware/ptv3/preprocess/preprocess_kernel.hpp"
#include "autoware/ptv3/ptv3_config.hpp"

#include <autoware/cuda_utils/cuda_utils.hpp>
#include <autoware/point_types/memory.hpp>
#include <autoware/universe_utils/math/constants.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>   // TODO(knzo25): delete this
#include <iostream>  // TODO(knzo25): delete this
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::ptv3
{

void loadIntegers(const std::string & filename, std::vector<int64_t> & integers)
{
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;

    char * endptr = nullptr;
    int64_t value = std::strtoll(line.c_str(), &endptr, 10);

    if (endptr != nullptr && *endptr == '\0') {
      integers.push_back(value);
    } else {
      // Not a valid integer line, skip
    }
  }
}

void loadIntegers(const std::string & filename, std::vector<uint64_t> & integers)
{
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;

    char * endptr = nullptr;
    uint64_t value = std::strtoull(line.c_str(), &endptr, 10);

    if (endptr != nullptr && *endptr == '\0') {
      integers.push_back(value);
    } else {
      // Not a valid integer line, skip
    }
  }
}

void loadFloats(const std::string & filename, std::vector<float> & floats)
{
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;

    char * endptr = nullptr;
    float value = std::strtof(line.c_str(), &endptr);

    if (endptr != nullptr && *endptr == '\0') {
      floats.push_back(value);
    } else {
      // Not a valid float line, skip
    }
  }
}

void writeIntegers(std::string const & file_name, const std::vector<std::int64_t> & data)
{
  std::ofstream file(file_name);
  if (file.is_open()) {
    for (const auto & value : data) {
      file << value << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << file_name << std::endl;
  }
}

PTv3TRT::PTv3TRT(const tensorrt_common::TrtCommonConfig & trt_config, const PTv3Config & config)
: config_(config)
{
  // TODO(knzo25): ptv3 uses a different voxelization strategy
  // vg_ptr_ = std::make_unique<VoxelGenerator>(densification_param, config_, stream_);

  stop_watch_ptr_ =
    std::make_unique<autoware::universe_utils::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("processing/inner");

  // network_trt_ptr->setTensorDataType("feat", nvinfer1::DataType::kFLOAT);

  initPtr();
  initTrt(trt_config);

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));

  // Test
  /**
   *
  network_io.emplace_back("coord", nvinfer1::Dims{2, {-1, 3}});
  network_io.emplace_back("grid_coord", nvinfer1::Dims{2, {-1, 3}});
  network_io.emplace_back("feat", nvinfer1::Dims{2, {-1, 4}});

  network_io.emplace_back("serialized_code", nvinfer1::Dims{2, {2, -1}});
  network_io.emplace_back("serialized_order", nvinfer1::Dims{2, {2, -1}});
  network_io.emplace_back("serialized_inverse", nvinfer1::Dims{2, {2, -1}});

   */
}

bool PTv3TRT::fake_segment(sensor_msgs::msg::PointCloud2 & out_msg)
{
  std::vector<float> voxelized_feat_host;
  std::vector<std::int64_t> voxelized_grid_coord_host;
  std::vector<std::int64_t> serialized_code_host;
  /* std::vector<std::int64_t> serialized_order;
  std::vector<std::int64_t> serialized_inverse; */

  std::vector<std::uint64_t> precomputed_hashes_host;

  std::vector<float> raw_feat_host;
  std::vector<std::int64_t> raw_grid_coord_host;

  loadFloats("voxelized_feat.txt", voxelized_feat_host);
  loadIntegers("voxelized_grid_coord.txt", voxelized_grid_coord_host);
  loadIntegers("serialized_code.txt", serialized_code_host);
  // loadIntegers("serialized_order.txt", serialized_order);
  // loadIntegers("serialized_inverse.txt", serialized_inverse);

  loadFloats("raw_feat.txt", raw_feat_host);
  loadIntegers("raw_grid_coord.txt", raw_grid_coord_host);
  /* loadFloats("voxelized_feat.txt", raw_feat_host);
  loadIntegers("voxelized_grid_coord.txt", raw_grid_coord_host); */

  loadIntegers("voxelization_hash.txt", precomputed_hashes_host);

  // Start test
  std::vector<InputPointType> input_points_host;
  input_points_host.resize(raw_feat_host.size() / 4);

  // create arange vector
  std::vector<std::int64_t> arange(raw_feat_host.size() / 4);
  std::iota(arange.begin(), arange.end(), 0);

  // random permutation
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  std::random_shuffle(arange.begin(), arange.end());

  for (std::size_t input_index = 0; input_index < input_points_host.size(); ++input_index) {
    auto output_index = arange[input_index];

    input_points_host[output_index].x = raw_feat_host[input_index * 4 + 0];
    input_points_host[output_index].y = raw_feat_host[input_index * 4 + 1];
    input_points_host[output_index].z = raw_feat_host[input_index * 4 + 2];
    input_points_host[output_index].intensity =
      static_cast<std::uint8_t>(255.f * raw_feat_host[input_index * 4 + 3]);
  }

  InputPointType * input_points_device = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(
    reinterpret_cast<void **>(&input_points_device),
    input_points_host.size() * sizeof(InputPointType)));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    input_points_device, input_points_host.data(),
    input_points_host.size() * sizeof(InputPointType), cudaMemcpyHostToDevice, stream_));

  auto num_unique_voxels = pre_ptr_->generateFeatures(
    input_points_device, input_points_host.size(), feat_d_.get(), grid_coord_d_.get(),
    serialized_code_d_.get(), precomputed_hashes_host.data());

  std::int64_t num_voxels = static_cast<std::int64_t>(num_unique_voxels);

  network_trt_ptr_->setInputShape("grid_coord", nvinfer1::Dims{2, {num_voxels, 3}});
  network_trt_ptr_->setInputShape("feat", nvinfer1::Dims{2, {num_voxels, 4}});
  network_trt_ptr_->setInputShape("serialized_code", nvinfer1::Dims{2, {2, num_voxels}});

  cudaStreamSynchronize(stream_);
  this->inference();
  cudaStreamSynchronize(stream_);

  // End test

  /* std::int64_t num_voxels = voxelized_feat_host.size() / 4;
  std::cout << "num_voxels: " << num_voxels << std::endl;

  int num_samples = 20;
  int num_features = 4;
  std::cout << "loaded feat: " << std::endl;
  for (std::int64_t i = 0; i < num_samples; ++i) {
    std::cout << "[";
    for (std::int64_t j = 0; j < num_features; ++j) {
      std::cout << voxelized_feat_host[i * num_features + j] << ", ";
    }
    std::cout << "]" << std::endl;
  }

  std::vector<float> test_input(4 * num_voxels);

  assert(static_cast<std::int64_t>(voxelized_feat_host.size()) == num_voxels * 4);
  assert(static_cast<std::int64_t>(voxelized_grid_coord_host.size()) == num_voxels * 3);
  assert(static_cast<std::int64_t>(serialized_code_host.size()) == num_voxels * 2);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    feat_d_.get(), voxelized_feat_host.data(), num_voxels * 4 * sizeof(float),
  cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    test_input.data(), feat_d_.get(), num_voxels * 4 * sizeof(float), cudaMemcpyDeviceToHost,
    stream_));
  cudaStreamSynchronize(stream_);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    grid_coord_d_.get(), voxelized_grid_coord_host.data(), num_voxels * 3 * sizeof(std::int64_t),
    cudaMemcpyHostToDevice, stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    serialized_code_d_.get(), serialized_code_host.data(), num_voxels * 2 * sizeof(std::int64_t),
    cudaMemcpyHostToDevice, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    test_input.data(), feat_d_.get(), num_voxels * 4 * sizeof(float), cudaMemcpyDeviceToHost,
    stream_));
  cudaStreamSynchronize(stream_);

  network_trt_ptr_->setInputShape("grid_coord", nvinfer1::Dims{2, {num_voxels, 3}});
  network_trt_ptr_->setInputShape("feat", nvinfer1::Dims{2, {num_voxels, 4}});
  network_trt_ptr_->setInputShape("serialized_code", nvinfer1::Dims{2, {2, num_voxels}});

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    test_input.data(), feat_d_.get(), num_voxels * 4 * sizeof(float), cudaMemcpyDeviceToHost,
    stream_));
  cudaStreamSynchronize(stream_);

  pre_ptr_->computeSerializationCodes(
    serialized_code_d_.get(), grid_coord_d_.get(), num_voxels);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  std::vector<std::int64_t> serialized_code_host_test(num_voxels * 2);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    serialized_code_host_test.data(), serialized_code_d_.get(), num_voxels * 2 *
  sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream_)); cudaStreamSynchronize(stream_);

  // Check that the computed serialized codes match the expected values
  for (std::int64_t i = 0; i < 2*num_voxels; ++i) {
    assert(serialized_code_host_test[i] == serialized_code_host[i]);
  }

  this->inference(); */

  std::vector<std::int64_t> label_pred_output_host(num_voxels);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    label_pred_output_host.data(), label_pred_output_d_.get(), num_voxels * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);
  writeIntegers("label_pred_output.txt", label_pred_output_host);
  std::cout << "Inference succeeded !" << std::endl << std::flush;

  out_msg.header.frame_id = "base_link";
  out_msg.height = 1;
  out_msg.width = num_voxels;

  sensor_msgs::PointCloud2Modifier modifier(out_msg);
  modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  modifier.resize(out_msg.width * out_msg.height);

  sensor_msgs::PointCloud2Iterator<float> iter_x(out_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(out_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(out_msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_rgb(out_msg, "rgb");

  for (int voxel_idx = 0; voxel_idx < num_voxels; ++voxel_idx) {
    float x = voxelized_feat_host[voxel_idx * 4 + 0];
    float y = voxelized_feat_host[voxel_idx * 4 + 1];
    float z = voxelized_feat_host[voxel_idx * 4 + 2];

    *iter_x = x;
    *iter_y = y;
    *iter_z = z;

    uint8_t r, g, b;

    switch (label_pred_output_host[voxel_idx]) {
      case 0:
        r = 255;
        g = 0;
        b = 0;
        break;
      case 1:
        r = 0;
        g = 255;
        b = 0;
        break;
      case 2:
        r = 0;
        g = 0;
        b = 255;
        break;
      case 3:
        r = 255;
        g = 0;
        b = 255;
        break;
      case 4:
        r = 0;
        g = 255;
        b = 255;
        break;
      case 5:
        r = 255;
        g = 255;
        b = 255;
        break;
      default:
        r = 255;
        g = 255;
        b = 255;
    }

    uint32_t rgb = (r << 16) | (g << 8) | b;
    *iter_rgb = *reinterpret_cast<float *>(&rgb);  // pack into float

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_rgb;
  }

  return true;
}

PTv3TRT::~PTv3TRT()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

void PTv3TRT::initPtr()
{
  // point cloud to voxels
  voxel_features_size_ = config_.max_num_voxels_ * config_.num_point_feature_size_;
  voxel_coords_size_ = 3 * config_.max_num_voxels_;

  // lidar branch
  points_d_ = autoware::cuda_utils::make_unique<float[]>(
    config_.cloud_capacity_ * config_.num_point_feature_size_);

  grid_coord_d_ = autoware::cuda_utils::make_unique<std::int64_t[]>(config_.max_num_voxels_ * 3);
  /* offset_d_ = autoware::cuda_utils::make_unique<std::int64_t>(); */
  feat_d_ = autoware::cuda_utils::make_unique<float[]>(config_.max_num_voxels_ * 4);
  serialized_code_d_ =
    autoware::cuda_utils::make_unique<std::int64_t[]>(config_.max_num_voxels_ * 2);
  /* serialized_order_d_ =
    autoware::cuda_utils::make_unique<std::int64_t[]>(config_.max_num_voxels_ * 2);
  serialized_inverse_d_ =
    autoware::cuda_utils::make_unique<std::int64_t[]>(config_.max_num_voxels_ * 2); */
  label_pred_output_d_ = autoware::cuda_utils::make_unique<std::int64_t[]>(config_.max_num_voxels_);

  pre_ptr_ = std::make_unique<PreprocessCuda>(config_, stream_, true);
  post_ptr_ = std::make_unique<PostprocessCuda>(config_, stream_);
}

void PTv3TRT::initTrt(const tensorrt_common::TrtCommonConfig & trt_config)
{
  std::vector<autoware::tensorrt_common::NetworkIO> network_io;

  // Lidar branch

  // network_io.emplace_back("coord", nvinfer1::Dims{2, {-1, 3}});
  network_io.emplace_back("grid_coord", nvinfer1::Dims{2, {-1, 3}});
  /* network_io.emplace_back("offset", nvinfer1::Dims{1, {1}}); */
  network_io.emplace_back("feat", nvinfer1::Dims{2, {-1, 4}});

  network_io.emplace_back("serialized_code", nvinfer1::Dims{2, {2, -1}});
  /* network_io.emplace_back("serialized_order", nvinfer1::Dims{2, {2, -1}});
  network_io.emplace_back("serialized_inverse", nvinfer1::Dims{2, {2, -1}}); */

  // Outputs
  network_io.emplace_back("seg", nvinfer1::Dims{1, {-1}});

  std::vector<autoware::tensorrt_common::ProfileDims> profile_dims;

  // profile_dims.emplace_back(
  //   "coord", nvinfer1::Dims{2, {config_.voxels_num_[0], 3}},
  //   nvinfer1::Dims{2, {config_.voxels_num_[1], 3}}, nvinfer1::Dims{2, {config_.voxels_num_[2],
  //   3}});

  profile_dims.emplace_back(
    "grid_coord", nvinfer1::Dims{2, {config_.voxels_num_[0], 3}},
    nvinfer1::Dims{2, {config_.voxels_num_[1], 3}}, nvinfer1::Dims{2, {config_.voxels_num_[2], 3}});

  /* profile_dims.emplace_back(
      "offset",
      nvinfer1::Dims{1, {1}},
      nvinfer1::Dims{1, {1}},
      nvinfer1::Dims{1, {1}}); */

  profile_dims.emplace_back(
    "feat", nvinfer1::Dims{2, {config_.voxels_num_[0], 4}},
    nvinfer1::Dims{2, {config_.voxels_num_[1], 4}}, nvinfer1::Dims{2, {config_.voxels_num_[2], 4}});

  profile_dims.emplace_back(
    "serialized_code", nvinfer1::Dims{2, {2, config_.voxels_num_[0]}},
    nvinfer1::Dims{2, {2, config_.voxels_num_[1]}}, nvinfer1::Dims{2, {2, config_.voxels_num_[2]}});

  /* profile_dims.emplace_back(
    "serialized_order", nvinfer1::Dims{2, {2, config_.voxels_num_[0]}},
    nvinfer1::Dims{2, {2, config_.voxels_num_[1]}}, nvinfer1::Dims{2, {2, config_.voxels_num_[2]}});

  profile_dims.emplace_back(
    "serialized_inverse", nvinfer1::Dims{2, {2, config_.voxels_num_[0]}},
    nvinfer1::Dims{2, {2, config_.voxels_num_[1]}}, nvinfer1::Dims{2, {2, config_.voxels_num_[2]}});
*/

  auto network_io_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::NetworkIO>>(network_io);
  auto profile_dims_ptr =
    std::make_unique<std::vector<autoware::tensorrt_common::ProfileDims>>(profile_dims);

  std::cout << "===================================================== 111111111111111111"
            << std::endl
            << std::flush;

  network_trt_ptr_ = std::make_unique<autoware::tensorrt_common::TrtCommon>(
    trt_config, std::make_shared<autoware::tensorrt_common::Profiler>(),
    std::vector<std::string>{config_.plugins_path_});

  std::cout << "===================================================== 22222222222222222"
            << std::endl
            << std::flush;

  if (!network_trt_ptr_->setup(std::move(profile_dims_ptr), std::move(network_io_ptr))) {
    throw std::runtime_error("Failed to setup TRT engine." + config_.plugins_path_);
  }

  std::cout << "===================================================== 333333333333333333"
            << std::endl
            << std::flush;

  // network_trt_ptr_->setTensorAddress("coord", coord_d_.get());
  network_trt_ptr_->setTensorAddress("grid_coord", grid_coord_d_.get());
  /* network_trt_ptr_->setTensorAddress("offset", offset_d_.get()); */
  network_trt_ptr_->setTensorAddress("feat", feat_d_.get());
  network_trt_ptr_->setTensorAddress("serialized_code", serialized_code_d_.get());
  /* network_trt_ptr_->setTensorAddress("serialized_order", serialized_order_d_.get());
  network_trt_ptr_->setTensorAddress("serialized_inverse", serialized_inverse_d_.get()); */

  network_trt_ptr_->setTensorAddress("seg", label_pred_output_d_.get());
}

bool PTv3TRT::segment(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pc_msg,
  [[maybe_unused]] sensor_msgs::msg::PointCloud2 & out_msg,
  std::unordered_map<std::string, double> & proc_timing)
{
  stop_watch_ptr_->toc("processing/inner", true);
  if (!preProcess(pc_msg)) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Pre-process failed. Skipping detection.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/preprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!inference()) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Inference failed. Skipping detection.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/inference_ms", stop_watch_ptr_->toc("processing/inner", true));

  /* if (!postProcess(det_boxes3d)) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Post-process failed. Skipping detection");
    return false;
  } */
  proc_timing.emplace(
    "debug/processing_time/postprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  return true;
}

bool PTv3TRT::preProcess(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pc_msg)
{
  using autoware::cuda_utils::clear_async;

  if (!autoware::point_types::is_data_layout_compatible_with_point_xyzirc(pc_msg->fields)) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Invalid point type. Skipping segmentation.");
    return false;
  }

  if (pc_msg->height * pc_msg->width == 0) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Empty pointcloud. Skipping segmentation.");
    return false;
  }

  // TODO(knzo25): these should be able to be removed as they are filled by TensorRT

  const auto num_points = pc_msg->height * pc_msg->width;

  if (num_points == 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("ptv3"),
      "Empty sweep points (check the capacity of the buffer). Skipping detection.");
    return false;
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  std::int64_t num_voxels = 1;  // TODO(knzo25): remove this line

  /* auto num_voxels = static_cast<std::int64_t>(pre_ptr_->generateVoxels(
    points_d_.get(), num_points, voxel_features_d_.get(), voxel_coords_d_.get(),
    num_points_per_voxel_d_.get())); */

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (num_voxels < config_.min_num_voxels_) {
    RCLCPP_ERROR_STREAM(
      rclcpp::get_logger("ptv3"), "Too few voxels (" << num_voxels
                                                     << ") for the actual optimization profile ("
                                                     << config_.min_num_voxels_ << ")");
    return false;
  }
  if (num_voxels > config_.max_num_voxels_) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("ptv3"), "Actual number of voxels ("
                                    << num_voxels
                                    << ") is over the limit for the actual optimization profile ("
                                    << config_.max_num_voxels_ << "). Clipping to the limit.");
    num_voxels = config_.max_num_voxels_;
  }

  // network_trt_ptr_->setInputShape("coord", nvinfer1::Dims{2, {num_voxels, 3}});
  network_trt_ptr_->setInputShape("grid_coord", nvinfer1::Dims{2, {num_voxels, 3}});
  /* network_trt_ptr_->setInputShape(
    "offset", nvinfer1::Dims{1, {1}}); */
  network_trt_ptr_->setInputShape("feat", nvinfer1::Dims{2, {num_voxels, 4}});
  network_trt_ptr_->setInputShape("serialized_code", nvinfer1::Dims{2, {2, num_voxels}});
  /* network_trt_ptr_->setInputShape("serialized_order", nvinfer1::Dims{2, {2, num_voxels}});
  network_trt_ptr_->setInputShape("serialized_inverse", nvinfer1::Dims{2, {2, num_voxels}}); */

  return true;
}

bool PTv3TRT::inference()
{
  auto status = network_trt_ptr_->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (!status) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "Fail to enqueue and skip to detect.");
    return false;
  }

  return true;
}

bool PTv3TRT::postProcess([[maybe_unused]] sensor_msgs::msg::PointCloud2 & out_msg)
{
  // Here should create a point cloud with the rgb field and copy it to host memory
  /* CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(post_ptr_->generateDetectedBoxes3D_launch(
    label_pred_output_d_.get(), bbox_pred_output_d_.get(), score_output_d_.get(), det_boxes3d,
    stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_)); */
  return true;
}

}  //  namespace autoware::ptv3
