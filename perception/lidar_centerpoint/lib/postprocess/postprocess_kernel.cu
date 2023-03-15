// Copyright 2022 TIER IV, Inc.
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

#include "lidar_centerpoint/postprocess/circle_nms_kernel.hpp"

#include <lidar_centerpoint/postprocess/postprocess_kernel.hpp>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace
{
const std::size_t THREADS_PER_BLOCK = 32;
}  // namespace

namespace centerpoint
{

struct is_score_greater
{
  is_score_greater(float t) : t_(t) {}

  __device__ bool operator()(const Box3D & b) {
    //if (iteration_debug_++ < 10) {
    //  printf("is_score_greater iteration %d, box.score=%.2f threshold=%.2f\n", iteration_debug_, b.score, t_);
    //}
     
    return b.score > t_; 
    }

private:
  float t_{0.0};
  int iteration_debug_{0};
};

struct is_score_greater_cpu
{
  is_score_greater_cpu(float t) : t_(t) {}

  __host__ bool operator()(const Box3D & b) {
    //if (iteration_debug_++ < 10) {
    //  printf("is_score_greater iteration %d, box.score=%.2f threshold=%.2f\n", iteration_debug_, b.score, t_);
    //}
     
    return b.score > t_; 
    }

private:
  float t_{0.0};
  int iteration_debug_{0};
};

struct is_kept
{
  __device__ bool operator()(const bool keep) { return keep; }
};

struct score_greater
{
  __device__ bool operator()(const Box3D & lb, const Box3D & rb) { return lb.score > rb.score; }
};

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void generateBoxes3D_kernel(
  const float * out_heatmap, const float * out_offset, const float * out_z, const float * out_dim,
  const float * out_rot, const float * out_vel, const float voxel_size_x, const float voxel_size_y,
  const float range_min_x, const float range_min_y, const std::size_t down_grid_size_x,
  const std::size_t down_grid_size_y, const std::size_t downsample_factor, const int class_size,
  const float * yaw_norm_thresholds, Box3D * det_boxes3d)
{
  // generate boxes3d from the outputs of the network.
  // shape of out_*: (N, DOWN_GRID_SIZE_Y, DOWN_GRID_SIZE_X)
  // heatmap: N = class_size, offset: N = 2, z: N = 1, dim: N = 3, rot: N = 2, vel: N = 2
  const auto yi = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  const auto xi = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
  const auto idx = down_grid_size_x * yi + xi;
  const auto down_grid_size = down_grid_size_y * down_grid_size_x;

  if (yi >= down_grid_size_y || xi >= down_grid_size_x) {
    return;
  }

  int label = -1;
  float max_score = -1;
  for (int ci = 0; ci < class_size; ci++) {
    float score = sigmoid(out_heatmap[down_grid_size * ci + idx]);
    if (score > max_score) {
      label = ci;
      max_score = score;
    }
  }

  const float offset_x = out_offset[down_grid_size * 0 + idx];
  const float offset_y = out_offset[down_grid_size * 1 + idx];
  const float x = voxel_size_x * downsample_factor * (xi + offset_x) + range_min_x;
  const float y = voxel_size_y * downsample_factor * (yi + offset_y) + range_min_y;
  const float z = out_z[idx];
  const float w = out_dim[down_grid_size * 0 + idx];
  const float l = out_dim[down_grid_size * 1 + idx];
  const float h = out_dim[down_grid_size * 2 + idx];
  const float yaw_sin = out_rot[down_grid_size * 0 + idx];
  const float yaw_cos = out_rot[down_grid_size * 1 + idx];
  const float yaw_norm = sqrtf(yaw_sin * yaw_sin + yaw_cos * yaw_cos);
  const float vel_x = out_vel[down_grid_size * 0 + idx];
  const float vel_y = out_vel[down_grid_size * 1 + idx];

  det_boxes3d[idx].label = label;
  det_boxes3d[idx].score = yaw_norm >= yaw_norm_thresholds[label] ? max_score : 0.f;
  det_boxes3d[idx].x = x;
  det_boxes3d[idx].y = y;
  det_boxes3d[idx].z = z;
  det_boxes3d[idx].length = expf(l);
  det_boxes3d[idx].width = expf(w);
  det_boxes3d[idx].height = expf(h);
  det_boxes3d[idx].yaw = atan2f(yaw_sin, yaw_cos);
  det_boxes3d[idx].vel_x = vel_x;
  det_boxes3d[idx].vel_y = vel_y;
}

PostProcessCUDA::PostProcessCUDA(const CenterPointConfig & config) : config_(config)
{
  const auto num_raw_boxes3d = config.down_grid_size_y_ * config.down_grid_size_x_;
  boxes3d_d_ = thrust::device_vector<Box3D>(num_raw_boxes3d);
  yaw_norm_thresholds_d_ = thrust::device_vector<float>(
    config_.yaw_norm_thresholds_.begin(), config_.yaw_norm_thresholds_.end());
}

// cspell: ignore divup
cudaError_t PostProcessCUDA::generateDetectedBoxes3D_launch(
  const float * out_heatmap, const float * out_offset, const float * out_z, const float * out_dim,
  const float * out_rot, const float * out_vel, std::vector<Box3D> & det_boxes3d,
  cudaStream_t stream)
{
  uintptr_t out_heatmap_alignment = ((uintptr_t)(const void *)(out_heatmap)) % sizeof(float);
  uintptr_t out_offset_alignment = ((uintptr_t)(const void *)(out_offset)) % sizeof(float);
  uintptr_t out_z_alignment = ((uintptr_t)(const void *)(out_z)) % sizeof(float);
  uintptr_t out_dim_alignment = ((uintptr_t)(const void *)(out_dim)) % sizeof(float);
  uintptr_t out_rot_alignment = ((uintptr_t)(const void *)(out_rot)) % sizeof(float);
  uintptr_t out_vel_alignment = ((uintptr_t)(const void *)(out_vel)) % sizeof(float);
  uintptr_t yaw_norm_thresholds_d_alignment = ((uintptr_t)(const void *)(thrust::raw_pointer_cast(boxes3d_d_.data()))) % sizeof(float);
  uintptr_t boxes3d_d_alignment = ((uintptr_t)(const void *)(thrust::raw_pointer_cast(boxes3d_d_.data()))) % sizeof(centerpoint::Box3D);
  uintptr_t boxes3d_d_alignment512 = ((uintptr_t)(const void *)(thrust::raw_pointer_cast(boxes3d_d_.data()))) % (16*sizeof(float));

  printf("out_heatmap_alignment: %lu\n", out_heatmap_alignment);
  printf("out_offset_alignment: %lu\n", out_offset_alignment);
  printf("out_z_alignment: %lu\n", out_z_alignment);
  printf("out_dim_alignment: %lu\n", out_dim_alignment);
  printf("out_rot_alignment: %lu\n", out_rot_alignment);
  printf("out_vel_alignment: %lu\n", out_vel_alignment);
  printf("yaw_norm_thresholds_d_alignment: %lu\n", yaw_norm_thresholds_d_alignment);
  printf("boxes3d_d_alignment: %lu size=%lu\n", boxes3d_d_alignment, sizeof(centerpoint::Box3D));
  printf("boxes3d_d_alignment512: %lu size=%lu\n", boxes3d_d_alignment512, sizeof(centerpoint::Box3D));

  int min_label = 100;
  float min_score = 1e10;
  float min_x = 1e10;
  float min_y = 1e10;
  float min_z = 1e10;
  float min_length = 1e10;
  float min_width = 1e10;
  float min_height = 1e10;
  float min_vx = 1e10;
  float min_vy = 1e10;

  int max_label = -100;
  float max_score = -1e10;
  float max_x = -1e10;
  float max_y = -1e10;
  float max_z = -1e10;
  float max_length = -1e10;
  float max_width = -1e10;
  float max_height = -1e10;
  float max_vx = -1e10;
  float max_vy = -1e10;
 
  dim3 blocks(
    divup(config_.down_grid_size_y_, THREADS_PER_BLOCK),
    divup(config_.down_grid_size_x_, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  generateBoxes3D_kernel<<<blocks, threads, 0, stream>>>(
    out_heatmap, out_offset, out_z, out_dim, out_rot, out_vel, config_.voxel_size_x_,
    config_.voxel_size_y_, config_.range_min_x_, config_.range_min_y_, config_.down_grid_size_x_,
    config_.down_grid_size_y_, config_.downsample_factor_, config_.class_size_,
    thrust::raw_pointer_cast(yaw_norm_thresholds_d_.data()),
    thrust::raw_pointer_cast(boxes3d_d_.data()));

  cudaDeviceSynchronize();

  thrust::host_vector<centerpoint::Box3D> host_vec = boxes3d_d_;
  centerpoint::Box3D* bbox_ptr = thrust::raw_pointer_cast(host_vec.data());
  for(unsigned long i = 0; i < boxes3d_d_.size(); i++) {

    min_label = (min_label < bbox_ptr[i].label) ? min_label : bbox_ptr[i].label;
    min_score = (min_score < bbox_ptr[i].score) ? min_score : bbox_ptr[i].score;
    min_x = (min_x < bbox_ptr[i].x) ? min_x : bbox_ptr[i].x;
    min_y = (min_y < bbox_ptr[i].y) ? min_y : bbox_ptr[i].y;
    min_z = (min_z < bbox_ptr[i].z) ? min_z : bbox_ptr[i].z;
    min_length = (min_length < bbox_ptr[i].length) ? min_length : bbox_ptr[i].length;
    min_width = (min_width < bbox_ptr[i].width) ? min_width : bbox_ptr[i].width;
    min_height = (min_height < bbox_ptr[i].height) ? min_height : bbox_ptr[i].height;
    min_vx = (min_vx < bbox_ptr[i].vel_x) ? min_vx : bbox_ptr[i].vel_x;
    min_vy = (min_vy < bbox_ptr[i].vel_y) ? min_vy : bbox_ptr[i].vel_y;

    max_label = (max_label > bbox_ptr[i].label) ? max_label : bbox_ptr[i].label;
    max_score = (max_score > bbox_ptr[i].score) ? max_score : bbox_ptr[i].score;
    max_x = (max_x > bbox_ptr[i].x) ? max_x : bbox_ptr[i].x;
    max_y = (max_y > bbox_ptr[i].y) ? max_y : bbox_ptr[i].y;
    max_z = (max_z > bbox_ptr[i].z) ? max_z : bbox_ptr[i].z;
    max_length = (max_length > bbox_ptr[i].length) ? max_length : bbox_ptr[i].length;
    max_width = (max_width > bbox_ptr[i].width) ? max_width : bbox_ptr[i].width;
    max_height = (min_height > bbox_ptr[i].height) ? max_height : bbox_ptr[i].height;
    max_vx = (max_vx > bbox_ptr[i].vel_x) ? max_vx : bbox_ptr[i].vel_x;
    max_vy = (max_vy > bbox_ptr[i].vel_y) ? max_vy : bbox_ptr[i].vel_y;
  }

  printf("label min=%d \t max=%d\n", min_label, max_label);
  printf("score min=%.2f \t max=%.2f\n", min_score, max_score);

  printf("x min=%.2f \t max=%.2f\n", min_x, max_x);
  printf("y min=%.2f \t max=%.2f\n", min_y, max_y);
  printf("z min=%.2f \t max=%.2f\n", min_z, max_z);

  printf("length min=%.2f \t max=%.2f\n", min_length, max_length);
  printf("width min=%.2f \t max=%.2f\n", min_width, max_width);
  printf("height min=%.2f \t max=%.2f\n", min_height, max_height);

  printf("vx min=%.2f \t max=%.2f\n", min_vx, max_vx);
  printf("vy min=%.2f \t max=%.2f\n", min_vy, max_vy);

  // suppress by score (CPU)
  const auto num_det_boxes3d_cpu = thrust::count_if(
    thrust::host, host_vec.begin(), host_vec.end(),
    is_score_greater_cpu(config_.score_threshold_));

  printf("Num boxes over thresh (CPU)=%lu\n", num_det_boxes3d_cpu);


  // suppress by score
  const auto num_det_boxes3d = thrust::count_if(
    thrust::device, boxes3d_d_.begin(), boxes3d_d_.end(),
    is_score_greater(config_.score_threshold_));

  printf("Num boxes over thresh (GPU)=%lu\n", num_det_boxes3d);

  cudaDeviceSynchronize();

  if (num_det_boxes3d == 0) {
    return cudaGetLastError();
  }
  thrust::device_vector<Box3D> det_boxes3d_d(num_det_boxes3d);
  thrust::copy_if(
    thrust::device, boxes3d_d_.begin(), boxes3d_d_.end(), det_boxes3d_d.begin(),
    is_score_greater(config_.score_threshold_));

  // sort by score
  thrust::sort(det_boxes3d_d.begin(), det_boxes3d_d.end(), score_greater());

  // supress by NMS
  thrust::device_vector<bool> final_keep_mask_d(num_det_boxes3d);
  const auto num_final_det_boxes3d =
    circleNMS(det_boxes3d_d, config_.circle_nms_dist_threshold_, final_keep_mask_d, stream);

  thrust::device_vector<Box3D> final_det_boxes3d_d(num_final_det_boxes3d);
  thrust::copy_if(
    thrust::device, det_boxes3d_d.begin(), det_boxes3d_d.end(), final_keep_mask_d.begin(),
    final_det_boxes3d_d.begin(), is_kept());

  // memcpy device to host
  det_boxes3d.resize(num_final_det_boxes3d);
  thrust::copy(final_det_boxes3d_d.begin(), final_det_boxes3d_d.end(), det_boxes3d.begin());

  return cudaGetLastError();
}

}  // namespace centerpoint
