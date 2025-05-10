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

#include "autoware/tensorrt_plugins/custom_unique_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"
#include "autoware/unique_ops/unique.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <chrono>  // TODO(knzo25): delete this
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>  // TODO(knzo25): delete this
#include <functional>
#include <iostream>  // TODO(knzo25): delete this
#include <numeric>   // TODO(knzo25): delete this
#include <string>
#include <thread>  // TODO(knzo25): delete this
#include <tuple>
#include <vector>

namespace nvinfer1::plugin
{

CustomUniquePlugin::CustomUniquePlugin(
  const std::string & name, CustomUniqueParameters const & params)
: layer_name_{name}, params_{params}
{
  initFieldsToSerialize();
}

void CustomUniquePlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * CustomUniquePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild *>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime *>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * CustomUniquePlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new CustomUniquePlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * CustomUniquePlugin::getPluginName() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_NAME;
}

char const * CustomUniquePlugin::getPluginVersion() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_VERSION;
}

char const * CustomUniquePlugin::getPluginNamespace() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_NAMESPACE;
}

std::int32_t CustomUniquePlugin::getNbOutputs() const noexcept
{
  return 4;
  // return 5;
}

std::int32_t CustomUniquePlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  // PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[1].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[2].desc.dims.nbDims == 1);
  // PLUGIN_ASSERT(out[3].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[3].desc.dims.nbDims == 0);

  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[1].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[2].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[3].desc.type == in[0].desc.type);

  return 0;
}

bool CustomUniquePlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  // PLUGIN_ASSERT(num_outputs == 5);

  return (
    in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
    in_out[pos].desc.type == nvinfer1::DataType::kINT64);
}

std::int32_t CustomUniquePlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  // PLUGIN_ASSERT(num_outputs == 5);

  output_types[0] = input_types[0];
  output_types[1] = input_types[0];
  output_types[2] = input_types[0];
  output_types[3] = input_types[0];
  // output_types[4] = input_types[0];

  return 0;
}

std::int32_t CustomUniquePlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs, IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  // PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(inputs[0].nbDims == 1);

  outputs[0].nbDims = 1;
  outputs[0].d[0] = expr_builder.declareSizeTensor(3, *inputs[0].d[0], *inputs[0].d[0]);

  outputs[1] = inputs[0];
  // outputs[2] = inputs[0];

  outputs[2].nbDims = 1;
  outputs[2].d[0] = expr_builder.declareSizeTensor(3, *inputs[0].d[0], *inputs[0].d[0]);

  // num_activate_out
  outputs[3].nbDims = 0;

  return 0;
}

void write_vector_to_text_file(
  std::string const & file_name, const std::vector<std::int64_t> & data)
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

std::int32_t CustomUniquePlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  // sleep 15s
  // std::this_thread::sleep_for(std::chrono::seconds(15));
  /* std::cout << "CustomUniquePlugin::enqueue::start" << std::endl;
  std::cout << "layer_name_: " << layer_name_ << std::endl;

  std::string layer_name2 = layer_name_;
  std::replace(layer_name2.begin(), layer_name2.end(), '/', '_');
  std::transform(layer_name2.begin(), layer_name2.end(), layer_name2.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  std::cout << "layer_name2: " << layer_name2 << std::endl; */

  // const int num_test_samples = 20;

  std::int64_t num_elements = input_desc[0].dims.d[0];

  // std::vector<std::int64_t> input_data(num_elements);
  //  std::vector<std::int64_t> output_indices(num_elements);
  // std::vector<std::int64_t> inverse_indices(num_elements);

  // std::int64_t * input_data_ptr = static_cast<std::int64_t *>(const_cast<void *>(inputs[0]));
  //  std::int64_t * output_indices_ptr = static_cast<std::int64_t *>(outputs[0]);

  // Copy input indices to host
  /* cudaMemcpy(
    input_data.data(), input_data_ptr, num_elements * sizeof(std::int64_t), cudaMemcpyDeviceToHost);

  std::cout << "input_data: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << input_data[i] << ", ";
  }
  std::cout << std::endl; */

  /* write_vector_to_text_file(layer_name2 + "_input_data.txt", input_data); */

  // std::vector<std::int64_t> unique_data;
  // std::vector<std::int64_t> unique_data_counts;
  /* std::vector<std::int64_t> unique_indices_inverse;
  std::vector<std::int64_t> unique_indices_order;
  std::int32_t num_unique_indices = 0;
  std::int32_t num_unique_indices_real = 0; */

  /* std::vector<std::int64_t> input_indices(num_elements);
  std::vector<std::int64_t> sorted_data = input_data;
  std::iota(input_indices.begin(), input_indices.end(), 0);

  std::sort(input_indices.begin(), input_indices.end(), [&input_data](auto i1, auto i2) {
    return input_data[i1] < input_data[i2];
  });

  std::sort(sorted_data.begin(), sorted_data.end());

  unique_data.push_back(sorted_data[0]);
  unique_data_counts.push_back(1);
  inverse_indices[input_indices[0]] = 0;

  for (std::size_t i = 1; i < sorted_data.size(); ++i) {
    if (sorted_data[i] != sorted_data[i - 1]) {
      unique_data.push_back(sorted_data[i]);
      unique_data_counts.push_back(1);
    } else {
      ++unique_data_counts.back();
    }

    int original_index = input_indices[i];
    inverse_indices[original_index] = unique_data.size() - 1;
  } */

  /* std::vector<std::int64_t> sorted_inverse_indices(num_elements);
  std::iota(sorted_inverse_indices.begin(), sorted_inverse_indices.end(), 0);

  std::sort(sorted_inverse_indices.begin(), sorted_inverse_indices.end(),
    [&inverse_indices](auto i1, auto i2) { return inverse_indices[i1] <= inverse_indices[i2]; }); */

  // std::int64_t * output_unique_indices_ptr = reinterpret_cast<std::int64_t *>(outputs[0]);
  // std::int64_t * output_inverse_indices_ptr = reinterpret_cast<std::int64_t *>(outputs[1]);
  /* std::int64_t * output_sorted_inverse_indices_ptr =
    reinterpret_cast<std::int64_t *>(outputs[2]); */
  // std::int64_t * output_unique_indices_counts_ptr =
  //   reinterpret_cast<std::int64_t *>(outputs[2]);
  // std::int64_t * output_num_unique_indices_ptr =
  //   reinterpret_cast<std::int64_t *>(outputs[3]);

  /* std::cout << "unique_data: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << unique_data[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "inverse_indices: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << inverse_indices[i] << ", ";
  }
  std::cout << std::endl; */

  /* std::cout << "sorted_inverse_indices: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << sorted_inverse_indices[i] << ", ";
  }
  std::cout << std::endl; */

  /* std::cout << "unique_indices_counts: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << unique_data_counts[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "num_unique_indices: " << unique_data.size() << std::endl;
  // std::int64_t num_unique_indices = unique_data.size();

  // Cuda start
  if (max_num_elements_ < static_cast<std::size_t>(num_elements)) {
    max_num_elements_ = static_cast<std::size_t>(num_elements);
    workspace_size_ = get_unique_workspace_size(max_num_elements_);
  } */

  std::int64_t num_unique_elements = unique(
    reinterpret_cast<const std::int64_t *>(inputs[0]), reinterpret_cast<std::int64_t *>(outputs[0]),
    reinterpret_cast<std::int64_t *>(outputs[1]), reinterpret_cast<std::int64_t *>(outputs[2]),
    workspace, num_elements, workspace_size_, stream);
  // cuda end

  /* cudaMemcpyAsync(
    output_unique_indices_ptr, unique_data.data(),
    unique_data.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    output_inverse_indices_ptr, inverse_indices.data(),
    inverse_indices.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    output_sorted_inverse_indices_ptr, sorted_inverse_indices.data(),
    sorted_inverse_indices.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    output_unique_indices_counts_ptr, unique_data_counts.data(),
    unique_data_counts.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    output_num_unique_indices_ptr, &num_unique_indices, sizeof(std::int64_t),
    cudaMemcpyHostToDevice, stream); */

  cudaMemcpyAsync(
    reinterpret_cast<std::int64_t *>(outputs[3]), &num_unique_elements, sizeof(std::int64_t),
    cudaMemcpyHostToDevice, stream);

  /* write_vector_to_text_file(layer_name2 + "_unique_data.txt", unique_data);
  write_vector_to_text_file(
    layer_name2 + "_inverse_indices.txt", inverse_indices);
  write_vector_to_text_file(
    layer_name2 + "_sorted_inverse_indices.txt", sorted_inverse_indices);
  write_vector_to_text_file(
    layer_name2 + "_unique_indices_counts.txt", unique_data_counts); */

  (void)input_desc;
  (void)output_desc;
  (void)inputs;
  (void)outputs;
  (void)workspace;
  (void)stream;

  /* std::cout << "CustomUniquePlugin::enqueue::end" << std::endl; */

  return 0;
}

std::int32_t CustomUniquePlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * CustomUniquePlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * CustomUniquePlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t CustomUniquePlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return get_unique_workspace_size(inputs[0].max.d[0]);
}

}  // namespace nvinfer1::plugin
