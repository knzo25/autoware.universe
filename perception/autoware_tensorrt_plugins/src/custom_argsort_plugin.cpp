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

#include "autoware/tensorrt_plugins/custom_argsort_plugin.hpp"

#include "autoware/argsort_ops/argsort.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cassert>  // TODO(knzo25): delete this
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>  // TODO(knzo25): delete this
#include <numeric>   // TODO(knzo25): delete this
#include <string>
#include <tuple>
#include <vector>

namespace nvinfer1::plugin
{

CustomArgsortPlugin::CustomArgsortPlugin(
  const std::string & name, CustomArgsortParameters const & params)
: layer_name_{name}, params_{params}
{
  initFieldsToSerialize();
}

void CustomArgsortPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * CustomArgsortPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 * CustomArgsortPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new CustomArgsortPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * CustomArgsortPlugin::getPluginName() const noexcept
{
  return kCUSTOM_ARGSORT_PLUGIN_NAME;
}

char const * CustomArgsortPlugin::getPluginVersion() const noexcept
{
  return kCUSTOM_ARGSORT_PLUGIN_VERSION;
}

char const * CustomArgsortPlugin::getPluginNamespace() const noexcept
{
  return kCUSTOM_ARGSORT_PLUGIN_NAMESPACE;
}

std::int32_t CustomArgsortPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t CustomArgsortPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 1);

  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);

  return 0;
}

bool CustomArgsortPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 1);

  return (
    in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
    in_out[pos].desc.type == nvinfer1::DataType::kINT64);
}

std::int32_t CustomArgsortPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 1);

  output_types[0] = input_types[0];

  return 0;
}

std::int32_t CustomArgsortPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 1);

  outputs[0] = inputs[0];

  return 0;
}

std::int32_t CustomArgsortPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  // const int num_test_samples = 20;
  /* std::cout << "CustomArgsortPlugin::enqueue::start" << std::endl; */

  auto num_elements = static_cast<std::size_t>(input_desc[0].dims.d[0]);
  if (max_num_elements_ < num_elements) {
    max_num_elements_ = num_elements;
    argsort_workspace_size_ = get_argsort_workspace_size(max_num_elements_);
  }

  // cuda version start
  argsort(
    reinterpret_cast<std::int64_t const *>(inputs[0]), reinterpret_cast<std::int64_t *>(outputs[0]),
    workspace, num_elements, argsort_workspace_size_, stream);

  /* std::vector<std::int64_t> cuda_result_host(num_elements);
  cudaMemcpyAsync(
    cuda_result_host.data(), outputs[0], num_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << "cuda_result_host: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << cuda_result_host[i] << ", ";
  }
  std::cout << std::endl; */

  // cuda version end

  // std::int64_t num_elements = input_desc[0].dims.d[0];
  // std::int64_t const * input_data_device_ptr = reinterpret_cast<std::int64_t const *>(inputs[0]);
  // std::int64_t * output_indices_device_ptr = reinterpret_cast<std::int64_t *>(outputs[0]);

  /* std::vector<std::int64_t> input_data_host(num_elements);
  std::vector<std::int64_t> output_indices_host(num_elements);

  std::vector<std::int64_t> indices(num_elements);
  std::iota(indices.begin(), indices.end(), 0);

  cudaMemcpyAsync(
    input_data_host.data(), input_data_device_ptr, num_elements * sizeof(std::int64_t),
    cudaMemcpyDeviceToHost);

  std::cout << "input_data: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << input_data_host[i] << ", ";
  }
  std::cout << std::endl;

  std::sort(indices.begin(), indices.end(), [&input_data_host](auto i1, auto i2) {
    return input_data_host[i1] < input_data_host[i2];
  });

  std::cout << "output_data: " << std::endl;
  for (std::int32_t i = 0; i < num_test_samples; ++i) {
    std::cout << indices[i] << ", ";
  }
  std::cout << std::endl;

  int different_elements = 0;
  for (std::size_t i = 0; i < num_elements; ++i) {
    if (indices[i] != cuda_result_host[i]) {
      // std::cout << "different_elements: " << indices[i] << ", " << cuda_result_host[i] <<
      // std::endl;
      different_elements++;
      assert(input_data_host[indices[i]] == input_data_host[cuda_result_host[i]]);
    }
  }

  cudaMemcpyAsync(
    output_indices_device_ptr, indices.data(), num_elements * sizeof(std::int64_t),
    cudaMemcpyHostToDevice); */

  (void)input_desc;
  (void)output_desc;
  (void)inputs;
  (void)outputs;
  (void)workspace;
  (void)stream;

  /* std::cout << "CustomArgsortPlugin::enqueue::end" << std::endl; */

  return 0;
}

std::int32_t CustomArgsortPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * CustomArgsortPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * CustomArgsortPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t CustomArgsortPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  int64_t max_num_elements = inputs[0].max.d[0];
  return get_argsort_workspace_size(max_num_elements) +
         sizeof(std::int64_t) * 2 * (max_num_elements + 1);
}

}  // namespace nvinfer1::plugin
