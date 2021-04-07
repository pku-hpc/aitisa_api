#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/conv.h"
//#include "src/tool/tool.h"
}

namespace aitisa_api {
namespace {

TEST(Conv2D, P0S1D1) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 5, 5};
  int64_t filter_dims[4] = {3, 2, 3, 3};
  aitisa_full(dtype, device, input_dims, 4, 1.0, &input);
  aitisa_full(dtype, device, filter_dims, 4, 1.0, &filter);

  int padding[2] = {0, 0};
  int stride[2] = {1, 1};
  int dilation[2] = {1, 1};
  int groups = 1;
  int ndim = aitisa_tensor_ndim(input);

  aitisa_conv2d(input, filter, stride, 2, padding, 2, 
                dilation, 2, groups, &output);
  /*
  tensor_printer2d(input);
  tensor_printer2d(filter);
  tensor_printer2d(output);
  */
  // int64_t *output_dims = aitisa_tensor_dims(output);
  // for (int i = 0; i < ndim; ++i) {
  //  std::cout << output_dims[i] << ", ";
  //}
  // std::cout << std::endl;

  // float *input_data = (float *)aitisa_tensor_data(input);
  // int64_t input_size = aitisa_tensor_size(input);
  // for (int i = 0; i < input_size; ++i) {
  //  std::cout << input_data[i] << ", ";
  //}
  // std::cout << std::endl;

  // float *filter_data = (float *)aitisa_tensor_data(filter);
  // int64_t filter_size = aitisa_tensor_size(filter);
  // for (int i = 0; i < filter_size; ++i) {
  //  std::cout << filter_data[i] << ", ";
  //}
  // std::cout << std::endl;

  float *output_data = (float *)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
    // std::cout << output_data[i] << ", ";
    EXPECT_EQ(18.0, output_data[i]);
  }
  // std::cout << std::endl;

  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}

TEST(Conv2D, P2S1D1) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {1, 1, 5, 5};
  int64_t filter_dims[4] = {1, 1, 2, 2};
  aitisa_full(dtype, device, input_dims, 4, 1.0, &input);
  aitisa_full(dtype, device, filter_dims, 4, 1.0, &filter);

  int padding[2] = {1, 1};
  int stride[2] = {2, 2};
  int dilation[2] = {2, 2};
  int groups = 1;
  int ndim = aitisa_tensor_ndim(input);

  aitisa_conv2d(input, filter, stride, 2, padding, 2, 
              dilation, 2, groups, &output);
  /*
  tensor_printer2d(input);
  tensor_printer2d(filter);
  tensor_printer2d(output);
  */
  float *output_data = (float *)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  ASSERT_THAT(std::vector<float>(output_data, output_data + output_size),
              ::testing::ElementsAre(1, 2, 1, 2, 4, 2, 1, 2, 1));

  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}

TEST(Conv3D, P0S1D1) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[5] = {2, 2, 5, 5, 5};
  int64_t filter_dims[5] = {3, 2, 3, 3, 3};
  aitisa_full(dtype, device, input_dims, 5, 1.0, &input);
  aitisa_full(dtype, device, filter_dims, 5, 1.0, &filter);

  int padding[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  int dilation[3] = {1, 1, 1};
  int groups = 1;
  int ndim = aitisa_tensor_ndim(input);

  aitisa_conv3d(input, filter, stride, 3, padding, 3, 
                dilation, 3, groups, &output);
  /*
  tensor_printer2d(input);
  tensor_printer2d(filter);
  tensor_printer2d(output);
  */
  int64_t *output_dims = aitisa_tensor_dims(output);
  // for (int i = 0; i < ndim; ++i) {
  //  std::cout << output_dims[i] << ", ";
  //}
  // std::cout << std::endl;

  // float *input_data = (float *)aitisa_tensor_data(input);
  // int64_t input_size = aitisa_tensor_size(input);
  // std::cout << "input_data" << std::endl;
  // for (int i = 0; i < input_size; ++i) {
  //  std::cout << input_data[i] << ", ";
  //}
  // std::cout << std::endl;

  // float *filter_data = (float *)aitisa_tensor_data(filter);
  // int64_t filter_size = aitisa_tensor_size(filter);
  // std::cout << "filter_data" << std::endl;
  // for (int i = 0; i < filter_size; ++i) {
  //  std::cout << filter_data[i] << ", ";
  //}
  // std::cout << std::endl;

  float *output_data = (float *)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  // std::cout << "output_data" << std::endl;
  for (int i = 0; i < output_size; ++i) {
    //  std::cout << output_data[i] << ", ";
    EXPECT_EQ(54.0, output_data[i]);
  }
  // std::cout << std::endl;

  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api