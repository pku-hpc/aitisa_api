#include <math.h>
#include <vector>
#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/nn/softmax.h"
//#include "src/tool/tool.h"
}

void quadratic_assign_float(Tensor t) {
  int64_t ndim = aitisa_tensor_ndim(t);
  int64_t* dims = aitisa_tensor_dims(t);
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * i * 0.1 + i * 0.1;
    data[i] = value;
  }
}

void quadratic_assign_double(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  double* data = (double*)aitisa_tensor_data(t);
  double value = 0;
  for (int i = 0; i < size; ++i) {
    value = value = i * i * 0.002 + i * 0.3;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Softmax, FloatAll) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  quadratic_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_softmax(input, -1, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.02870, 0.03506, 0.05230, 0.09530, 0.21210, 0.57654};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
                   are equal when their difference is less than 0.00001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Softmax, DoubleAxis1) {
  Tensor input;
  DataType dtype = kDouble;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {2, 3, 5};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  quadratic_assign_double(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_softmax(input, 1, &output);
  // tensor_printer2d(output);

  double* out_data = (double*)aitisa_tensor_data(output);

  double test_data[] = {0.033064, 0.031908, 0.030789, 0.029706, 0.028659,
                        0.155782, 0.153369, 0.150980, 0.148614, 0.146272,
                        0.811154, 0.814723, 0.818232, 0.821680, 0.825069,
                        0.019208, 0.018514, 0.017843, 0.017196, 0.016571,
                        0.122161, 0.120123, 0.118111, 0.116125, 0.114165,
                        0.858631, 0.861363, 0.864046, 0.866679, 0.869264};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
             are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api