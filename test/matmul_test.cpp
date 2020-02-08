#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/math/matmul.h"
// #include "src/tool/tool.h"
}

void natural_assign(Tensor t) {
  int64_t ndim = aitisa_tensor_ndim(t);
  int64_t *dims = aitisa_tensor_dims(t);
  int64_t size = aitisa_tensor_size(t);
  float *data = (float *)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Matmul, vector_vector) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {1};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[1] = {0.14};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, matrix_vector) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[2] = {5, 4};
  int64_t dims2[1] = {4};
  aitisa_full(dtype, device, dims1, 2, 1.4, &tensor1);
  aitisa_full(dtype, device, dims2, 1, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer2d(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {5};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[5] = {0.14, 0.38, 0.62, 0.86, 1.10};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, matrix_matrix) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[2] = {5, 4};
  int64_t dims2[2] = {4, 3};
  aitisa_full(dtype, device, dims1, 2, 2.1, &tensor1);
  aitisa_full(dtype, device, dims2, 2, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer2d(tensor1);
  tensor_printer2d(tensor2);
  tensor_printer2d(output);
  */
  int64_t expected_ndim = 2;
  int64_t expected_dims[2] = {5, 3};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[15] = {0.42, 0.48, 0.54, 1.14, 1.36, 1.58, 1.86, 2.24,
                      2.62, 2.58, 3.12, 3.66, 3.3,  4,    4.7};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, vector_matrix) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[1] = {5};
  int64_t dims2[2] = {5, 4};
  aitisa_full(dtype, device, dims1, 1, 1.4, &tensor1);
  aitisa_full(dtype, device, dims2, 2, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer2d(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[4] = {1.2, 1.3, 1.4, 1.5};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, cube_vector) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[3] = {5, 4, 3};
  int64_t dims2[1] = {3};
  aitisa_full(dtype, device, dims1, 3, 1.4, &tensor1);
  aitisa_full(dtype, device, dims2, 1, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer2d(tensor1);
  tensor_printer(tensor2);
  tensor_printer2d(output);
  */
  int64_t expected_ndim = 2;
  int64_t expected_dims[2] = {5, 4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[20] = {0.05, 0.14, 0.23, 0.32, 0.41, 0.5,  0.59,
                      0.68, 0.77, 0.86, 0.95, 1.04, 1.13, 1.22,
                      1.31, 1.4,  1.49, 1.58, 1.67, 1.76};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, vector_cube) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[1] = {3};
  int64_t dims2[3] = {5, 3, 4};
  aitisa_full(dtype, device, dims1, 1, 1.4, &tensor1);
  aitisa_full(dtype, device, dims2, 3, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer2d(tensor2);
  tensor_printer2d(output);
  */
  int64_t expected_ndim = 2;
  int64_t expected_dims[2] = {5, 4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[20] = {0.2,  0.23, 0.26, 0.29, 0.56, 0.59, 0.62,
                      0.65, 0.92, 0.95, 0.98, 1.01, 1.28, 1.31,
                      1.34, 1.37, 1.64, 1.67, 1.7,  1.73};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Matmul, tensor_tensor) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims1[4] = {2, 1, 3, 4};
  int64_t dims2[3] = {2, 4, 2};
  aitisa_full(dtype, device, dims1, 4, 1.4, &tensor1);
  aitisa_full(dtype, device, dims2, 3, 2.0, &tensor2);
  natural_assign(tensor1);
  natural_assign(tensor2);
  Tensor output;
  aitisa_matmul(tensor1, tensor2, &output);
  /*
  tensor_printer2d(tensor1);
  tensor_printer2d(tensor2);
  tensor_printer2d(output);
  */
  int64_t expected_ndim = 4;
  int64_t expected_dims[4] = {2, 2, 3, 2};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  float *data = (float *)aitisa_tensor_data(output);
  float result[24] = {0.28, 0.34, 0.76, 0.98, 1.24, 1.62, 0.76, 0.82,
                      2.52, 2.74, 4.28, 4.66, 1.72, 2.26, 2.2,  2.9,
                      2.68, 3.54, 6.04, 6.58, 7.8,  8.5,  9.56, 10.42};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

// TEST(Matmul, ttt) {
//   Tensor tensor1;
//   DataType dtype = kFloat;
//   Device device = {DEVICE_CPU, 0};
//   int64_t dims1[2] = {5, 4};
//   aitisa_full(dtype, device, dims1, 2, 2.1, &tensor1);
//   Shape shape = aitisa_tensor_shape(tensor1);
//   printf("old_ndim = %ld\n", aitisa_tensor_ndim(tensor1));
//   shape.ndim = 10;
//   printf("new_ndim = %ld\n", aitisa_tensor_ndim(tensor1));
// }

}  // namespace
}  // namespace aitisa_api