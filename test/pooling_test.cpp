#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <vector>
#include <math.h>

extern "C"{
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/nn/pooling.h"
}

void natural_assign_float(Tensor t) {
	int64_t ndim = aitisa_tensor_ndim(t);
	int64_t* dims = aitisa_tensor_dims(t);
	int64_t size = aitisa_tensor_size(t);
	float* data = (float*)aitisa_tensor_data(t);
	float value = 0;
	for (int i = 0; i < size; ++i) {
		value = i * 0.1;
		data[i] = value;
	}
}

void natural_assign_double(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  double *data = (double *)aitisa_tensor_data(t);
  double value = 0;
  for (int i = 0; i < size; ++i) {
	value = i * 0.1;
	data[i] = value;
  }
}

void natural_assign_int32(Tensor t) {
	int64_t size = aitisa_tensor_size(t);
	int32_t* data = (int32_t *)aitisa_tensor_data(t);
	int32_t value = 0;
	for (int i = 0; i < size; ++i) {
		data[i] = i;
	}
}

namespace aitisa_api {
namespace {

TEST(Pooling1D, MaxFloatK2S3P0D2) {
	Tensor input;
	DataType dtype = { TYPE_FLOAT, sizeof(float) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[3] = { 2, 3, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &input);
	natural_assign_float(input);
	//tensor_printer2d(input);

	Tensor output;
	int ksize[1] = { 2 };
	int stride[1] = { 2 };
	int padding[1] = { 0 };
	int dilation[1] = { 2 };
	aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
	//tensor_printer2d(output);

	float* output_data = (float*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	float test_data[] = { 0.2, 0.4, 0.6, 0.8,
									                 1.1, 1.3, 1.5, 1.7,
									                 2.0, 2.2, 2.4, 2.6,
												                2.9, 3.1, 3.3, 3.5,
												                3.8, 4.0, 4.2, 4.4,
												                4.7, 4.9, 5.1, 5.3};

	for (int i = 0; i < output_size; i++) {
		// Due to the problem of precision, consider the two numbers are equal when their difference is less than 0.0001
		EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
	}
}

TEST(Pooling1D, AvgFloatK2S2P1D1) {
	Tensor input;
	DataType dtype = { TYPE_FLOAT, sizeof(float) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[3] = { 2, 3, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &input);
	natural_assign_float(input);
	//tensor_printer2d(input);

	Tensor output;
	int ksize[1] = { 2 };
	int stride[1] = { 2 };
	int padding[1] = { 1 };
	int dilation[1] = { 1 };
	aitisa_pooling(input, "avg", ksize, stride, padding, dilation, &output);
	//tensor_printer2d(output);
	
	float* output_data = (float*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	float test_data[] = { 0.05, 0.25, 0.45, 0.65, 0.4,
									                0.95, 1.15, 1.35, 1.55, 0.85,
									                1.85, 2.05, 2.25, 2.45, 1.3,
											                  2.75, 2.95, 3.15, 3.35, 1.75,
											                  3.65, 3.85, 4.05, 4.25, 2.2,
											                  4.55, 4.75, 4.95, 5.15, 2.65};

	for (int i = 0; i < output_size; i++) {
		// Due to the problem of precision, consider the two numbers are equal when their difference is less than 0.0001
		EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
	}
}

TEST(Pooling2D, AvgInt32K3S3P0D1) {
	Tensor input;
	DataType dtype = { TYPE_INT32, sizeof(int32_t) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[4] = {1, 2, 9, 9};
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 4, &input);
	natural_assign_int32(input);

	Tensor output;
	int ksize[2] = {3, 3};
	int stride[2] = {3, 3};
	int padding[2] = {0, 0};
	int dilation[2] = { 1, 1 };
	aitisa_pooling(input, "avg", ksize, stride, padding, dilation,&output);
	
	int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
  int32_t test_data[] = { 10,  13,  16,  37,  40,  43,  64,  67,  70,
                  91,  94,  97, 118, 121, 124, 145, 148, 151 };
  for (int i = 0; i < output_size; i++) {
    EXPECT_TRUE(output_data[i] == test_data[i]);
  }
}

TEST(Pooling2D, MaxDoubleK4S4P3D1) {
	Tensor input;
	DataType dtype = { TYPE_DOUBLE, sizeof(double) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[4] = { 1, 2, 9, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 4, &input);
	natural_assign_double(input);
	//tensor_printer2d(input);

	Tensor output;
	int ksize[2] = { 4, 4 };
	int stride[2] = { 4, 4 };
	int padding[2] = { 3, 3 };
	int dilation[2] = { 1, 1 };
	aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
	//tensor_printer2d(output);

	double* output_data = (double*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	double test_data[] = { 2,  2.4,  2.6,  5.6,  6.0,  6.2,  7.4,  7.8,  8.0,
								   10.1, 10.5, 10.7, 13.7, 14.1, 14.3, 15.5, 15.9, 16.1 };
	
	for (int i = 0; i < output_size; i++) {
		// Due to the problem of precision, consider the two numbers are equal when their difference is less than 0.0001
		EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
	}
}

TEST(Pooling3D, MaxInt32K4S4P3D2) {
	Tensor input;
	DataType dtype = { TYPE_INT32, sizeof(int32_t) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[5] = { 1, 2, 5, 10, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 5, &input);
	natural_assign_int32(input);
	//tensor_printer2d(input);

	Tensor output;
	int ksize[3] = { 3, 3, 3 };
	int stride[3] = { 5, 5, 5 };
	int padding[3] = { 0, 0, 1 };
	int dilation[3] = { 2, 2, 2 };
	aitisa_pooling(input, "max", ksize, stride, padding, dilation, &output);
	//tensor_printer2d(output);
	
	int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	int32_t test_data[] = { 400, 403, 445, 448,
											850, 853, 895, 898};

	for (int i = 0; i < output_size; i++) {
		EXPECT_TRUE(output_data[i] == test_data[i]);
	}
}

}//namespace
}//namespace aitisa_api
