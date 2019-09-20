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

TEST(Pooling1D, AvgFloatK2S2P1) {
	Tensor input;
	DataType dtype = { TYPE_FLOAT, sizeof(float) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[3] = { 2, 3, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &input);
	natural_assign_float(input);

	Tensor output;
	int ksize[1] = { 2 };
	int stride[1] = { 2 };
	int padding[1] = { 1 };

	aitisa_pooling(input, "avg", ksize, stride, padding, &output);
	
	float* output_data = (float*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	std::vector<float> test_data = { 0.05, 0.25, 0.45, 0.65, 0.4,
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

TEST(Pooling2D, AvgInt32K3S3P0) {
	Tensor input;
	DataType dtype = { TYPE_INT32, sizeof(int32_t) };
	//DataType dtype = { TYPE_FLOAT, sizeof(float) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[4] = {1, 2, 9, 9};
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 4, &input);
	natural_assign_int32(input);

	Tensor output;
	int ksize[2] = {3, 3};
	int stride[2] = {3, 3};
	int padding[2] = {0, 0};

	aitisa_pooling(input, "avg", ksize, stride, padding, &output);
	
	int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	ASSERT_THAT(std::vector<int32_t>(output_data, output_data + output_size),
		::testing::ElementsAre( 10,  13,  16,  37,  40,  43,  64,  67,  70,
							    91,  94,  97, 118, 121, 124, 145, 148, 151));
}

TEST(Pooling2D, MaxDoubleK4S4P3) {
	Tensor input;
	DataType dtype = { TYPE_DOUBLE, sizeof(double) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[4] = { 1, 2, 9, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 4, &input);
	natural_assign_double(input);

	Tensor output;
	int ksize[2] = { 4, 4 };
	int stride[2] = { 4, 4 };
	int padding[2] = { 3, 3 };

	aitisa_pooling(input, "max", ksize, stride, padding, &output);

	double* output_data = (double*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	std::vector<double> test_data = { 2,  2.4,  2.6,  5.6,  6.0,  6.2,  7.4,  7.8,  8.0,
								   10.1, 10.5, 10.7, 13.7, 14.1, 14.3, 15.5, 15.9, 16.1 };
	
	for (int i = 0; i < output_size; i++) {
		// Due to the problem of precision, consider the two numbers are equal when their difference is less than 0.0001
		EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.0001);
	}
}

TEST(Pooling3D, MaxInt32K4S4P3) {
	Tensor input;
	DataType dtype = { TYPE_INT32, sizeof(int32_t) };
	Device device = { DEVICE_CPU, 0 };
	int64_t dims[5] = { 1, 2, 6, 10, 9 };
	aitisa_create(dtype, device, LAYOUT_DENSE, dims, 5, &input);
	natural_assign_int32(input);

	Tensor output;
	int ksize[3] = { 4, 4, 4 };
	int stride[3] = { 4, 4, 4 };
	int padding[3] = { 2, 2, 3 };

	aitisa_pooling(input, "max", ksize, stride, padding, &output);
	
	int32_t* output_data = (int32_t*)aitisa_tensor_data(output);
	int64_t output_size = aitisa_tensor_size(output);
	std::vector<int32_t> test_data = { 200, 204, 206, 236, 240, 242, 263, 267, 269,
									   470, 474, 476, 506, 510, 512, 533, 537, 539,
											740, 744, 746, 776, 780, 782, 803, 807, 809,
										   1010,1014,1016,1046,1050,1052,1073,1077,1079};

	for (int i = 0; i < output_size; i++) {
		EXPECT_TRUE(output_data[i] == test_data[i]);
	}
}

}//namespace
}//namespace aitisa_api
