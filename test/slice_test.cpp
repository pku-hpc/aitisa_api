#include "gtest/gtest.h"
#include <math.h>
#include <vector> 
extern "C" {
#include "src/basic/factories.h"
#include "src/core/tensor.h"
#include "src/basic/slice.h"
//#include "src/tool/tool.h"
}

void slice_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value = i;
    data[i] = value;
  }
}

void slice_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Slice, StepIsOne) {
  Tensor input;
  DataType dtype = { TYPE_FLOAT, sizeof(float) };
  Device device = { DEVICE_CPU, 0 };
  int64_t dims[3] = { 2, 4, 5 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &input);
  slice_assign_float(input);

  Tensor output;
  int begin[3] = { 1, 1, 2 };
  int size[3] = { 1, 2, 3 };
  int step[3] = { 1, 1, 1 };
  aitisa_slice(input, begin, size, step, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  int64_t out_size = aitisa_tensor_size(output);
  std::vector<float> test_data = { 2.7, 2.8, 2.9,
										               3.2, 3.3, 3.4};
  for (int64_t i = 0; i < out_size; i++) {
	  /* Due to the problem of precision, consider the two numbers 
			 are equal when their difference is less than 0.00001*/
		EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
	}
}

TEST(Slice, StepNotOne) {
  Tensor input;
  DataType dtype = { TYPE_INT32, sizeof(int32_t) };
  Device device = { DEVICE_CPU, 0 };
  int64_t dims[3] = { 2, 5, 10 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 3, &input);
  slice_assign_int32(input);

  Tensor output;
  int begin[3] = { 0, 1, 2 };
  int size[3] = { 2, 4, 6 };
  int step[3] = { 1, 2, 3 };
  aitisa_slice(input, begin, size, step, &output);
  
  int32_t *out_data = (int32_t *)aitisa_tensor_data(output);
  int64_t out_size = aitisa_tensor_size(output);
	std::vector<int32_t> test_data = { 12, 15, 32, 35, 62, 65, 82, 85};
	for (int64_t i = 0; i < out_size; i++) {
		EXPECT_EQ(out_data[i],  test_data[i]);
	}
}

}//namespace
}//namespace aitisa_api