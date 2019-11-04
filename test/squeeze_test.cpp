#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/basic/squeeze.h"
//#include "src/tool/tool.h"
}

void squeeze_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value = i;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {
TEST(Squeeze, Int32) {
  Tensor input;
  DataType dtype = { TYPE_INT32, sizeof(int32_t) };
  Device device = { DEVICE_CPU, 0 };
  int64_t dims[5] = { 2, 1, 5, 1, 10 };
  aitisa_create(dtype, device, LAYOUT_DENSE, dims, 5, &input);
  squeeze_assign_int32(input);
  //tensor_printer2d(input);
  int64_t in_size = aitisa_tensor_size(input);

  Tensor output;
  int64_t axis[2] = { 1, 3 };
  aitisa_squeeze(input, axis, 2, &output);
  //tensor_printer2d(output);
  int64_t out_ndim = aitisa_tensor_ndim(output);
  int64_t* out_dims = aitisa_tensor_dims(output);
  int64_t test_out_dims[3] = { 2, 5, 10 };
  int64_t test_out_ndim = 3;
  EXPECT_EQ(out_ndim, test_out_ndim);
  for (int64_t i = 0; i < out_ndim; i++) {
    EXPECT_EQ(out_dims[i], test_out_dims[i]);
  }
}
  }//namespace
}//namespace aitisa_api