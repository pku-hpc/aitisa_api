#include "gtest/gtest.h"
extern "C" {
#include "src/basic/broadcast.h"
#include "src/basic/factories.h"
}

namespace aitisa_api {
namespace {

TEST(Broadcast, 4D_6D) {
  int64_t dims_in1[4] = {1, 3, 4, 2};
  int64_t dims_in2[6] = {5, 1, 2, 3, 1, 2};
  int64_t dims_out[6] = {0, 0, 0, 0, 0, 0};

  aitisa_broadcast_array(dims_in1, 4, dims_in2, 6, dims_out, 6);

  // printf("dims=");
  // for (int64_t i = 0; i < 6; i++) {
  //   printf("%ld, ", dims_out[i]);
  // }
  // printf("\n");
}

}  // namespace
}  // namespace aitisa_api