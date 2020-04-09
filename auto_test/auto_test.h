#pragma once

#include "auto_test/test_code/binary_op_test.h"
#include "auto_test/test_code/matmul_test.h"
#include "auto_test/test_code/conv_test.h"
#include "auto_test/test_code/activation_test.h"

#define REGISTER_OP(ADD, SUB, MUL, DIV, MATMUL, CONV)   \
  REGISTER_BINARY_OP(ADD, SUB, MUL, DIV);               \
  REGISTER_MATMUL(MATMUL);                              \
  REGISTER_CONV(CONV); 

#define PERFORM_TEST                                    \
  ::testing::InitGoogleTest(&argc, argv);               \
  return RUN_ALL_TESTS();

