# Add unit test files
list(APPEND aitisa_test_srcs 
  tensor_test.cpp
  conv_test.cpp
  matmul_test.cpp
  broadcast_test.cpp
  binary_test.cpp
  pooling_test.cpp
  softmax_test.cpp
  slice_test.cpp
  dot_test.cpp
  cast_test.cpp
  duplicate_test.cpp
  dropout_test.cpp
  sqrt_test.cpp
  squeeze_test.cpp
  reshape_test.cpp
  batch_norm_test.cpp
  sigmoid_test.cpp
  relu_test.cpp
)

# Build a exe for each test files
foreach(test_src ${aitisa_test_srcs})
  get_filename_component(test_name ${test_src} NAME_WE)
  aitisa_api_register_exe(${test_name} 
      "${test_src}" "test" "aitisa_api::aitisa_api" "aitisa_api::gmock")
endforeach()

# Build a exe including all test files 
set(AITISA_BUILD_TEST_MAIN ON)
if(${AITISA_BUILD_TEST_MAIN})
  add_executable(main main.cpp ${aitisa_test_srcs})
  target_link_libraries(main "aitisa_api::aitisa_api" "aitisa_api::gmock")
endif()