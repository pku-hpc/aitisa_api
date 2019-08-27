# Add unit test files
list(APPEND aitisa_test_srcs 
  tensor_test.cpp
  conv_test.cpp
  matmul_test.cpp
  broadcast_test.cpp
  binary_test.cpp
)

# Build a exe for each test files
foreach(test_src ${aitisa_test_srcs})
  get_filename_component(test_name ${test_src} NAME_WE)
  aitisa_api_register_exe(${test_name} 
      "${test_src}" "test" "aitisa_api::aitisa_api" "aitisa_api::gmock")
endforeach()

# Build a exe including all test files 
set(AITISA_BUILD_TEST_MAIN OFF)
if(${AITISA_BUILD_TEST_MAIN})
  add_executable(main main.cpp ${aitisa_test_srcs})
  target_link_libraries(main "aitisa_api::aitisa_api" "aitisa_api::gmock")
endif()