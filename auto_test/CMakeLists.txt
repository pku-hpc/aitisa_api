# Add unit test files
list(APPEND auto_test_srcs 
  auto_add_test.cpp
)

# Build a exe for each test files
foreach(test_src ${auto_test_srcs})
  get_filename_component(test_name ${test_src} NAME_WE)
  aitisa_api_register_exe(${test_name} 
      "${test_src}" "test" "aitisa_api::aitisa_api" "aitisa_api::gmock")
endforeach()
