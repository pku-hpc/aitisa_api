add_executable(tensor_test tensor_test.c)
target_link_libraries(tensor_test -Wl,--whole-archive std_ai_api -Wl,--no-whole-archive)