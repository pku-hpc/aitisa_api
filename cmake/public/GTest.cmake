# if gtest isn't found, cmake will throw an error
# `Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR GTEST_MAIN_LIBRARY)`
# rather than set `GTEST_FOUND` variable and running into if-else branch
find_package(GTest QUIET REQUIRED)
if(GTEST_FOUND)
  message(WARNING "std_ai_api: Found GTest.")
else()
  message(WARNING "std_ai_api: Cannot find GTest, turn off the `USE_GTEST` option automatically.")
  set(USE_GTEST OFF)
  return()
endif()