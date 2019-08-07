cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(std_ai_api VERSION 0.0.1 LANGUAGES C)

option(ENABLE_TESTING "Enable Tests" ON)
option(USE_OPENMP "Build with OpenMP support" ON)
option(USE_GTEST "Test with GTest support" ON)

# Show the detail of compile messages
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_COLOR_MAKEFILE   ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib CACHE PATH "Single directory for all static libraries.")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib CACHE PATH "Single directory for all dynamic libraries.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin CACHE PATH "Single directory for all executable.")
set(EXTERNAL_INSTALL_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

# Find all the files under `src/`
FILE(GLOB_RECURSE cpu_source "src/*.c" "src/*.h")
set(srcs ${cpu_source})

# Create library target with all the source codes.
add_library(std_ai_api ${srcs})

set_target_properties(std_ai_api PROPERTIES 
  CUDA_SEPERABLE_COMPILATION ON 
  INTERFACE_POSITION_INDEPENDENT_CODE ON 
)

# change "PRIVATE" to "PUBLIC" or test file won't compile under c_std_99
target_compile_features(std_ai_api PUBLIC c_std_99)

target_include_directories(std_ai_api
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  PRIVATE
    # ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(std_ai_api_link_libraries)

if(USE_OPENMP)
  include(cmake/public/OpenMP.cmake)
endif()

target_link_libraries(std_ai_api PUBLIC ${std_ai_api_link_libraries})

if(ENABLE_TESTING)
  enable_testing()
  if(USE_GTEST)
    include(cmake/local/GTest.cmake)
  endif()
  add_subdirectory(test)
endif()