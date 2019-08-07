include(ExternalProject)
set(_external_target_name gtest)

# create a custom target to drive download , configure, 
# build, install and test steps of an external project
ExternalProject_Add(googletest
  GIT_REPOSITORY   https://github.com/google/googletest
  GIT_TAG          v1.8.x
  GIT_PROGRESS     true
  PREFIX           "${EXTERNAL_INSTALL_DIRECTORY}/${_external_target_name}"
  UPDATE_DISCONNECTED true
  SOURCE_DIR       "${EXTERNAL_INSTALL_DIRECTORY}/${_external_target_name}/googletest-src"
  BINARY_DIR       "${EXTERNAL_INSTALL_DIRECTORY}/${_external_target_name}/googletest-build"
  INSTALL_COMMAND  ""
)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
if(NOT TARGET gtest_main)
  add_subdirectory(
    ${EXTERNAL_INSTALL_DIRECTORY}/${_external_target_name}/googletest-src
    ${EXTERNAL_INSTALL_DIRECTORY}/${_external_target_name}/googletest-build
    EXCLUDE_FROM_ALL)
endif()

add_library(rgknn::gtest ALIAS gtest_main)
unset(_external_target_name)
