set(_external_target_name gtest)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(cmake/local/DownloadProject.cmake)
download_project(PROJ                ${_external_target_name} 
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             v1.8.x  
                 GIT_PROGRESS        TRUE
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
                 PREFIX "${AITISA_API_EXTERNAL_DIR}/${_external_target_name}"
)


#set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
if(NOT TARGET gtest_main)
  add_subdirectory(
    ${${_external_target_name}_SOURCE_DIR} 
    ${${_external_target_name}_BINARY_DIR}
    EXCLUDE_FROM_ALL)
endif()

add_library(aitisa_api::gtest ALIAS gtest_main)
add_library(aitisa_api::gmock ALIAS gmock_main)

unset(_external_target_name)