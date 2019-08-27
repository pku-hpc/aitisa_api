##############################################################################
# Locates Doxygen and configures documentation generation
##############################################################################

if(aitisa_api_public_doxygen_cmake_included)
  return()
endif()
set(aitisa_api_public_doxygen_cmake_included true)

set(aitisa_api_public_doxygen_cmake_included true)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(DOXYGEN_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/docs.stamp)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY)
    file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/src/*.h)
    add_custom_command(
        OUTPUT ${DOXYGEN_STAMP_FILE}
        DEPENDS ${HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        COMMAND ${CMAKE_COMMAND} -E touch ${DOXYGEN_STAMP_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
    add_custom_target(docs DEPENDS ${DOXYGEN_STAMP_FILE})

    if(NOT AITISA_API_INSTALL_MODE STREQUAL "BUNDLE")
        install(
            DIRECTORY ${DOXYGEN_OUTPUT_DIR}
            DESTINATION share/doc/${LIB_NAME} OPTIONAL)
    endif()
endif(DOXYGEN_FOUND)
