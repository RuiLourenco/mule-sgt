cmake_minimum_required(VERSION 3.19)
project(mule_sgt LANGUAGES C CXX)

set(TEMPLATE_DIR "${CMAKE_SOURCE_DIR}/01_TemplateFolder")
set(TEST_RUN_DIR "${CMAKE_SOURCE_DIR}/results") # Or wherever you want tests to run from
message(STATUS "DEBUG: CMake version: ${CMAKE_VERSION}")
message(STATUS "DEBUG: Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "DEBUG: Template directory: ${TEMPLATE_DIR}")
message(STATUS "DEBUG: Test run directory: ${TEST_RUN_DIR}")
# Guard Condition: Only run this setup on the very first configure.
# We check for the existence of the destination to see if we've run before.
if(NOT IS_DIRECTORY "${TEST_RUN_DIR}")
    message(STATUS "First-time setup for this worktree. Deploying test template...")

    # The command to copy the entire template directory.
    # The trailing slash on the source is important!
    file(COPY "${TEMPLATE_DIR}/" DESTINATION "${TEST_RUN_DIR}")

    message(STATUS "Test template successfully deployed to ${TEST_RUN_DIR}")
else()
    message(STATUS "Test directory already exists. Skipping template deployment.")
endif()

include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Add an option to enable AddressSanitizer (ASan)
# It will be OFF by default.
option(ENABLE_ASAN "Enable AddressSanitizer for debugging memory errors" OFF)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer (ASan) enabled. This will slow down the executable.")
    # Add the necessary compiler and linker flags
    add_compile_options(-fsanitize=address -g)
    add_link_options(-fsanitize=address)
endif()

# export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "/nfs/home/ruilourenco.it/miniconda3/envs/slantKLT/lib/python3.10/site-packages/torch/share/cmake/")
find_package(Torch REQUIRED)
find_package(LAPACK REQUIRED)

enable_testing()
message(STATUS "DEBUG: We reach the end?")

add_subdirectory(src) 
add_subdirectory(tests)



