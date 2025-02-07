cmake_minimum_required(VERSION 3.19)
project(mule_sgt LANGUAGES C CXX)

# export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "/nfs/home/ruilourenco.it/miniconda3/envs/slantKLT/lib/python3.10/site-packages/torch/share/cmake/")
find_package(Torch REQUIRED)
find_package(LAPACK REQUIRED)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)



