
add_library(GTest::GTest INTERFACE IMPORTED)
find_package(Boost COMPONENTS headers iostreams filesystem REQUIRED)

target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(IO_gtests IO_test.cpp)

target_link_libraries(IO_gtests
 PRIVATE
  GTest::GTest
  IO
  torch
  Boost::headers 
  Boost::filesystem 
  Boost::iostreams)

add_test(IO_gtests IO_gtests)