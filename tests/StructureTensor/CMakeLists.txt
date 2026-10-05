add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(StructureTensor_gtests StructureTensor_test.cpp)

target_link_libraries(StructureTensor_gtests
 PRIVATE
  GTest::GTest
  LightField
  torch)

add_test(StructureTensor_gtests StructureTensor_gtests)
