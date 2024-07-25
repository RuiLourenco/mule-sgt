add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(LightField_gtests LightField_test.cpp)

target_link_libraries(LightField_gtests
 PRIVATE
  GTest::GTest
  LightField
  OldDCT
  torch)

add_test(LightField_gtests LightField_gtests )