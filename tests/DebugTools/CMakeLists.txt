add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(DebugTools_gtests DebugTools_test.cpp)

target_link_libraries(DebugTools_gtests
 PRIVATE
  GTest::GTest
  DebugTools
  LightField
  torch)

add_test(DebugTools_gtests DebugTools_gtests)
