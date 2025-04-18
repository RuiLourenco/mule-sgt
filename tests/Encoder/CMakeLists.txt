add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(Encoder_gtests Encoder_test.cpp)

target_link_libraries(Encoder_gtests
 PRIVATE
  GTest::GTest
  Encoder
  Decoder
  LightField
  OldDCT
  ProbabilityModel
  DebugTools
  torch)

add_test(Encoder_gtests Encoder_gtests )