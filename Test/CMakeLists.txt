add_executable(multiply_test multiply_test.cpp)

target_link_libraries(LightField_test
 PRIVATE
  GTest::GTest
  LightField)

add_test(LightField_gtests multiply_test)