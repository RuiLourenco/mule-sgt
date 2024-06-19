include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.14.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(LightField_gtests LightField_test)

target_link_libraries(LightField_test
 PRIVATE
  GTest::GTest
  LightField)

add_test(LightField_gtests LightField_test)