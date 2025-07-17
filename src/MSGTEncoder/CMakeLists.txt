find_package(Boost COMPONENTS program_options REQUIRED)
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4/)
find_package(OpenCV REQUIRED)

add_executable(MSGTEncoder MSGTEncoder.cpp)
target_link_options(MSGTEncoder PRIVATE -rdynamic)



target_link_libraries(MSGTEncoder
 PRIVATE
  LightField
  Encoder
  Boost::program_options
  torch
  ${OpenCV_LIBS})
