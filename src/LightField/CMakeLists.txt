add_library(LightField LightField.cpp Matrix.cpp View.cpp Block4D.cpp Block4D_.cpp)
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4/)
find_package(OpenCV REQUIRED)
find_package(osqp REQUIRED)

target_include_directories(LightField PUBLIC include)

target_link_libraries(LightField PUBLIC torch osqp::osqp IO ${OpenCV_LIBS})
