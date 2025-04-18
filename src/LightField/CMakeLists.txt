add_library(LightField LightField.cpp Matrix.cpp View.cpp Block4D.cpp Block4D_.cpp)
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4/)
find_package(OpenCV REQUIRED)
find_package(osqp REQUIRED)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz DOWNLOAD_EXTRACT_TIMESTAMP true)
FetchContent_MakeAvailable(json)

target_include_directories(LightField PUBLIC include)

target_link_libraries(LightField PUBLIC nlohmann_json::nlohmann_json torch osqp::osqp IO ${OpenCV_LIBS}) 
