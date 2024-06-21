add_library(LightField LightField.cpp Matrix.cpp View.cpp Block4D.cpp Block4D_.cpp)

target_include_directories(LightField PUBLIC include)

target_link_libraries(LightField PUBLIC torch IO)
