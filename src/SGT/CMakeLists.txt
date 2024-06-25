add_library(SGT MultiscaleTransform.cpp)

target_include_directories(SGT PUBLIC include)

target_link_libraries(SGT PUBLIC torch LightField)
