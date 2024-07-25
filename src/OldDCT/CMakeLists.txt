add_library(OldDCT MultiscaleTransform.cpp)

target_include_directories(OldDCT PUBLIC include)

target_link_libraries(OldDCT PUBLIC torch LightField)
