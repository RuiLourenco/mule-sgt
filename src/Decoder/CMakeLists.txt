add_library(Decoder Hierarchical4DDecoder.cpp PartitionDecoder.cpp ABADecoder.cpp)

target_include_directories(Decoder PUBLIC include)

target_link_libraries(Decoder PUBLIC torch ProbabilityModel LightField)
