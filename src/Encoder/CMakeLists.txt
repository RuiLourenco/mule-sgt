add_library(Encoder Hierarchical4DEncoder.cpp TransformPartition.cpp ABACoder.cpp)

target_include_directories(Encoder PUBLIC include)

target_link_libraries(Encoder 
PUBLIC 
    torch 
    DebugTools 
    ProbabilityModel  
    OldDCT 
    LightField)  
