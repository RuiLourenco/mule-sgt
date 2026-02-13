add_library(Encoder Hierarchical4DEncoder.cpp TransformPartition.cpp ABACoder.cpp)

target_include_directories(Encoder PUBLIC include)
find_package(OpenMP REQUIRED)

target_link_libraries(Encoder 
PUBLIC 
    torch 
    ProbabilityModel   
    OpenMP::OpenMP_CXX
    LightField)  
