add_library(DebugTools CodingPartitionInfo.cpp CodingUnitInfo.cpp)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz DOWNLOAD_EXTRACT_TIMESTAMP true)
FetchContent_MakeAvailable(json)

target_include_directories(DebugTools PUBLIC include) 

target_link_libraries(DebugTools PUBLIC nlohmann_json::nlohmann_json torch LightField)