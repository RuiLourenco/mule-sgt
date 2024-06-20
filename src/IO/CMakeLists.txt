find_package(Boost COMPONENTS headers iostreams filesystem REQUIRED)

add_library(IO io.cpp)

target_link_libraries(IO PUBLIC torch Boost::headers Boost::filesystem Boost::iostreams)

target_include_directories(IO PUBLIC include)