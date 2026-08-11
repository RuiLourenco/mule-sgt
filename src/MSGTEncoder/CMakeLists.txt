find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(MSGTEncoder MSGTEncoder.cpp)
target_link_options(MSGTEncoder PRIVATE -rdynamic)



target_link_libraries(MSGTEncoder
 PRIVATE
  LightField
  Encoder
  Boost::program_options
  torch)
