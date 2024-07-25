find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(MSGTEncoder MSGTEncoder.cpp)

target_link_libraries(MSGTEncoder
 PRIVATE
  LightField
  Encoder
  Boost::program_options
  torch)
  set_target_properties(MSGTEncoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")