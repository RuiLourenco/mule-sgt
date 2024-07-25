find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(MSGTDecoder MSGTDecoder.cpp)

target_link_libraries(MSGTDecoder
 PRIVATE
  LightField
  Decoder
  Boost::program_options
  torch)
  set_target_properties(MSGTDecoder PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")