add_executable(metal_ops_test metal_ops_test.mm)
target_compile_options(metal_ops_test PRIVATE -x objective-c++)
target_link_libraries(metal_ops_test
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
)

add_executable(metal_buffer_test metal_buffer_test.mm)
target_compile_options(metal_buffer_test PRIVATE -x objective-c++)
target_link_libraries(metal_buffer_test
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
)

add_subdirectory(ops)

add_executable(metal_device_info_test metal_device_info_test.mm)
target_compile_options(metal_device_info_test PRIVATE -x objective-c++)
target_link_libraries(metal_device_info_test
  "-framework Metal"
  "-framework Foundation"
)
