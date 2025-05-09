include_directories(${CMAKE_SOURCE_DIR}/include)

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


# --- Attention Test ---
add_executable(metal_attention_test metal_attention_test.mm)
target_compile_options(metal_attention_test PRIVATE -x objective-c++ -I${CMAKE_SOURCE_DIR}/include)
target_link_libraries(metal_attention_test
  ${PROJECT_NAME}
  gtest
  gtest_main
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
)
target_include_directories(metal_attention_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link against the main ctranslate2 library and gtest
# Note: This assumes CTranslate2::ctranslate2 and gtest/gtest_main targets are available
# from the parent CMake scope.

