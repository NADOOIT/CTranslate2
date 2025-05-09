enable_testing()

option(BUILD_GMOCK "" OFF)
option(INSTALL_GTEST "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest)

add_executable(ctranslate2_test
  batching_test.cc
  decoding_test.cc
  layers_test.cc
  model_test.cc
  storage_view_test.cc
  ops_test.cc
  primitives_test.cc
  translator_test.cc
  test_utils.cc
  test.cc)
target_include_directories(ctranslate2_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )
target_link_libraries(ctranslate2_test
  ${PROJECT_NAME}
  gtest_main
  )

if(WITH_METAL)
  add_custom_command(TARGET ctranslate2_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/default.metallib
    $<TARGET_FILE_DIR:ctranslate2_test>/default.metallib
    COMMENT "Copying default.metallib to test directory for ctranslate2_test"
  )
endif()

add_test(NAME ctranslate2_test COMMAND ctranslate2_test ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_executable(benchmark_ops
  benchmark_ops.cc
  )
target_link_libraries(benchmark_ops
  ${PROJECT_NAME}
  )

if(WITH_CUDA)
  target_link_libraries(benchmark_ops ${CUDA_LIBRARIES})
endif()

