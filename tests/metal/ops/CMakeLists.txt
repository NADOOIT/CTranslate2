cmake_minimum_required(VERSION 3.14)
project(GEMM_Benchmarks LANGUAGES CXX OBJCXX)

if(BUILD_TESTS OR BUILD_TESTING)
  add_subdirectory(${CMAKE_SOURCE_DIR}/../../../third_party/googletest ${CMAKE_BINARY_DIR}/googletest)
endif()


# add_executable(gemm_test gemm_test.mm)
# target_compile_options(gemm_test PRIVATE -x objective-c++)
# target_link_libraries(gemm_test
#   "-framework Metal"
#   "-framework Foundation"
#   "-framework MetalKit"
#   "-framework MetalPerformanceShaders"
#   "-framework Accelerate"
# )

add_executable(gemm_test_main gemm_test_main.mm gemm_utils.mm gemm_metal_runner.mm gemm_cpu_accel.mm)
target_compile_options(gemm_test_main PRIVATE -x objective-c++)
target_link_libraries(gemm_test_main
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
  "-framework MetalPerformanceShaders"
  "-framework Accelerate"
)
add_test(NAME gemm_test_main COMMAND gemm_test_main)

add_executable(gemm_multi_device_bench_test_cpu gemm_multi_device_bench_test_cpu.mm gemm_bench_helpers.mm gemm_utils.mm gemm_cpu_accel.mm)
target_compile_options(gemm_multi_device_bench_test_cpu PRIVATE -x objective-c++)
target_link_libraries(gemm_multi_device_bench_test_cpu
  "-framework Accelerate"
  "-framework Foundation"
)

add_executable(gemm_multi_device_bench_test_metal gemm_multi_device_bench_test_metal.mm gemm_bench_helpers.mm gemm_utils.mm gemm_metal.mm gemm_cpu_accel.mm)
target_compile_options(gemm_multi_device_bench_test_metal PRIVATE -x objective-c++)
target_link_libraries(gemm_multi_device_bench_test_metal
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
  "-framework MetalPerformanceShaders"
  "-framework Accelerate"
)

# Old monolithic target (deprecated)
# add_executable(gemm_multi_device_bench_test gemm_multi_device_bench_test.mm)
# target_compile_options(gemm_multi_device_bench_test PRIVATE -x objective-c++)
# target_link_libraries(gemm_multi_device_bench_test
#   "-framework Metal"
#   "-framework Foundation"
#   "-framework MetalKit"
#   "-framework MetalPerformanceShaders"
#   "-framework Accelerate"
# )

add_executable(relu_multi_device_bench_test relu_multi_device_bench_test.mm)
target_compile_options(relu_multi_device_bench_test PRIVATE -x objective-c++)
target_link_libraries(relu_multi_device_bench_test
  "-framework Metal"
  "-framework Foundation"
  "-framework MetalKit"
)

add_executable(add_multi_device_bench_test add_multi_device_bench_test.mm add_metal_batch.mm)
target_compile_options(add_multi_device_bench_test PRIVATE -x objective-c++)
target_link_libraries(add_multi_device_bench_test
  "-framework Metal"
  "-framework Foundation"
)

add_executable(multiply_multi_device_bench_test multiply_multi_device_bench_test.mm multiply_metal_batch.mm)
target_compile_options(multiply_multi_device_bench_test PRIVATE -x objective-c++)
target_link_libraries(multiply_multi_device_bench_test
  "-framework Metal"
  "-framework Foundation"
)

add_executable(softmax_multi_device_bench_test softmax_multi_device_bench_test.mm softmax_metal_batch.mm)
target_compile_options(softmax_multi_device_bench_test PRIVATE -x objective-c++)
target_link_libraries(softmax_multi_device_bench_test
  "-framework Metal"
  "-framework Foundation"
)

add_executable(max_load_bench max_load_bench.cpp add_metal_batch.mm multiply_metal_batch.mm)
target_compile_options(max_load_bench PRIVATE -x objective-c++)
target_link_libraries(max_load_bench
  "-framework Metal"
  "-framework Foundation"
)

add_executable(adaptive_dispatch_test adaptive_dispatch_test.cpp relu_metal_batch.mm)
target_include_directories(adaptive_dispatch_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)
target_compile_options(adaptive_dispatch_test PRIVATE -x objective-c++)
target_link_libraries(adaptive_dispatch_test gtest gtest_main pthread
  "-framework Metal"
  "-framework Foundation"
)

add_executable(adaptive_dispatch_bench adaptive_dispatch_bench.cpp relu_metal_batch.mm)
target_compile_options(adaptive_dispatch_bench PRIVATE -x objective-c++)
target_link_libraries(adaptive_dispatch_bench pthread
  "-framework Metal"
  "-framework Foundation"
)

# Add adaptive dispatch tests for all operators
add_executable(gemm_adaptive_dispatch_test gemm_adaptive_dispatch_test.cpp gemm_metal_batch.mm)
target_include_directories(gemm_adaptive_dispatch_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)
target_compile_options(gemm_adaptive_dispatch_test PRIVATE -x objective-c++)
target_link_libraries(gemm_adaptive_dispatch_test gtest gtest_main pthread
  "-framework Metal"
  "-framework Foundation"
)

add_executable(add_adaptive_dispatch_test add_adaptive_dispatch_test.cpp add_metal_batch.mm)
target_include_directories(add_adaptive_dispatch_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)
target_compile_options(add_adaptive_dispatch_test PRIVATE -x objective-c++)
target_link_libraries(add_adaptive_dispatch_test gtest gtest_main pthread
  "-framework Metal"
  "-framework Foundation"
)

add_executable(multiply_adaptive_dispatch_test multiply_adaptive_dispatch_test.cpp multiply_metal_batch.mm)
target_include_directories(multiply_adaptive_dispatch_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)
target_compile_options(multiply_adaptive_dispatch_test PRIVATE -x objective-c++)
target_link_libraries(multiply_adaptive_dispatch_test gtest gtest_main pthread
  "-framework Metal"
  "-framework Foundation"
)

add_executable(softmax_adaptive_dispatch_test softmax_adaptive_dispatch_test.cpp softmax_metal_batch.mm)
target_include_directories(softmax_adaptive_dispatch_test PRIVATE ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include)
target_compile_options(softmax_adaptive_dispatch_test PRIVATE -x objective-c++)
target_link_libraries(softmax_adaptive_dispatch_test gtest gtest_main pthread
  "-framework Metal"
  "-framework Foundation"
)

# Kompiliere gemm_kernel.metal zu gemm_kernel.metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gemm_kernel.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/gemm_kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/gemm_kernel.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/gemm_kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/gemm_kernel.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_kernel.metal
  COMMENT "Compiling Metal kernel to metallib"
)
add_custom_target(gemm_kernel_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gemm_kernel.metallib)
# add_dependencies(gemm_test gemm_kernel_lib)
add_dependencies(gemm_test_main gemm_kernel_lib)

# Kompiliere relu_kernel.metal zu relu_kernel.metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/relu_kernel.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/relu_kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/relu_kernel.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/relu_kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/relu_kernel.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/relu_kernel.metal
)
add_custom_target(relu_kernel_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/relu_kernel.metallib)
add_dependencies(relu_multi_device_bench_test relu_kernel_lib)

# Kompiliere add_kernel.metal zu add_kernel.metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add_kernel.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/add_kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/add_kernel.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/add_kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/add_kernel.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add_kernel.metal
)
add_custom_target(add_kernel_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add_kernel.metallib)
add_dependencies(add_adaptive_dispatch_test add_kernel_lib)

# Kompiliere multiply_kernel.metal zu multiply_kernel.metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/multiply_kernel.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/multiply_kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/multiply_kernel.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/multiply_kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/multiply_kernel.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/multiply_kernel.metal
)
add_custom_target(multiply_kernel_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/multiply_kernel.metallib)
add_dependencies(multiply_adaptive_dispatch_test multiply_kernel_lib)

# Kompiliere softmax_kernel.metal zu softmax_kernel.metallib
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/softmax_kernel.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/softmax_kernel.metal -o ${CMAKE_CURRENT_BINARY_DIR}/softmax_kernel.air
  COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/softmax_kernel.air -o ${CMAKE_CURRENT_BINARY_DIR}/softmax_kernel.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/softmax_kernel.metal
)
add_custom_target(softmax_kernel_lib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/softmax_kernel.metallib)
add_dependencies(softmax_adaptive_dispatch_test softmax_kernel_lib)
