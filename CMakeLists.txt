cmake_minimum_required(VERSION 3.7)

# Set policy for setting the MSVC runtime library for static MSVC builds
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(ctranslate2)

# Read version from version.py
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/python/ctranslate2/version.py VERSION_FILE)
foreach(line IN LISTS VERSION_FILE)
  if (line MATCHES "__version__")
    string(REGEX MATCH "[0-9.]+" CTRANSLATE2_VERSION ${line})
    break()
  endif()
endforeach()

if(NOT CTRANSLATE2_VERSION)
  message(FATAL_ERROR "Version can't be read from version.py")
endif()

string(REPLACE "." ";" CTRANSLATE2_VERSION_LIST ${CTRANSLATE2_VERSION})
list(GET CTRANSLATE2_VERSION_LIST 0 CTRANSLATE2_MAJOR_VERSION)

option(WITH_MKL "Compile with Intel MKL backend" ON)
option(WITH_DNNL "Compile with DNNL backend" OFF)
option(WITH_ACCELERATE "Compile with Accelerate backend" OFF)
option(WITH_OPENBLAS "Compile with OpenBLAS backend" OFF)
option(WITH_RUY "Compile with Ruy backend" OFF)
option(WITH_CUDA "Compile with CUDA backend" OFF)
option(WITH_CUDNN "Compile with cuDNN backend" OFF)
option(WITH_METAL "Enable Metal backend" ON)
option(CUDA_DYNAMIC_LOADING "Dynamically load CUDA libraries at runtime" OFF)
option(ENABLE_CPU_DISPATCH "Compile CPU kernels for multiple ISA and dispatch at runtime" ON)
option(ENABLE_PROFILING "Compile with profiling support" OFF)
option(BUILD_CLI "Compile the clients" ON)
option(BUILD_TESTS "Compile the tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(WITH_TENSOR_PARALLEL "Compile with NCCL and MPI backend" OFF)
option(WITH_FLASH_ATTN "Compile with Flash Attention 2" OFF)

if(ENABLE_PROFILING)
  message(STATUS "Enable profiling support")
  add_definitions(-DCT2_ENABLE_PROFILING)
endif()

# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 17)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -Xpreprocessor -fopenmp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
  include_directories("/opt/homebrew/opt/libomp/include")
  link_directories("/opt/homebrew/opt/libomp/lib")
  
  # Add ARM64 build definition
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_definitions(-DCT2_ARM64_BUILD)
  endif()
endif()

find_package(Threads REQUIRED)
add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)

set(PRIVATE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set(LIBRARIES
  ${CMAKE_THREAD_LIBS_INIT}
  spdlog::spdlog_header_only
)

set(SOURCES
  src/allocator.cc
  src/batch_reader.cc
  src/buffered_translation_wrapper.cc
  src/cpu/allocator.cc
  src/cpu/backend.cc
  src/cpu/cpu_info.cc
  src/cpu/cpu_isa.cc
  src/cpu/kernels.cc
  src/cpu/parallel.cc
  src/cpu/primitives.cc
  src/decoding.cc
  src/decoding_utils.cc
  src/devices.cc
  src/dtw.cc
  src/encoder.cc
  src/env.cc
  src/filesystem.cc
  src/generator.cc
  src/layers/attention_layer.cc
  src/layers/attention.cc
  src/layers/flash_attention.cc
  src/layers/common.cc
  src/layers/decoder.cc
  src/layers/transformer.cc
  src/layers/wav2vec2.cc
  src/layers/wav2vec2bert.cc
  src/layers/whisper.cc
  src/logging.cc
  src/models/language_model.cc
  src/models/model.cc
  src/models/model_factory.cc
  src/models/model_reader.cc
  src/models/sequence_to_sequence.cc
  src/models/transformer.cc
  src/models/wav2vec2.cc
  src/models/wav2vec2bert.cc
  src/models/whisper.cc
  src/ops/activation.cc
  src/ops/add.cc
  src/ops/alibi_add.cc
  src/ops/alibi_add_cpu.cc
  src/ops/bias_add.cc
  src/ops/bias_add_cpu.cc
  src/ops/concat.cc
  src/ops/concat_split_slide_cpu.cc
  src/ops/conv1d.cc
  src/ops/conv1d_cpu.cc
  src/ops/cos.cc
  src/ops/dequantize.cc
  src/ops/dequantize_cpu.cc
  src/ops/flash_attention.cc
  src/ops/flash_attention_cpu.cc
  src/ops/gather.cc
  src/ops/gather_cpu.cc
  src/ops/gelu.cc
  src/ops/gemm.cc
  src/ops/gumbel_max.cc
  src/ops/gumbel_max_cpu.cc
  src/ops/layer_norm.cc
  src/ops/layer_norm_cpu.cc
  src/ops/log.cc
  src/ops/matmul.cc
  src/ops/mean.cc
  src/ops/mean_cpu.cc
  src/ops/median_filter.cc
  src/ops/min_max.cc
  src/ops/mul.cc
  src/ops/multinomial.cc
  src/ops/multinomial_cpu.cc
  src/ops/quantize.cc
  src/ops/quantize_cpu.cc
  src/ops/relu.cc
  src/ops/rms_norm.cc
  src/ops/rms_norm_cpu.cc
  src/ops/rotary.cc
  src/ops/rotary_cpu.cc
  src/ops/sin.cc
  src/ops/softmax.cc
  src/ops/softmax_cpu.cc
  src/ops/split.cc
  src/ops/slide.cc
  src/ops/sub.cc
  src/ops/sigmoid.cc
  src/ops/swish.cc
  src/ops/tanh.cc
  src/ops/tile.cc
  src/ops/tile_cpu.cc
  src/ops/topk.cc
  src/ops/topk_cpu.cc
  src/ops/topp_mask.cc
  src/ops/topp_mask_cpu.cc
  src/ops/transpose.cc
  src/ops/nccl_ops.cc
  src/ops/nccl_ops_cpu.cc
  src/ops/awq/dequantize.cc
  src/ops/awq/dequantize_cpu.cc
  src/ops/awq/gemm.cc
  src/ops/awq/gemm_cpu.cc
  src/ops/awq/gemv.cc
  src/ops/awq/gemv_cpu.cc
  src/ops/sum.cc
  src/padder.cc
  src/profiler.cc
  src/random.cc
  src/sampling.cc
  src/scoring.cc
  src/storage_view.cc
  src/thread_pool.cc
  src/translator.cc
  src/types.cc
  src/utils.cc
  src/vocabulary.cc
  src/vocabulary_map.cc
)

if(WITH_METAL)
  list(APPEND SOURCES
    src/metal/metal_device.mm
    src/metal/metal_utils.mm
    src/metal/metal_allocator.mm
    src/metal/metal_kernels.mm
  )
  
  list(APPEND LIBRARIES
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalKit"
  )
endif()

if (WITH_CUDA)
  find_package(CUDA 11.0 REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  if (WITH_TENSOR_PARALLEL)
    find_package(MPI REQUIRED)
    find_package(NCCL REQUIRED)
    add_definitions(-DCT2_WITH_TENSOR_PARALLEL)
  endif()
  
  if (WITH_FLASH_ATTN)
    add_definitions(-DCT2_WITH_FLASH_ATTN)
  endif()
  
  # Create the main library target
  cuda_add_library(${PROJECT_NAME}
    ${SOURCES}
    src/cuda/allocator.cc
    src/cuda/primitives.cu
    src/cuda/random.cu
    src/cuda/utils.cc
    src/ops/alibi_add_gpu.cu
    src/ops/bias_add_gpu.cu
    src/ops/concat_split_slide_gpu.cu
    src/ops/conv1d_gpu.cu
    src/ops/dequantize_gpu.cu
    src/ops/flash_attention_gpu.cu
    src/ops/gather_gpu.cu
    src/ops/gumbel_max_gpu.cu
    src/ops/layer_norm_gpu.cu
    src/ops/mean_gpu.cu
    src/ops/multinomial_gpu.cu
    src/ops/quantize_gpu.cu
    src/ops/rms_norm_gpu.cu
    src/ops/rotary_gpu.cu
    src/ops/softmax_gpu.cu
    src/ops/tile_gpu.cu
    src/ops/topk_gpu.cu
    src/ops/topp_mask_gpu.cu
    src/ops/nccl_ops_gpu.cu
    src/ops/awq/gemm_gpu.cu
    src/ops/awq/gemv_gpu.cu
    src/ops/awq/dequantize_gpu.cu
  )
  
  if (WITH_FLASH_ATTN)
    target_sources(${PROJECT_NAME} PRIVATE
      src/ops/flash-attention/flash_fwd_hdim32_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim32_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim64_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim64_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim96_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim96_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim128_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim128_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim160_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim160_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim192_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim192_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim224_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim224_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim256_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim256_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim32_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim32_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim64_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim64_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim96_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim96_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim128_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim128_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim160_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim160_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim192_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim192_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim224_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim224_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim256_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim256_fp16_sm80.cu
    )
    
    set_source_files_properties(
      src/ops/flash-attention/flash_fwd_hdim32_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim32_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim64_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim64_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim96_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim96_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim128_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim128_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim160_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim160_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim192_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim192_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim224_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim224_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim256_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_hdim256_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim32_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim32_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim64_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim64_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim96_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim96_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim128_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim128_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim160_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim160_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim192_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim192_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim224_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim224_fp16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim256_bf16_sm80.cu
      src/ops/flash-attention/flash_fwd_split_hdim256_fp16_sm80.cu
      PROPERTIES COMPILE_FLAGS "--use_fast_math")
  endif()

elseif(WITH_CUDNN)
  message(FATAL_ERROR "WITH_CUDNN=ON requires WITH_CUDA=ON")
else()
  add_library(${PROJECT_NAME} ${SOURCES})
endif()

if (WITH_METAL)
  enable_language(OBJCXX)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
  
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
  )
  
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    WITH_METAL
  )
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${CTRANSLATE2_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION ${CTRANSLATE2_MAJOR_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CTRANSLATE2_MAJOR_VERSION})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION
)

list(APPEND LIBRARIES ${CMAKE_DL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
target_include_directories(${PROJECT_NAME} BEFORE
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${PRIVATE_INCLUDE_DIRECTORIES}
)

include_directories(${PROJECT_SOURCE_DIR}/src)

if (WITH_TENSOR_PARALLEL AND CUDA_DYNAMIC_LOADING)
  target_compile_options(${PROJECT_NAME} PRIVATE -DOMPI_SKIP_MPICXX)
endif()

if(BUILD_CLI)
  add_subdirectory(cli)
endif()

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h*"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${CTRANSLATE2_VERSION}
  COMPATIBILITY AnyNewerVersion
)

if(BUILD_SHARED_LIBS)
  export(EXPORT ${PROJECT_NAME}Targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake"
    NAMESPACE CTranslate2::
  )
endif()

configure_file(cmake/${PROJECT_NAME}Config.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake"
  COPYONLY
)

configure_file(cmake/FindNCCL.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/FindNCCL.cmake"
  COPYONLY
)

set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

if(BUILD_SHARED_LIBS)
  install(EXPORT ${PROJECT_NAME}Targets
    FILE
      ${PROJECT_NAME}Targets.cmake
    NAMESPACE
      CTranslate2::
    DESTINATION
      ${ConfigPackageLocation}
  )
endif()

install(
  FILES
    cmake/${PROJECT_NAME}Config.cmake
    cmake/FindNCCL.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)
