cmake_minimum_required(VERSION 3.10)
project(CTranslate2ThirdParty)

# Disable problematic GoogleTest features
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_DISABLE_PTHREADS=1")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Minimal configuration to prevent build errors
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# Add GoogleTest as a direct subdirectory
add_subdirectory(googletest/googletest)

# Explicitly set up gtest and gtest_main targets
if(TARGET gtest)
  set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()
if(TARGET gtest_main)
  set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()

# Prevent building any gmock-related targets
set(UNWANTED_TARGETS 
  gmock gmock_main gmock-main gmock_test gmock_builder
  gtest_test
)
foreach(target ${UNWANTED_TARGETS})
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endforeach()
