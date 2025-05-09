#pragma once

#include <stdexcept>

#include "ctranslate2/devices.h"

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr ctranslate2::Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__

#ifndef CT2_WITH_CUDA
#  ifndef CT2_WITH_METAL
#    define DEVICE_DISPATCH(DEVICE, STMTS)                \
     switch (DEVICE) {                                    \
       UNSUPPORTED_DEVICE_CASE(ctranslate2::Device::CUDA)              \
       UNSUPPORTED_DEVICE_CASE(ctranslate2::Device::METAL)             \
       DEVICE_CASE(ctranslate2::Device::CPU, SINGLE_ARG(STMTS))        \
     }
#  else
#    define DEVICE_DISPATCH(DEVICE, STMTS)                \
     switch (DEVICE) {                                    \
       UNSUPPORTED_DEVICE_CASE(ctranslate2::Device::CUDA)              \
       DEVICE_CASE(ctranslate2::Device::METAL, SINGLE_ARG(STMTS))      \
       DEVICE_CASE(ctranslate2::Device::CPU, SINGLE_ARG(STMTS))        \
     }
#  endif
#else
#  ifndef CT2_WITH_METAL
#    define DEVICE_DISPATCH(DEVICE, STMTS)                \
     switch (DEVICE) {                                    \
       DEVICE_CASE(ctranslate2::Device::CUDA, SINGLE_ARG(STMTS))       \
       UNSUPPORTED_DEVICE_CASE(ctranslate2::Device::METAL)             \
       DEVICE_CASE(ctranslate2::Device::CPU, SINGLE_ARG(STMTS))        \
     }
#  else
#    define DEVICE_DISPATCH(DEVICE, STMTS)                \
     switch (DEVICE) {                                    \
       DEVICE_CASE(ctranslate2::Device::CUDA, SINGLE_ARG(STMTS))       \
       DEVICE_CASE(ctranslate2::Device::METAL, SINGLE_ARG(STMTS))      \
       DEVICE_CASE(ctranslate2::Device::CPU, SINGLE_ARG(STMTS))        \
     }
#  endif
#endif
