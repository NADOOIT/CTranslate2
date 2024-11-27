#pragma once

#include <string>
#include <memory>
#include <vector>
#include "ctranslate2/types.h"
#include "ctranslate2/devices.h"

namespace ctranslate2 {

  class IDevice {
  public:
    virtual ~IDevice() = default;
    virtual std::string name() const = 0;
    virtual void synchronize() = 0;
    virtual void synchronize_stream(void* stream) = 0;
    virtual void* allocate(std::size_t size) const = 0;
    virtual void free(void* data) const = 0;
    virtual void* allocate_stream() const = 0;
    virtual void free_stream(void* stream) const = 0;
  };

  class DeviceBackend {
  public:
    virtual ~DeviceBackend() = default;
    virtual Device type() const = 0;
    virtual int index() const = 0;
    virtual bool supports_model(const std::string& model_type) const = 0;
    virtual bool supports_int8() const = 0;
    virtual bool supports_int16() const = 0;
    virtual bool supports_float16() const = 0;
    virtual bool supports_bfloat16() const = 0;
    virtual bool supports_mixed_precision() const = 0;
    virtual bool supports_packed_gemm() const = 0;
    virtual bool supports_quant_gemm() const = 0;
    virtual bool supports_inplace_add() const = 0;
  };

} // namespace ctranslate2
