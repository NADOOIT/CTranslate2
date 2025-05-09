#include "ctranslate2/devices.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif
#ifdef CT2_WITH_METAL
#  include "metal/utils.h"
#endif
#ifdef CT2_WITH_TENSOR_PARALLEL
#  include <unistd.h>
#endif

#include "device_dispatch.h"

#ifdef CT2_WITH_METAL
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// This namespace block is specifically for the Metal utilities moved here.
// It's distinct from the broader ctranslate2 namespace below.
namespace ctranslate2 {
namespace metal {
  // Static variables previously in utils.mm
  static id<MTLDevice> current_device_cached = nil;
  static id<MTLCommandQueue> command_queue_cached = nil;
  static std::vector<id<MTLDevice>> available_devices_cached;

  // Forward declaration
  void init_metal_internal();

  // Implementation of has_metal
  bool has_metal() {
    @autoreleasepool {
      return MTLCreateSystemDefaultDevice() != nil;
    }
  }

  // Implementation of get_metal_device_count
  int get_metal_device_count() {
    @autoreleasepool {
      if (available_devices_cached.empty()) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices) {
          for (id<MTLDevice> device_item in devices) {
            available_devices_cached.push_back(device_item);
          }
            // [devices release]; // Release since MTLCopyAllDevices has "Copy" - Removed for ARC
        } else if (MTLCreateSystemDefaultDevice() != nil) {
           // Fallback if MTLCopyAllDevices returns nil (e.g. on newer macOS versions where it might be restricted)
           // and a default device exists.
           available_devices_cached.push_back(MTLCreateSystemDefaultDevice());
        }
      }
      return available_devices_cached.size();
    }
  }

  // Implementation of init_metal_internal
  void init_metal_internal() {
    @autoreleasepool {
      if (!has_metal()) { 
        throw std::runtime_error("No Metal-capable device found (checked in devices.mm)");
      }
      if (current_device_cached == nil) {
        current_device_cached = MTLCreateSystemDefaultDevice();
        if (current_device_cached == nil) {
          throw std::runtime_error("Failed to create Metal device (checked in devices.mm)");
        }
      }
      if (command_queue_cached == nil) {
        command_queue_cached = [current_device_cached newCommandQueue];
        if (command_queue_cached == nil) {
          throw std::runtime_error("Failed to create Metal command queue (checked in devices.mm)");
        }
      }
    }
  }

  // Helper to get current device, ensures initialization
  id<MTLDevice> get_current_metal_device_cached() {
    if (current_device_cached == nil) {
      init_metal_internal();
    }
    return current_device_cached;
  }

  // Implementation of set_metal_device
  void set_metal_device(int index) {
    @autoreleasepool {
      get_metal_device_count(); 
      if (index < 0 || static_cast<size_t>(index) >= available_devices_cached.size()) {
        throw std::runtime_error("Invalid Metal device index: " + std::to_string(index) + " (in devices.mm)");
      }
      if (command_queue_cached != nil) {
        // [command_queue_cached release]; // Removed for ARC
        command_queue_cached = nil;
      }
      current_device_cached = available_devices_cached[index];
      if (current_device_cached) {
          command_queue_cached = [current_device_cached newCommandQueue];
          if (command_queue_cached == nil) {
               throw std::runtime_error("Failed to create Metal command queue for new device (in devices.mm)");
          }
      } else {
        // If index led to a nil device somehow, ensure command_queue is also nil.
        command_queue_cached = nil;
      }
    }
  }

  // Implementation of synchronize_device
  void synchronize_device() {
    @autoreleasepool {
      get_current_metal_device_cached(); 
      if (command_queue_cached != nil) {
        id<MTLCommandBuffer> command_buffer = [command_queue_cached commandBuffer];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }
\
  }

  // Public version of init_metal, calls internal version
  void init_metal() {
    init_metal_internal();
  }

  // Public version of get_metal_device, calls cached version
  id<MTLDevice> get_metal_device() {
    return get_current_metal_device_cached();
  }

} // namespace metal
} // namespace ctranslate2
#endif // CT2_WITH_METAL

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
    if (device == "cuda" || device == "CUDA")
#ifdef CT2_WITH_CUDA
      return Device::CUDA;
#else
      throw std::invalid_argument("This CTranslate2 package was not compiled with CUDA support");
#endif
    if (device == "metal" || device == "METAL" || device == "mps" || device == "MPS")
#ifdef CT2_WITH_METAL
      return Device::METAL;
#else
      throw std::invalid_argument("This CTranslate2 package was not compiled with Metal support");
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
    if (device == "auto" || device == "AUTO") {
#ifdef CT2_WITH_CUDA
      if (cuda::has_gpu())
        return Device::CUDA;
#endif
#ifdef CT2_WITH_METAL
      if (metal::has_metal())
        return Device::METAL;
#endif
      return Device::CPU;
    }
    throw std::invalid_argument("unsupported device " + device);
  }

  std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "cuda";
    case Device::METAL:
      return "metal";
    case Device::CPU:
      return "cpu";
    }
    return "";
  }

  std::string device_to_str(Device device, int index) {
    return device_to_str(device) + ":" + std::to_string(index);
  }

  int get_device_count(Device device) {
    switch (device) {
    case Device::CUDA:
#ifdef CT2_WITH_CUDA
      return cuda::get_gpu_count();
#else
      return 0;
#endif
    case Device::METAL:
#ifdef CT2_WITH_METAL
      return metal::get_metal_device_count();
#else
      return 0;
#endif
    case Device::CPU:
      return 1;
    }
    return 0;
  }

  template <Device D>
  int get_device_index();
  template <Device D>
  void set_device_index(int index);

  template<>
  int get_device_index<Device::CPU>() {
    return 0;
  }

  template<>
  void set_device_index<Device::CPU>(int index) {
    if (index != 0)
      throw std::invalid_argument("Invalid CPU device index: " + std::to_string(index));
  }

#ifdef CT2_WITH_CUDA
  template<>
  int get_device_index<Device::CUDA>() {
    int index = 0;
    CUDA_CHECK(cudaGetDevice(&index));
    return index;
  }

  template<>
  void set_device_index<Device::CUDA>(int index) {
    CUDA_CHECK(cudaSetDevice(index));
  }
#endif

#ifdef CT2_WITH_METAL
  template<>
  int get_device_index<Device::METAL>() {
    return metal::get_metal_device_count() > 0 ? 0 : -1;
  }

  template<>
  void set_device_index<Device::METAL>(int index) {
    metal::set_metal_device(index);
  }
#endif

  int get_device_index(Device device) {
    int index = 0;
    DEVICE_DISPATCH(device, index = get_device_index<D>());
    return index;
  }

  void set_device_index(Device device, int index) {
    DEVICE_DISPATCH(device, set_device_index<D>(index));
  }

  void synchronize_device(Device device, int index) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA) {
      const ScopedDeviceSetter scoped_device_setter(device, index);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
#ifdef CT2_WITH_METAL
    if (device == Device::METAL) {
      const ScopedDeviceSetter scoped_device_setter(device, index);
      metal::synchronize_device();
    }
#endif
  }

  void synchronize_stream(Device device) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA)
      CUDA_CHECK(cudaStreamSynchronize(cuda::get_cuda_stream()));
#endif
#ifdef CT2_WITH_METAL
    if (device == Device::METAL)
      metal::synchronize_device();
#endif
  }

  // Initialize the static member variable
#ifdef CT2_WITH_TENSOR_PARALLEL
    std::vector<ncclComm_t*> ScopedMPISetter::_nccl_comms;
#endif
  int my_rank = 0;
  int local_rank = 0;
  int n_ranks = 1;

  ScopedMPISetter::ScopedMPISetter() {
#ifdef CT2_WITH_TENSOR_PARALLEL
    // initializing MPI
    MPI_CHECK(MPI_Init(nullptr, nullptr));
    MPI_CHECK(MPI_Comm_rank(STUB_MPI_COMM_WORLD, &my_rank));
    MPI_CHECK(MPI_Comm_size(STUB_MPI_COMM_WORLD, &n_ranks));

    uint64_t hostHashs[n_ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[my_rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, STUB_MPI_DATATYPE_NULL,
                           hostHashs, sizeof(uint64_t), STUB_MPI_BYTE, STUB_MPI_COMM_WORLD));
    for (int p = 0; p < n_ranks; p++) {
      if (p == my_rank) {
        break;
      }
      if (hostHashs[p] == hostHashs[my_rank]) {
        local_rank++;
      }
    }
    atexit(finalize);
#endif
  }

  ScopedMPISetter::~ScopedMPISetter() = default;

#ifdef CT2_WITH_TENSOR_PARALLEL
  uint64_t ScopedMPISetter::getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
      result = ((result << 5) + result) + string[c];
    }
    return result;
    }

  void ScopedMPISetter::getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
      if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
      }
    }
  }

  ncclComm_t ScopedMPISetter::getNcclComm() {
    static thread_local ncclComm_t comm;
    static thread_local ncclUniqueId id;

    if (comm == nullptr) {
      int nRanks = ScopedMPISetter::getNRanks();
      int myRank = ScopedMPISetter::getCurRank();
      if (myRank == 0) {
          ncclGetUniqueId(&id);
      }
      MPI_CHECK(MPI_Bcast((void *) &id, sizeof(id), STUB_MPI_BYTE, 0, STUB_MPI_COMM_WORLD));
      NCCL_CHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
      _nccl_comms.push_back(&comm);
    }
    return comm;
  }
#endif

  void ScopedMPISetter::finalize() {
#ifdef CT2_WITH_TENSOR_PARALLEL
    for (auto* comm : _nccl_comms) {
        //finalizing NCCL
        if (*comm) {
          NCCL_CHECK(ncclCommFinalize(*comm));
          NCCL_CHECK(ncclCommDestroy(*comm));
        }
    }
    MPI_CHECK(MPI_Finalize());
#endif
  }

  int ScopedMPISetter::getNRanks() {
    return n_ranks;
  }

  int ScopedMPISetter::getCurRank() {
    return my_rank;
  }

  int ScopedMPISetter::getLocalRank() {
    return local_rank;
  }
}
