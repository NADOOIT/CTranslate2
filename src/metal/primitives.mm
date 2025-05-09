#include "ctranslate2/primitives.h"
#include "ctranslate2/storage_view.h" // For StorageView and Allocator
#include "metal_device.h" // We'll need the MetalDevice class
#include "metal_allocator.h"  // Added for MetalAllocator
#include "utils.h"      // For get_metal_device()

// Include necessary Metal headers
#include <Metal/Metal.h>
#include <stdexcept>

namespace ctranslate2 {

  // Forward declaration if MetalDevice instance management is complex
  // metal::MetalDevice& get_metal_device_instance(int index);

// Explicit definition of the specialized static member function
  void primitives<Device::METAL>::attention(
      StorageView& output,
      const StorageView& queries,
      const StorageView& keys,
      const StorageView& values,
      const StorageView* bias,
      const StorageView* key_lengths,
      const StorageView* query_lengths,
      const float scale,
      const int num_heads,
      const bool causal) {

    // 0. Validate devices and retrieve Metal context
    if (queries.device() != Device::METAL ||
        keys.device() != Device::METAL ||
        values.device() != Device::METAL ||
        output.device() != Device::METAL) {
        throw std::runtime_error("All core StorageViews (queries, keys, values, output) for Metal attention must be on Device::METAL.");
    }
    if (bias && bias->device() != Device::METAL) {
        // Optional bias must also be on Metal if provided
        throw std::runtime_error("Bias StorageView for Metal attention must be on Device::METAL if provided.");
    }
    // Note on key_lengths and query_lengths:
    // Their types and device locations depend on how they are used by the Metal kernel.
    // If they are StorageViews on Metal, device checks would be similar.
    // If they are CPU data or not StorageViews, their handling is different.
    // The current signature assumes they might be StorageViews.

    // Get the allocator for the device of the output StorageView.
    // output.device() should be Device::METAL.
    ctranslate2::Allocator& generic_allocator = ctranslate2::get_allocator(output.device());

    // Cast to MetalAllocator. This assumes the allocator for Device::METAL is indeed a MetalAllocator.
    ctranslate2::metal::MetalAllocator* metal_alloc_ptr = dynamic_cast<ctranslate2::metal::MetalAllocator*>(&generic_allocator);
    if (!metal_alloc_ptr) {
        // Fallback or error if dynamic_cast fails. This indicates a setup or architectural mismatch.
        throw std::runtime_error("Failed to cast Allocator to MetalAllocator for device: " +
                                 ctranslate2::device_to_str(output.device(), output.device_index()) +
                                 ". Ensure Metal backend is correctly initialized and allocator is a MetalAllocator.");
    }

    // Get the MetalDevice from the MetalAllocator.
    // The MetalAllocator should hold a reference to its specific MetalDevice instance.
    const ctranslate2::metal::MetalDevice& md = metal_alloc_ptr->device();

    // Now get the command queue and pipeline state from the obtained MetalDevice.
    id<MTLCommandQueue> queue = md.getCommandQueue();
    id<MTLComputePipelineState> attention_pipeline = md.getAttentionPipeline();

    if (queue == nil) {
        throw std::runtime_error("Metal command queue is nil");
    }
    if (attention_pipeline == nil) {
        throw std::runtime_error("Attention pipeline state is nil");
    }

    // 2. Get Command Buffer & Encoder
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
    if (compute_encoder == nil) {
        throw std::runtime_error("Failed to create Metal compute command encoder");
    }
    [compute_encoder setComputePipelineState:attention_pipeline]; // Use passed-in pipeline

    // 3. Get MTLBuffers from StorageViews
    // The void* in StorageView IS the id<MTLBuffer> (retained)
    id<MTLBuffer> queries_buffer = (__bridge id<MTLBuffer>)queries.buffer();
    id<MTLBuffer> keys_buffer = (__bridge id<MTLBuffer>)keys.buffer();
    id<MTLBuffer> values_buffer = (__bridge id<MTLBuffer>)values.buffer();
    id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output.buffer();
    id<MTLBuffer> bias_buffer = bias ? (__bridge id<MTLBuffer>)bias->buffer() : nil;
    // Note: key_lengths and query_lengths are not directly used by this kernel signature.
    // Masking/length handling is likely done via the 'mask' (bias) buffer or within the kernel.

    // 4. Set pipeline state and bind arguments (matching kernel signature)
    int buffer_index = 0;
    [compute_encoder setBuffer:queries_buffer offset:0 atIndex:buffer_index++];   // 0: query
    [compute_encoder setBuffer:keys_buffer offset:0 atIndex:buffer_index++];    // 1: key
    [compute_encoder setBuffer:values_buffer offset:0 atIndex:buffer_index++];   // 2: value
    [compute_encoder setBuffer:bias_buffer offset:0 atIndex:buffer_index++];    // 3: mask (using bias)
    [compute_encoder setBuffer:output_buffer offset:0 atIndex:buffer_index++];   // 4: output

    // Bind dimensions and parameters struct
    struct AttentionParams {
        uint32_t batch_size;
        uint32_t num_heads;
        uint32_t query_seq_len;
        uint32_t key_seq_len;
        uint32_t head_dim;
        float scale;
        uint32_t is_causal; // Match kernel type
    };

    AttentionParams params;
    // TODO: These shape assumptions might be incorrect depending on layout (e.g., NCHW vs NHWC)
    params.batch_size = static_cast<uint32_t>(queries.dim(0));
    params.num_heads = static_cast<uint32_t>(num_heads);
    params.query_seq_len = static_cast<uint32_t>(queries.dim(1));
    params.key_seq_len = static_cast<uint32_t>(keys.dim(1));
    params.head_dim = static_cast<uint32_t>(queries.dim(3)); // Assuming dim -1 is head_dim
    params.scale = scale;
    params.is_causal = static_cast<uint32_t>(causal); // Cast bool to uint32_t

    [compute_encoder setBytes:&params length:sizeof(AttentionParams) atIndex:buffer_index++]; // 5: params

    // 5. Dispatch kernel using dispatchThreadgroups
    // Grid size covers total query positions (Q) and batch items (B).
    // Each thread calculates output for one (query_pos, batch_idx) across all heads.
    MTLSize grid_size = MTLSizeMake(params.query_seq_len, params.batch_size, 1);

    // Threadgroup size - requires tuning based on kernel logic & device.
    // This placeholder might be okay, but optimal perf needs experimentation.
    // Also consider kernel's potential use of threadgroup memory (e.g., scores[1024]).
    MTLSize threadgroup_size = MTLSizeMake(16, 16, 1); // Placeholder, needs tuning!

    // Calculate the number of threadgroups needed in each dimension.
    MTLSize num_threadgroups = MTLSizeMake(
        (grid_size.width + threadgroup_size.width - 1) / threadgroup_size.width,
        (grid_size.height + threadgroup_size.height - 1) / threadgroup_size.height,
        1); // Z dimension is 1

    [compute_encoder dispatchThreadgroups:num_threadgroups threadsPerThreadgroup:threadgroup_size];

    // 6. End encoding and commit
    [compute_encoder endEncoding];
    [command_buffer commit];
    // Optional: Synchronize immediately if needed
    // [commandBuffer waitUntilCompleted];

    // Note: StorageView's void* holds a retained buffer, so no explicit release here.
    // The Allocator::free call will handle the CFRelease.
  }

  // =================== Other Primitives (Placeholders) ===================

  // Define other necessary primitive specializations here...
  // e.g., gemm, layer_norm, activation functions, etc.
  // If not defined, they might fall back to CPU via inheritance (needs verification).

} // namespace ctranslate2
