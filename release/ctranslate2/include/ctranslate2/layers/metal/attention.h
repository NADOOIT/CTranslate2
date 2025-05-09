#pragma once

#include <ctranslate2/layers/attention.h>
#include <ctranslate2/models/model.h> // For layer_spec type (assuming it's ModelReader or similar)

// Forward declarations for Metal types if needed, or include Metal headers
// #include <Metal/Metal.h>

namespace ctranslate2 {
  namespace layers {
    namespace metal { // Match the namespace from the error message

      class MultiHeadAttentionMetal : public MultiHeadAttention {
      public:
        // Constructor signature based on the call inferred from transformer.h error
        // NOTE: The type for 'layer_spec' needs verification. Assuming Model for now.
        MultiHeadAttentionMetal(const ctranslate2::models::Model& model,
                                const std::string& scope,
                                dim_t num_heads,
                                bool self_attention,
                                bool pre_norm = true,
                                bool is_decoder = false,
                                ctranslate2::layers::Alibi* alibi = nullptr);

        ~MultiHeadAttentionMetal() override = default;

        // Override the virtual operator() from the base class
        void operator()(const StorageView& queries,
                        const StorageView* memory,
                        const StorageView* memory_lengths,
                        StorageView& output,
                        StorageView* cached_keys = nullptr,
                        StorageView* cached_values = nullptr,
                        StorageView* attention = nullptr,
                        const Padder* queries_padder = nullptr,
                        const Padder* memory_padder = nullptr,
                        bool use_flash_attention = false);
                        
        // Override other virtual methods if necessary
        DataType output_type() const override;
        dim_t output_size() const override;

      private:
        // Placeholder for Metal-specific members
        bool _self_attention;
        bool _multi_query;
      };

    } // namespace metal
  } // namespace layers
} // namespace ctranslate2
