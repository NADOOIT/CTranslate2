from typing import List, Optional, Tuple

import numpy as np

from ctranslate2.specs import common_spec, model_spec, transformer_spec


class Wav2Vec2Config(model_spec.ModelConfig):
    """Configuration for the Wav2Vec2 model."""

    def __init__(self):
        return


class Wav2Vec2Spec(model_spec.LanguageModelSpec):
    def __init__(self, feat_layers, num_layers, num_heads):
        super().__init__()
        self.encoder = Wav2Vec2EncoderSpec(feat_layers, num_layers, num_heads)

    @property
    def name(self):
        return "Wav2Vec2Spec"

    @property
    def revision(self):
        return 3

    def get_default_config(self):
        return Wav2Vec2Config()

    def get_vocabulary_size(self):
        return self.encoder.lm_head.weight.shape[0]


class Wav2Vec2LayerNormConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()
        self.layer_norm = common_spec.LayerNormSpec()


class Wav2Vec2PosEmbedConvLayer(model_spec.LayerSpec):
    def __init__(self):
        self.conv = common_spec.Conv1DSpec()


class Wav2Vec2EncoderSpec(model_spec.LayerSpec):
    def __init__(self, feat_layers, num_layers, num_heads):
        self.num_heads = np.dtype("int16").type(num_heads)
        self.feat_layer0 = Wav2Vec2LayerNormConvLayer()
        self.feat_layer = [Wav2Vec2LayerNormConvLayer() for i in range(feat_layers - 1)]
        self.fp_layer_norm = common_spec.LayerNormSpec()
        self.fp_projection = common_spec.LinearSpec()
        self.pos_conv_embed = Wav2Vec2PosEmbedConvLayer()
        self.layer_norm = common_spec.LayerNormSpec()
        self.layer = [transformer_spec.TransformerEncoderLayerSpec() for _ in range(num_layers)]
        self.lm_head = common_spec.LinearSpec()
