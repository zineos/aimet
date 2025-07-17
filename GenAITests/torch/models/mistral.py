# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Mistral model class"""

import contextlib
from torch import nn
import torch

from aimet_common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.nn.transformers.models.mistral.modeling_mistral import (
    QuantizedMistralRMSNorm,
)

from transformers.models.mistral import modeling_mistral
from aimet_torch.nn.modules.custom import Add
from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.mistral import Mistral_03
from GenAITests.shared.models.generator import Generator
from GenAITests.shared.models.utils.model_utils import ONNXExportableModuleWithCache


class MistralDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = modeling_mistral.MistralAttention(
            config=config, layer_idx=layer_idx
        )
        self.mlp = modeling_mistral.MistralMLP(config)
        self.input_layernorm = modeling_mistral.MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = modeling_mistral.MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_add = Add()
        self.mlp_add = Add()

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,  # necessary, but kept here for BC
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.attn_add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_add(residual, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@contextlib.contextmanager
def swap_decoder_module():
    old_decoder = modeling_mistral.MistralDecoderLayer
    modeling_mistral.MistralDecoderLayer = MistralDecoderLayer

    try:
        yield
    finally:
        modeling_mistral.MistralDecoderLayer = old_decoder


@YAMLConfigParser.register_model
class Mistral_03_Torch(Mistral_03):
    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
    ) -> QuantizationSimModel:
        with swap_decoder_module():
            model = cls.instantiate_model(model_id, small_model)

        # Need to wrap model in this in order to enable JIT trace
        traceable_model = ONNXExportableModuleWithCache(model)

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = Generator.prepare_inputs(
            model=traceable_model,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            past_key_values=[],
            context_length=context_length,
            sequence_length=sequence_length,
        )

        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=assembled_dummy_inputs,
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in quantsim.model.named_modules():
            if isinstance(module, QuantizedMistralRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        return quantsim
