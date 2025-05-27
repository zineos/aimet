# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
"""Qwen model class"""

import types
import torch

from transformers import AutoTokenizer, AutoConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.models.gemma3 import modeling_gemma3
from transformers.cache_utils import HybridCache

from aimet_common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.utils import remove_param_quantizers
from aimet_torch.v2.nn.transformers.models.gemma3.modeling_gemma3 import (
    QuantizedGemma3RMSNorm,
)

from .genai_model import GenAIModel
from .utils.model_utils import TorchExportableModuleWithCache


class TorchExportableModuleWithHybridCache(TorchExportableModuleWithCache):
    def __init__(self, model: PreTrainedModel, cache_length: int):
        super().__init__(model)
        self.cache_length = cache_length

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        """Redefine model forward to convert to/from Huggingface HybridCache objects"""
        cache = HybridCache(
            config=self.model.config,
            batch_size=1,
            max_cache_len=self.cache_length,
            device=self.model.device,
            dtype=self.model.dtype,
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                print(
                    key_states.shape,
                    value_states.shape,
                    layer_idx,
                    cache.key_cache[layer_idx].shape,
                    cache.value_cache[layer_idx].shape,
                )
                cache.update(
                    key_states,
                    value_states,
                    layer_idx,
                    {
                        "cache_position": torch.arange(self.cache_length),
                        "sliding_window": self.model.config.sliding_window,
                    },
                )

        lm_logits, new_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            num_logits_to_return=0,
            return_dict=False,
            *args,
            **kwargs,
        )

        legacy_cache = ()
        for layer_idx in range(len(past_key_values)):
            legacy_cache += (
                (cache.key_cache[layer_idx], cache.value_cache[layer_idx]),
            )

        return lm_logits, legacy_cache


class Gemma_3(GenAIModel):
    """Generic quantized Gemma 3"""

    DEFAULT_MODEL_ID = "google/gemma-3-1b-it"

    @classmethod
    def _instantiate_model(cls, model_id: str, small_model=False) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2

        model = modeling_gemma3.Gemma3ForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

        return model

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def instantiate_quantsim(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int,
        small_model: bool = False,
    ) -> QuantizationSimModel:
        model = cls._instantiate_model(model_id, small_model)

        # Need to wrap model in this in order to enable JIT trace
        traceable_model = TorchExportableModuleWithHybridCache(
            model, context_length - sequence_length
        )

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = cls.static_graph_prepare_inputs(
            traceable_model,
            dummy_input_ids,
            dummy_attention_mask,
            {"past_key_values": None},
            context_length,
            sequence_length,
        )

        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=tuple(assembled_dummy_inputs.values()),
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config(),
        )

        quantsim.model.orig_forward = quantsim.model.forward
        quantsim.model.forward = types.MethodType(
            cls.create_static_graph_forward(context_length, sequence_length),
            quantsim.model,
        )
        quantsim.model.prepare_inputs_for_generation = types.MethodType(
            cls.static_graph_prepare_inputs_for_generation, quantsim.model
        )

        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in quantsim.model.named_modules():
            if isinstance(module, QuantizedGemma3RMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        return quantsim
