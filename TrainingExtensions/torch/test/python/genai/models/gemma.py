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
""" Qwen model class """

import types
import torch

from transformers import AutoTokenizer, AutoConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.models.gemma3 import modeling_gemma3
from transformers.activations import PytorchGELUTanh

from aimet_common.defs import QuantScheme
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.utils import remove_param_quantizers

from .genai_model import GenAIModel
from .utils.model_utils import TorchExportableModuleWithCache

if QuantizationMixin.cls_to_qcls.get(modeling_gemma3.Gemma3RMSNorm, None) is None:
    @QuantizationMixin.implements(modeling_gemma3.Gemma3RMSNorm)
    class QuantizedQwen2RMSNorm(QuantizationMixin, modeling_gemma3.Gemma3RMSNorm):
        """ Implement Quantized Gemma RMSNorm """
        def __quant_init__(self):
            # pylint: disable=useless-parent-delegation
            super().__quant_init__()

            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])
            self.param_quantizers = torch.nn.ModuleDict({"weight": None})

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                hidden_states = self.input_quantizers[0](hidden_states)

            with self._patch_quantized_parameters():
                ret = super().forward(hidden_states)

            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret


if QuantizationMixin.cls_to_qcls.get(modeling_gemma3.Gemma3RotaryEmbedding, None) is None:
    @QuantizationMixin.implements(modeling_gemma3.Gemma3RotaryEmbedding)
    class QuantizedQwen2RotaryEmbedding(QuantizationMixin, modeling_gemma3.Gemma3RotaryEmbedding):
        """ Implement Quantized Gemma Rotary Embedding """
        def __quant_init__(self):
            # pylint: disable=useless-parent-delegation
            super().__quant_init__()

        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
            # pylint: disable=arguments-differ
            return super().forward(x, position_ids)


if QuantizationMixin.cls_to_qcls.get(modeling_gemma3.Gemma3TextScaledWordEmbedding, None) is None:
    @QuantizationMixin.implements(modeling_gemma3.Gemma3TextScaledWordEmbedding)
    class QuantizedGemma3TextScaledWordEmbedding(QuantizationMixin, modeling_gemma3.Gemma3TextScaledWordEmbedding):
        """ Implement Quantized Gemma Text Scaled Word Embedding """
        def __quant_init__(self):
            # pylint: disable=useless-parent-delegation
            super().__quant_init__()

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # pylint: disable=arguments-differ
            return super().forward(input_ids)


if QuantizationMixin.cls_to_qcls.get(PytorchGELUTanh, None) is None:
    @QuantizationMixin.implements(PytorchGELUTanh)
    class QuantizedPytorchGELUTanh(QuantizationMixin, PytorchGELUTanh):
        """ Implement Quantized Transformers PytorchGELUTanh function """
        def __quant_init__(self):
            # pylint: disable=useless-parent-delegation
            super().__quant_init__()

            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # pylint: disable=arguments-differ
            if self.input_quantizers[0]:
                hidden_states = self.input_quantizers[0](hidden_states)

            with self._patch_quantized_parameters():
                ret = super().forward(hidden_states)

            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret


class Gemma_3_1b(GenAIModel):
    """ Generic quantized Gemma 3 """
    MODEL_ID = "google/gemma-3-1b-it"

    @classmethod
    def _instantiate_model(cls, small_model=False) -> PreTrainedModel:
        llm_config = AutoConfig.from_pretrained(cls.MODEL_ID, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
        model = modeling_gemma3.Gemma3ForCausalLM.from_pretrained(cls.MODEL_ID, config=llm_config)

        for decoder in model.model.layers:
            decoder.is_sliding = decoder.self_attn.is_sliding = False

        return model

    @classmethod
    def instantiate_tokenizer(cls) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(cls.MODEL_ID, use_fast=True, trust_remote_code=True)

    @classmethod
    def instantiate_quantsim(cls, context_length, sequence_length) -> QuantizationSimModel:
        model = cls._instantiate_model()

        # Need to wrap model in this in order to enable JIT trace
        traceable_model = TorchExportableModuleWithCache(model)

        dummy_input_ids = torch.zeros((1, sequence_length), dtype=torch.int)
        dummy_attention_mask = torch.ones((1, sequence_length), dtype=torch.int)

        assembled_dummy_inputs = cls.static_graph_prepare_inputs(
            traceable_model,
            dummy_input_ids,
            dummy_attention_mask,
            {"past_key_values": None},
            context_length,
            sequence_length
        )

        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=tuple(assembled_dummy_inputs.values()),
            default_output_bw=16,
            default_param_bw=4,
            in_place=True,
            config_file=cls.get_quantsim_config()
        )

        quantsim.model.orig_forward = quantsim.model.forward
        quantsim.model.forward = types.MethodType(
            cls.create_static_graph_forward(context_length, sequence_length),
            quantsim.model
        )
        quantsim.model.prepare_inputs_for_generation = types.MethodType(
            cls.static_graph_prepare_inputs_for_generation,
            quantsim.model
        )

        remove_param_quantizers(quantsim.model.model.model.embed_tokens)
        quantsim.model.model.lm_head.param_quantizers["weight"].bitwidth = 8
        for _, module in quantsim.model.named_modules():
            if isinstance(module, QuantizationMixin.cls_to_qcls[modeling_gemma3.Gemma3RMSNorm]):
                module.param_quantizers["weight"].bitwidth = 16

        return quantsim
