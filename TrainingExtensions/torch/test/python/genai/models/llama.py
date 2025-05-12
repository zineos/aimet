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
""" Llama model class """

import types
import torch
from copy import deepcopy
import functools

from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, PreTrainedModel
from transformers.models.llama import modeling_llama

from aimet_common.defs import QuantScheme
from aimet_torch import QuantizationSimModel
from aimet_torch.v2.utils import remove_param_quantizers
from aimet_torch.v2.nn.transformers.models.llama.modeling_llama import QuantizedLlamaRMSNorm

from .genai_model import GenAIModel
from .utils.model_utils import TorchExportableModuleWithCache


class Llama_32(GenAIModel):
    """ Generic LLaMa 3.2 """
    DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    @classmethod
    def _instantiate_model(cls, model_id: str, small_model: bool = False) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
        return modeling_llama.LlamaForCausalLM.from_pretrained(model_id, config=llm_config)

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    @classmethod
    def instantiate_quantsim(cls, model_id: str, context_length: int, sequence_length: int, small_model: bool = False) -> QuantizationSimModel:
        model = cls._instantiate_model(model_id, small_model)

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
            if isinstance(module, QuantizedLlamaRMSNorm):
                module.param_quantizers["weight"].bitwidth = 16

        return quantsim
