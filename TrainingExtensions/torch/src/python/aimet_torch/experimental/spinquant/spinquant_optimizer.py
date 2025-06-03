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
"""Optimizer for Spinquant"""

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from aimet_common.utils import AimetLogger

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

RMSNORM_LINEAR_PAIRS = "RMSNORM_LINEAR_PAIRS"
R1_LINEAR_FUSION = "R1_LINEAR_FUSION"

# Dictionary of supported modules and associated information for RMSNORM Linear fusion pairs as well as R1 Linear fusion pairs.
SUPPORTED_MODULE_DICT = {
    LlamaForCausalLM: {
        RMSNORM_LINEAR_PAIRS: lambda module: [
            (
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            for layer in module.model.layers
        ]
        + [(module.model.norm, [module.lm_head])],
        R1_LINEAR_FUSION: lambda _: [],  # TODO: fill this in when R1 fusion is implemented
    },
    Qwen2ForCausalLM: {
        RMSNORM_LINEAR_PAIRS: lambda module: [
            (
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            for layer in module.model.layers
        ]
        + [(module.model.norm, [module.lm_head])],
        R1_LINEAR_FUSION: lambda _: [],  # TODO: fill this in when R1 fusion is implemented
    },
}


def apply_spinquant(model: torch.nn.Module):
    """
    Apply SpinQuant to the model, modifying weights in place. https://arxiv.org/pdf/2405.16406
    Currently only R1 rotations without optimization are supported.

    :param model: The model to apply SpinQuant to.
    """
    # Ensure that any user registered module types are fully registered
    for supported_module_type, module_info in SUPPORTED_MODULE_DICT.items():
        if module_info.get(RMSNORM_LINEAR_PAIRS) is None:
            raise RuntimeError(
                f"{RMSNORM_LINEAR_PAIRS} info missing for module type {supported_module_type.__name__}"
            )
        if module_info.get(R1_LINEAR_FUSION) is None:
            raise RuntimeError(
                f"{R1_LINEAR_FUSION} info missing for module type {supported_module_type.__name__}"
            )

    found_module = False
    for module in model.modules():
        if isinstance(module, tuple(SUPPORTED_MODULE_DICT.keys())):
            if module.model.embed_tokens.weight is module.lm_head.weight:
                raise RuntimeError(
                    "SpinQuant requires embed_tokens and lm_head weights to be untied. Ensure that model.config.tie_word_embeddings or a similar relevant "
                    "setting is set to False for the model."
                )

            found_module = True
            _identify_and_fuse_rmsnorms_into_linears(module)

            # TODO: Add R1 fusion here

    if not found_module:
        _logger.warning(
            "SpinQuant optimizer did not find any modules to apply SpinQuant for in the model."
        )


def _identify_and_fuse_rmsnorms_into_linears(llm_model: torch.nn.Module):
    for rmsnorm, linears in SUPPORTED_MODULE_DICT[type(llm_model)][
        RMSNORM_LINEAR_PAIRS
    ](llm_model):
        _fuse_rmsnorm_into_linear(rmsnorm, linears)


def _fuse_rmsnorm_into_linear(rmsnorm, linear_layers):
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        W = linear.weight.data
        linear.weight.data = (W * rmsnorm.weight).to(linear_dtype)

        if hasattr(rmsnorm, "bias"):
            if linear.bias is not None:
                linear.bias.data = linear.bias.data.double() + torch.matmul(
                    W, rmsnorm.bias.double()
                )
                linear.bias.data = linear.bias.data.to(linear_dtype)

    rmsnorm.weight.data = torch.ones_like(rmsnorm.weight.data)
