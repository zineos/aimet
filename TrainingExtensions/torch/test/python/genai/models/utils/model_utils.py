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
"""Utils for building GenAI models"""

import torch
from transformers import PreTrainedModel, DynamicCache


class TorchExportableModuleWithCache(torch.nn.Module):
    """
    Helper class to enable Torch JIT trace and ONNX export of HuggingFace models that produce and consume Cache objects
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    @property
    def device(self):
        """Return model device"""
        return self.model.device

    @property
    def dtype(self):
        """Return model dtype"""
        return self.model.dtype

    @property
    def config(self):
        """Return model config"""
        return self.model.config

    # pylint: disable=keyword-arg-before-vararg
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: DynamicCache = None,
        *args,
        **kwargs,
    ):
        """Redefine model forward to convert to/from Huggingface DynamicCache objects"""
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        lm_logits, new_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            num_logits_to_return=0,
            return_dict=False,
            *args,
            **kwargs,
        )
        return lm_logits, new_past_key_values.to_legacy_cache()
