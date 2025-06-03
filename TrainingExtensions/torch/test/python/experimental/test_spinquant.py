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
"""Test SpinQuant functions"""

import contextlib
import copy
import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaForCausalLM
from unittest.mock import MagicMock
from aimet_torch.experimental.spinquant import spinquant_optimizer
from aimet_torch.experimental.spinquant.spinquant_optimizer import (
    _fuse_rmsnorm_into_linear,
    _identify_and_fuse_rmsnorms_into_linears,
    apply_spinquant,
    RMSNORM_LINEAR_PAIRS,
    R1_LINEAR_FUSION,
)
from aimet_torch.quantsim import QuantizationSimModel


@contextlib.contextmanager
def _register_module_for_spinquant(
    module, rmsnorm_linear_identifier, r1_linear_identifier
):
    try:
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module] = {}
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module][RMSNORM_LINEAR_PAIRS] = (
            rmsnorm_linear_identifier
        )
        spinquant_optimizer.SUPPORTED_MODULE_DICT[module][R1_LINEAR_FUSION] = (
            r1_linear_identifier
        )
        yield

    finally:
        del spinquant_optimizer.SUPPORTED_MODULE_DICT[module]


class RMSNormWithLinear(torch.nn.Module):
    def __init__(self, bias):
        super(RMSNormWithLinear, self).__init__()
        self.rmsnorm = LlamaRMSNorm(10)
        self.q = torch.nn.Linear(10, 10, bias=bias)
        self.k = torch.nn.Linear(10, 10, bias=bias)
        self.v = torch.nn.Linear(10, 10, bias=bias)

    def forward(self, x):
        x = self.rmsnorm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


def mock_llm_module_with_tied_weights():
    mock_module = MagicMock(spec=LlamaForCausalLM)
    mock_module.model = MagicMock()
    mock_module.model.embed_tokens = MagicMock()

    mock_module.lm_head = MagicMock()

    tied_weight = MagicMock()
    mock_module.model.embed_tokens.weight = tied_weight
    mock_module.lm_head.weight = tied_weight

    mock_module.modules = MagicMock(return_value=[mock_module])

    return mock_module


@pytest.mark.parametrize("bias", [True, False])
def test_fuse_rmsnorm_into_linear(bias):
    """Test fuse_rmsnorm_into_linear"""
    torch.manual_seed(0)
    dummy_input = torch.randn(10, 10)
    model = RMSNormWithLinear(bias)
    with torch.no_grad():
        model.rmsnorm.weight.copy_(torch.randn(model.rmsnorm.weight.shape))

    orig_q_out, orig_k_out, orig_v_out = model(dummy_input)
    orig_rmsnorm_weight = model.rmsnorm.weight.clone()
    orig_q_weight = model.q.weight.clone()
    orig_k_weight = model.k.weight.clone()
    orig_v_weight = model.v.weight.clone()
    _fuse_rmsnorm_into_linear(model.rmsnorm, [model.q, model.k, model.v])

    new_q_out, new_k_out, new_v_out = model(dummy_input)
    assert torch.allclose(orig_q_out, new_q_out, atol=1e-6)
    assert torch.allclose(orig_k_out, new_k_out, atol=1e-6)
    assert torch.allclose(orig_v_out, new_v_out, atol=1e-6)

    assert torch.equal(model.rmsnorm.weight, torch.ones(orig_rmsnorm_weight.shape))
    assert not torch.equal(orig_rmsnorm_weight, model.rmsnorm.weight)
    assert not torch.equal(orig_q_weight, model.q.weight)
    assert not torch.equal(orig_k_weight, model.k.weight)
    assert not torch.equal(orig_v_weight, model.v.weight)


def test_identify_and_fuse_llama_model_rmsnorms():
    torch.manual_seed(0)
    with _register_module_for_spinquant(
        RMSNormWithLinear, lambda m: [(m.rmsnorm, [m.q, m.k, m.v])], lambda _: []
    ):
        model = RMSNormWithLinear(bias=True)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LlamaRMSNorm):
                    module.weight.copy_(torch.randn(module.weight.shape))

        dummy_input = torch.randn(10, 10)
        orig_q_out, orig_k_out, orig_v_out = model(dummy_input)

        orig_rmsnorm_weight = model.rmsnorm.weight.clone()

        _identify_and_fuse_rmsnorms_into_linears(model)

        new_q_out, new_k_out, new_v_out = model(dummy_input)
        new_rmsnorm_weight = model.rmsnorm.weight.clone()

        assert torch.allclose(orig_q_out, new_q_out, atol=1e-6)
        assert torch.allclose(orig_k_out, new_k_out, atol=1e-6)
        assert torch.allclose(orig_v_out, new_v_out, atol=1e-6)
        assert not torch.equal(orig_rmsnorm_weight, new_rmsnorm_weight)


def test_fuse_rmsnorm_fp_quantsim_equivalence():
    torch.manual_seed(0)
    with _register_module_for_spinquant(
        RMSNormWithLinear, lambda m: [(m.rmsnorm, [m.q, m.k, m.v])], lambda _: []
    ):
        model = RMSNormWithLinear(bias=True)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LlamaRMSNorm):
                    module.weight.copy_(torch.randn(module.weight.shape))
        model_2 = copy.deepcopy(model)
        dummy_input = torch.randn(10, 10)

        qsim = QuantizationSimModel(model, dummy_input=dummy_input)
        _identify_and_fuse_rmsnorms_into_linears(qsim.model)
        qsim.compute_encodings(lambda m: m(dummy_input))

        qsim_2 = QuantizationSimModel(model_2, dummy_input=dummy_input)
        _identify_and_fuse_rmsnorms_into_linears(qsim_2.model)
        qsim_2.compute_encodings(lambda m: m(dummy_input))

        qsim_out_q, qsim_out_k, qsim_out_v = qsim.model(dummy_input)
        qsim_2_out_q, qsim_2_out_k, qsim_2_out_v = qsim_2.model(dummy_input)
        assert torch.allclose(qsim_out_q, qsim_2_out_q, atol=1e-6)
        assert torch.allclose(qsim_out_k, qsim_2_out_k, atol=1e-6)
        assert torch.allclose(qsim_out_v, qsim_2_out_v, atol=1e-6)


def test_raise_error_on_tied_word_embeddings():
    model = mock_llm_module_with_tied_weights()
    with pytest.raises(RuntimeError):
        apply_spinquant(model)
