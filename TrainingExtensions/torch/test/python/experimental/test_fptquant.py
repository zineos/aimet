# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Test aimet-torch fptquant"""

import pytest
import torch
from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin
from aimet_torch.experimental.fptquant.fptquant_transforms import (
    GroupedHadamardTransformOp,
)
from aimet_torch.quantsim import QuantizationSimModel


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("size", [24, 64])
def test_grouped_hadamard_transform(size):
    linear = torch.nn.Linear(size, size, bias=False)
    transformed_linear = TransformationMixin.from_module(linear)
    transform = GroupedHadamardTransformOp(size)

    # Adding mergeable transform first since transforms are added as a stack
    transformed_linear.add_left_hand_transform(transform.get_inverted_op())
    transformed_linear.add_left_hand_transform(transform)

    assert len(transformed_linear.left_hand_transforms) == 2

    dummy_input = torch.randn(size, size)
    transformed_out = transformed_linear(dummy_input)
    orig_out = linear(dummy_input)
    assert torch.allclose(transformed_out, orig_out, atol=1e-6)

    transformed_linear.merge()
    assert len(transformed_linear.left_hand_transforms) == 1
    new_out = transformed_linear(dummy_input)
    assert torch.allclose(transformed_out, new_out, atol=1e-6)


@pytest.mark.skip("Enable this when quantized GroupedHadamardOp is supported")
@pytest.mark.parametrize("size", [24, 64])
def test_quantized_grouped_hadamard_transform(size):
    model = LinearModel(size)
    model.linear = TransformationMixin.from_module(model.linear)
    transform = GroupedHadamardTransformOp(size)

    # Adding mergeable transform first since transforms are added as a stack
    model.linear.add_left_hand_transform(transform.get_inverted_op())
    model.linear.add_left_hand_transform(transform)
    import pdb

    pdb.set_trace()

    dummy_input = torch.randn(size, size)
    qsim = QuantizationSimModel(model, dummy_input)
    qsim.compute_encodings(lambda m: m(dummy_input))

    before_merge_out = qsim.model(dummy_input)

    assert (
        qsim.model.linear.left_hand_transforms[0].output_quantizers[0].get_min()
        is not None
    )
    assert qsim.model.linear.left_hand_transforms[1].output_quantizers[0] is None

    qsim.model.linear.merge()

    assert (
        qsim.model.linear.left_hand_transforms[0].output_quantizers[0].get_min()
        is not None
    )
    assert len(qsim.model.linear.left_hand_transforms) == 1

    after_merge_out = qsim.model(dummy_input)
    assert torch.allclose(before_merge_out, after_merge_out, atol=1e-6)
