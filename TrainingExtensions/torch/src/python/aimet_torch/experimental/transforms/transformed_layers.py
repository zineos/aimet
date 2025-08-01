# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import functools
from contextlib import contextmanager

import torch
from torch import nn

from .transform_ops import TransformOp
from aimet_torch.v2.nn.true_quant import QuantizationMixin


# pylint: disable=abstract-method
class TransformationMixin(torch.nn.Module):
    cls_to_tcls: dict = {}
    tcls_to_cls: dict = {}

    def __init__(self, module: torch.nn.Module):  # pylint: disable=protected-access
        # pylint: disable=super-init-not-called
        super().__new__(self.__class__)  # pylint: disable=no-value-for-parameter
        self.__dict__.update(module.__dict__)
        self._modules = module._modules.copy()
        self._parameters = module._parameters.copy()
        self._buffers = module._buffers.copy()

        self.right_hand_transforms: nn.ModuleList = nn.ModuleList()
        self.left_hand_transforms: nn.ModuleList = nn.ModuleList()

    def add_right_hand_transform(self, transform: TransformOp):
        # we can't interleave mergeable and non-mergeable transforms
        # all non-mergeable RHS transforms should be applied AFTER all mergeable RHS transforms
        if (
            transform.mergeable
            and len(self.right_hand_transforms) > 0
            and not self.right_hand_transforms[-1].mergeable
        ):
            # only an issue if a mergeable transform follows a non-mergeable transform
            raise RuntimeError(
                "Cannot add mergeable transform to RHS after non-mergeable transform."
            )

        self.right_hand_transforms.append(transform)

    def add_left_hand_transform(self, transform: TransformOp):
        # we can't interleave mergeable and non-mergeable transforms
        # all non-mergeable LHS transforms should be applied BEFORE all mergeable LHS transforms
        if (
            transform.mergeable
            and len(self.left_hand_transforms) > 0
            and not self.left_hand_transforms[0].mergeable
        ):
            # only an issue if a mergeable transform is before a non-mergeable transform
            raise RuntimeError(
                "Cannot add mergeable transform to LHS before non-mergeable transform."
            )
        self.left_hand_transforms.insert(0, transform)

    def _compute_merged_params(self):
        mergeable_left_hand_transforms = [
            transform for transform in self.left_hand_transforms if transform.mergeable
        ]
        mergeable_right_hand_transforms = [
            transform for transform in self.right_hand_transforms if transform.mergeable
        ]

        transformed_weight = functools.reduce(
            lambda weight, transform: transform.right_hand_merge(weight),
            mergeable_right_hand_transforms,
            functools.reduce(
                lambda weight, transform: transform.left_hand_merge(weight),
                mergeable_left_hand_transforms[::-1],
                self.weight,
            ),
        )

        if getattr(self, "bias", None) is not None:
            transformed_bias = functools.reduce(
                lambda bias, transform: transform.right_hand_merge(bias),
                mergeable_right_hand_transforms,
                functools.reduce(
                    lambda bias, transform: transform.left_hand_merge(bias),
                    mergeable_left_hand_transforms[::-1],
                    self.bias,
                ),
            )
        else:
            transformed_bias = None

        return transformed_weight, transformed_bias

    def merge(self):
        transformed_weight, transformed_bias = self._compute_merged_params()

        setattr(self, "weight", nn.Parameter(transformed_weight))
        if transformed_bias is not None:
            setattr(self, "bias", nn.Parameter(transformed_bias))

        self.right_hand_transforms = torch.nn.ModuleList(
            [
                transform
                for transform in self.right_hand_transforms
                if not transform.mergeable
            ]
        )
        self.left_hand_transforms = torch.nn.ModuleList(
            [
                transform
                for transform in self.left_hand_transforms
                if not transform.mergeable
            ]
        )

    @contextmanager
    def _patch_transformed_parameters(self):
        transformed_weight, transformed_bias = self._compute_merged_params()

        orig_weight = getattr(self, "weight", None)
        orig_bias = getattr(self, "bias", None)

        setattr(self, "weight", nn.Parameter(transformed_weight))
        if transformed_bias is not None:
            setattr(self, "bias", nn.Parameter(transformed_bias))

        yield

        setattr(self, "weight", orig_weight)
        if transformed_bias is not None:
            setattr(self, "bias", orig_bias)

    def get_original_module(self):
        if len(self.right_hand_transforms) > 0 or len(self.left_hand_transforms) > 0:
            raise RuntimeError(
                "Cannot obtain original module with unmerged transforms."
            )

        original_cls = self.tcls_to_cls[type(self)]
        if isinstance(self, QuantizationMixin):
            original_cls = QuantizationMixin.cls_to_qcls[original_cls]

        orig_module = super().__new__(original_cls)  # pylint: disable=no-value-for-parameter
        orig_module.__dict__ = self.__dict__.copy()
        orig_module.__dict__.pop("forward", None)

        orig_module._parameters = self._parameters.copy()  # pylint: disable=protected-access
        orig_module._buffers = self._buffers.copy()  # pylint: disable=protected-access
        orig_module._modules = self._modules.copy()  # pylint: disable=protected-access

        return orig_module

    @classmethod
    def implements(cls, module_cls):
        def wrapper(quantized_cls):
            cls.cls_to_tcls[module_cls] = quantized_cls
            cls.tcls_to_cls[quantized_cls] = module_cls
            return quantized_cls

        return wrapper

    @classmethod
    def from_module(cls, module: nn.Module):
        if isinstance(module, TransformationMixin):
            # the layer has already been converted to its Transformed equivalent
            return module

        module_cls = type(module)
        if isinstance(module, QuantizationMixin):
            module_cls = QuantizationMixin.qcls_to_cls[module_cls]
        transformed_cls = cls.cls_to_tcls[module_cls]
        if isinstance(module, QuantizationMixin):
            transformed_cls = QuantizationMixin.cls_to_qcls[transformed_cls]

        return transformed_cls(module)


@TransformationMixin.implements(nn.Linear)
class TransformedLinear(TransformationMixin, nn.Linear):
    # pylint: disable=redefined-builtin
    # pylint: disable=arguments-differ
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = functools.reduce(
            lambda x, transform: transform(x), self.left_hand_transforms, input
        )
        output = super().forward(transformed_input)
        transformed_output = functools.reduce(
            lambda x, transform: transform(x), self.right_hand_transforms, output
        )
        return transformed_output


@QuantizationMixin.implements(TransformedLinear)
class QuantizedTransformedLinear(QuantizationMixin, TransformedLinear):
    # pylint: disable=redefined-builtin
    # pylint: disable=arguments-differ
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1) apply non mergeable transforms (they will have their own quantizers present)
        non_mergeable_left_hand_transforms = [
            transform
            for transform in self.left_hand_transforms
            if not transform.mergeable
        ]
        transformed_input = functools.reduce(
            lambda x, transform: transform.forward(x),
            non_mergeable_left_hand_transforms,
            input,
        )

        # 2) quantize inputs
        if self.input_quantizers[0]:
            transformed_input = self.input_quantizers[0](transformed_input)

        # 3) Forward
        with self._patch_transformed_parameters():
            with self._patch_quantized_parameters():
                transformed_output = nn.Linear.forward(self, transformed_input)

        # 4) Apply mergeable right hand transforms
        non_mergeable_right_hand_transforms = [
            transform
            for transform in self.right_hand_transforms
            if not transform.mergeable
        ]

        # 5) quantize outputs
        if self.output_quantizers[0]:
            transformed_output = self.output_quantizers[0](transformed_output)

        # 6) Apply non-mergeable left hand transforms
        transformed_output = functools.reduce(
            lambda x, transform: transform.forward(x),
            non_mergeable_right_hand_transforms,
            transformed_output,
        )
        return transformed_output
