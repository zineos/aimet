# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import torch

from aimet_torch.experimental.transforms.transform_ops import TransformOp


class LocalTransformOptimizer:
    p: float = 4.0
    num_iterations: int = 200
    lr: float = 1e-2

    @staticmethod
    def compute_loss(weight: torch.Tensor) -> torch.Tensor:
        return torch.mean((weight.abs() ** LocalTransformOptimizer.p)) ** (
            1 / LocalTransformOptimizer.p
        )

    def __init__(self, transforms: list[TransformOp]):
        self.parameters = []
        for transform in transforms:
            self.parameters.extend(list(transform.parameters()))
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr)

    def optimize(self):
        for parameter in self.parameters:
            parameter.requires_grad_(True)

        for _ in range(self.num_iterations):
            self.optimizer.zero_grad()
            loss = torch.stack(
                tuple(self.compute_loss(param) for param in self.parameters)
            ).sum(dim=0)
            loss.backward()
            self.optimizer.step()

        for parameter in self.parameters:
            parameter.requires_grad_(False)
