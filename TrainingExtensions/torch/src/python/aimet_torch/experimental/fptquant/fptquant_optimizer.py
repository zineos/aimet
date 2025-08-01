# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

import torch
from typing import Type
from types import NoneType
from tqdm import tqdm

from transformers import PretrainedConfig

from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin

from .fptquant_config import fptquant_model_config_dict, FPTQuantConfig, BlockInterface
from .fptquant_transforms import ScaledRotateTransformOp, ScalingTransformOp
from .fptquant_local_transform_optimizer import LocalTransformOptimizer


class FPTQuant:
    @staticmethod
    def insert_fpt_quant_transforms(model: torch.nn.Module, config: PretrainedConfig):
        for block_interface in tqdm(
            FPTQuant._get_blocks(model), desc="Block transforms inserted"
        ):
            block_interface.q_proj = TransformationMixin.from_module(
                block_interface.q_proj
            )
            block_interface.k_proj = TransformationMixin.from_module(
                block_interface.k_proj
            )
            block_interface.v_proj = TransformationMixin.from_module(
                block_interface.v_proj
            )
            block_interface.o_proj = TransformationMixin.from_module(
                block_interface.o_proj
            )
            block_interface.up_proj = TransformationMixin.from_module(
                block_interface.up_proj
            )
            block_interface.down_proj = TransformationMixin.from_module(
                block_interface.down_proj
            )
            block_interface.gate_proj = TransformationMixin.from_module(
                block_interface.gate_proj
            )

            FPTQuant._apply_prerope_transform(block_interface, config)
            FPTQuant._apply_value_transform(block_interface)
            FPTQuant._apply_up_down_scaling_transform(block_interface, config)
            FPTQuant._apply_residual_transform(block_interface)
            FPTQuant._apply_down_projection_transform(block_interface)

    @staticmethod
    def merge_fpt_quant_transforms(model: torch.nn.Module):
        for block_interface in FPTQuant._get_blocks(model):
            block_interface.q_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.q_proj
                )
            )
            block_interface.k_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.k_proj
                )
            )
            block_interface.v_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.v_proj
                )
            )
            block_interface.o_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.o_proj
                )
            )
            block_interface.up_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.up_proj
                )
            )
            block_interface.down_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.down_proj
                )
            )
            block_interface.gate_proj = (
                FPTQuant._merge_transforms_and_recover_original_layer(
                    block_interface.gate_proj
                )
            )

    @staticmethod
    def _merge_transforms_and_recover_original_layer(layer: torch.nn.Module):
        if not isinstance(layer, TransformationMixin):
            return layer  # Do nothing if it is not a transformed layer

        layer.merge()
        if len(layer.right_hand_transforms) == len(layer.left_hand_transforms) == 0:
            return layer.get_original_module()
        return layer

    @staticmethod
    def _screen_for_target_type(model: torch.nn.Module) -> Type:
        for module in model.modules():
            for target in fptquant_model_config_dict:
                if isinstance(module, target):
                    return target
        # No targets found in provided model
        return NoneType

    @staticmethod
    def _get_blocks(model: torch.nn.Module) -> list[BlockInterface]:
        target_type = FPTQuant._screen_for_target_type(model)
        config = fptquant_model_config_dict.get(target_type, FPTQuantConfig())
        target_modules = []
        if config.block_type is not None:
            target_modules = [
                config.block_interface(m)
                for m in model.modules()
                if isinstance(m, config.block_type)
            ]
        return target_modules

    @staticmethod
    def _apply_prerope_transform(block: BlockInterface, config: PretrainedConfig):
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        transform = ScaledRotateTransformOp(
            head_dim, num_attention_heads, num_key_value_heads
        )
        transform_optimizer = LocalTransformOptimizer([transform])
        transform_optimizer.optimize()

        block.q_proj.add_right_hand_transform(transform.get_inverted_op())
        block.k_proj.add_right_hand_transform(transform)

    @staticmethod
    def _apply_value_transform(block: BlockInterface):
        pass

    @staticmethod
    def _apply_up_down_scaling_transform(
        block: BlockInterface, config: PretrainedConfig
    ):
        transform = ScalingTransformOp(config.intermediate_size)
        transform_optimizer = LocalTransformOptimizer([transform])
        transform_optimizer.optimize()

        block.up_proj.add_right_hand_transform(transform)
        block.down_proj.add_left_hand_transform(transform.get_inverted_op())

    @staticmethod
    def _apply_residual_transform(block: BlockInterface):
        pass

    @staticmethod
    def _apply_down_projection_transform(block: BlockInterface):
        pass
