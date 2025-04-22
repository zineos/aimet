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

"""Blockwise sampling utilty"""
import itertools
import sys
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Union, Tuple, Generator, Callable

import torch
from torch.utils.data import DataLoader

from aimet_torch.v2.utils import default_forward_fn, patch_attr
from aimet_torch.utils import change_tensor_device_placement
from aimet_torch import QuantizationSimModel, utils

logger = utils.AimetLogger.get_area_logger(utils.AimetLogger.LogAreas.Utils)

def change_tensor_and_cache_device_placement(inputs, device, cache_movement_fn=None):
    """ This function moves all tensors and huggingface Cache objects to the provided device"""

    # Move all tensors to the provided device
    moved_inputs = change_tensor_device_placement(inputs, device)
    # At this point everything except the Cache objects should be placed on the correct device

    # Helper function to find Cache objects in the provided inputs
    def find_cache(inputs):
        if 'transformers' not in sys.modules:
            # If the transformers module has not been imported, then there cannot be any Cache objects present
            # Since the base class is provided in transformers
            return None
        if isinstance(inputs, sys.modules['transformers'].cache_utils.Cache):
            return inputs
        if isinstance(inputs, (tuple, list, dict)):
            recursive_results = (find_cache(inp) for inp in inputs.values()) if isinstance(inputs, dict) else (find_cache(inp) for inp in inputs)
            try:
                return next(item for item in recursive_results if item is not None)
            except StopIteration:
                return None
        return None

    cache_obj = find_cache(moved_inputs)
    if cache_obj:
        if cache_movement_fn:
            cache_movement_fn(cache_obj)
        else:
            try:
                # Generally, this strategy should work in most cases. However, there is no guarantee from the base
                # Cache class on this. So, if this fails we will ask users to provide a custom function for moving
                # the contents of their Cache object between devices
                cache_obj.key_cache = change_tensor_device_placement(cache_obj.key_cache, device)
                cache_obj.value_cache = change_tensor_device_placement(cache_obj.value_cache, device)
            except Exception as e:
                logger.error("Please provide a cache_movement_fn to move contents of the Cache object used by the model"
                               " between devices. Or, please modify your model to use a DynamicCache object.")
                raise e

    return moved_inputs


class BlockwiseSampler:
    """
    Class providing blockwise sampling utilities. Specifically, BlockWise sampler allows users to specify a list of
    sequential blocks in the module, and the sampler will yield each block, along with the FP and QT inputs to that
    block as a part of the sample() loop.
    BlockwiseSampler caches each block input and resumes computation from that point when the user returns control to
    the sampling loop. This way, excess computational costs for large models is avoided, and the loop is able to make
    use of any user adjustments to quantization parameters for a particular block made in the sampling loop.
    NOTE: users CAN NOT modify model weights during the sample() loop
    """
    def __init__(self,
                 sim: QuantizationSimModel,
                 blocks: List[torch.nn.Module],
                 dataloader: DataLoader,
                 forward_fn: Callable = default_forward_fn
                 ):
        self.sim = sim
        self.blocks = blocks
        self.dataloader = dataloader
        self.forward_fn = forward_fn

    @torch.no_grad()
    def run_inference(self, sample) -> Generator[torch.Tensor, None, None]:
        """
        Helper function to run inference on the model using the given sample, pausing and yielding the results after
        each block.
        """

        @dataclass
        class InputHolder:
            """Dataclass to hold input args and kwargs to a pytorch module."""
            args: tuple
            kwargs: dict

            def to(self, device):
                """ Moves the contents of InputHolder to the provided device """
                self.args = change_tensor_and_cache_device_placement(self.args, device)
                self.kwargs = change_tensor_and_cache_device_placement(self.kwargs, device)

            def __deepcopy__(self, memo):
                with patch_attr(torch.Tensor, '__deepcopy__', lambda self, memo: self.detach().clone()):
                    return InputHolder(deepcopy(self.args), deepcopy(self.kwargs))

        class StopForwardExceptionWithInput(utils.StopForwardException):
            """Exception raised in order to stop forward execution through the model. Holds module input data."""
            def __init__(self, captured_input):
                self.captured_input = captured_input

        def hook_fn(_, args, kwargs):
            raise StopForwardExceptionWithInput(deepcopy(InputHolder(args, kwargs)))

        try:
            hook = self.blocks[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.forward_fn(self.sim.model, sample)
        except StopForwardExceptionWithInput as e:
            # pylint: disable=used-before-assignment
            hook.remove()
            next_block_input = e.captured_input
            next_block_input.to("cpu")
            e.captured_input = None

            yield next_block_input.args, next_block_input.kwargs

        for block in self.blocks[:-1]:
            next_block_input.to(utils.get_device(block))
            next_block_input.args = block(*next_block_input.args, **next_block_input.kwargs)
            next_block_input.to("cpu")

            if not isinstance(next_block_input.args, tuple):
                next_block_input.args = (next_block_input.args,)

            yield next_block_input.args, next_block_input.kwargs


    def sample(self, keep_unused_blocks_on_cpu:bool=True, device=None, desc:str="Blocks processed") -> Generator[Tuple[Union[torch.nn.Module, torch.nn.ModuleList], torch.Tensor, torch.Tensor], None, None]:
        """
        Main generator function for blockwise sampler. Each loop of this generator yields a tuple of
        (block, [list of FP inputs to block], [list of QT inputs to block]) based on the list of blocks provided during
        initialization.
        """
        device = device if device else utils.get_device(self.sim.model)

        fp_inferences = []
        qt_inferences = []

        for sample in itertools.islice(self.dataloader, len(self.dataloader)):
            fp_inferences.append(self.run_inference(sample))
            qt_inferences.append(self.run_inference(sample))

        if keep_unused_blocks_on_cpu:
            self.sim.model.to("cpu")

        blocks = iter(self.blocks)
        prev_block = self.blocks[0]
        with tqdm(total=len(self.blocks), desc=desc) as pbar:
            while True:
                try:
                    block = next(blocks)

                    if keep_unused_blocks_on_cpu:
                        prev_block.to(device)

                    # Quantizers must be ENABLED when calculating quantized block inputs
                    qt_block_inputs = [next(block_input) for block_input in qt_inferences]

                    # Quantizers must be DISABLED when calculating FP block inputs
                    with utils.disable_all_quantizers(self.sim.model):
                        fp_block_inputs = [next(block_input) for block_input in fp_inferences]

                    if keep_unused_blocks_on_cpu:
                        prev_block.to("cpu")
                        block.to(device)

                    yield block, fp_block_inputs, qt_block_inputs

                    pbar.update(1)
                    prev_block = block
                except StopIteration:
                    break

            if keep_unused_blocks_on_cpu:
                block.to("cpu")
