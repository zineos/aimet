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
""" Base class for GenAI models """

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedTokenizer
from aimet_torch.utils import change_tensor_device_placement
from aimet_torch import QuantizationSimModel

from .utils.llm_utils import (
    slice_inputs_for_inference,
    trim_pad_logits,
    pad_inputs,
    get_dummy_kv,
    pad_past_kv,
    pad_input_attn_mask,
    create_kv_attn_mask,
    create_1d_attn_mask,
    get_position_ids_from_attention_mask,
    update_kv_cache
)

class GenAIModel(ABC):
    """ Generic GenAI model """
    @staticmethod
    @abstractmethod
    def instantiate_quantsim(context_length: int, sequence_length: int) -> QuantizationSimModel:
        """ Instantiate QuantSim of model """

    @staticmethod
    @abstractmethod
    def instantiate_tokenizer() -> PreTrainedTokenizer:
        """ Instantiate model tokenizer """

    @staticmethod
    def get_quantsim_config() -> str:
        """ Get default QuantSim config """
        config_path = Path(__file__).parent / "config/default_config.json"
        return str(config_path.resolve())

    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-locals
    @staticmethod
    def static_graph_prepare_inputs(
            model: torch.nn.Module,
            input_ids_slice: torch.Tensor,
            attention_mask_slice: torch.Tensor,
            previous_outputs: dict[str, torch.Tensor],
            context_length: int,
            sequence_length: int,
    ):
        """ Prepare provided inputs for model forward pass with static graph constraints """
        batch_size = input_ids_slice.shape[0]
        pad_token = 0  # TODO: switch back to tokenize.eos_token_id
        head_dim = model.config.head_dim if hasattr(model.config,
                                                    'head_dim') else model.config.hidden_size // model.config.num_attention_heads

        # input id preparation
        pad_input_ids = pad_inputs(pad_token=pad_token,
                                   max_input_tokens=sequence_length,
                                   input_ids_slice=input_ids_slice)

        # KV input preparation
        dummy_kv = get_dummy_kv(batch_size=batch_size,
                                model_context_len=context_length,
                                max_input_tokens=sequence_length,
                                num_key_value_heads=model.config.num_key_value_heads,
                                head_dim=head_dim,
                                device=model.device)

        padded_past_kv_in = pad_past_kv(dummy_past_kv=dummy_kv,
                                        past_kv=previous_outputs['past_key_values'],
                                        num_hidden_layers=model.config.num_hidden_layers)

        # attention mask input preparation
        inp_attn_mask = pad_input_attn_mask(attn_mask_slice=attention_mask_slice,
                                            max_input_tokens=sequence_length, )
        past_kv_attn_mask = create_kv_attn_mask(unpadded_past_kv=previous_outputs['past_key_values'],
                                                model_context_len=context_length,
                                                max_input_tokens=sequence_length,
                                                batch_size=batch_size,
                                                device=model.device)
        prepared_1d_attention_mask = create_1d_attn_mask(attn_mask_past_kv=past_kv_attn_mask,
                                                         attn_mask_input=inp_attn_mask)

        # position ID preparation
        position_ids = get_position_ids_from_attention_mask(attention_mask=prepared_1d_attention_mask,
                                                            max_input_tokens=sequence_length,
                                                            model_context_len=context_length, )

        prepared_inputs = {
            'input_ids': pad_input_ids,
            'attention_mask': prepared_1d_attention_mask,
            'position_ids': position_ids,
            'past_key_values': padded_past_kv_in,
        }
        return prepared_inputs


    @classmethod
    def create_static_graph_forward(cls, context_length: int, sequence_length: int):
        """ Higher order function to produce a new model forward function based on context length and sequence length """

        # pylint: disable=too-many-positional-arguments
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        # pylint: disable=unused-argument
        def static_graph_forward(
                model: torch.nn.Module,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_values: tuple[tuple[torch.Tensor]] = None,
                inputs_embeds: torch.Tensor = None,
                labels: torch.Tensor = None,
                use_cache: bool = None,
                output_attentions: torch.Tensor = None,
                output_hidden_states: torch.Tensor = None,
                return_dict: bool = False,
                cache_position: torch.Tensor = None):
            """ Patched model forward function to mock static graph constraints """

            # create the generator which slices input into chunks of AR (and pads if necessary)
            slice_inputs_gen_obj = slice_inputs_for_inference(sequence_length=sequence_length,
                                                              input_ids=input_ids,
                                                              inputs_embeds=inputs_embeds,
                                                              attention_mask=attention_mask,
                                                              position_ids=position_ids)

            # dictionary to store the running output which contains the logits and the useful past kv cache until that execution
            outputs = {}
            outputs['past_key_values'] = past_key_values
            for inputs in slice_inputs_gen_obj:
                input_ids_slice = inputs['input_ids_slice']
                attn_mask_slice = inputs['attn_mask_slice']

                prepared_inputs = cls.static_graph_prepare_inputs(model,
                                                                  input_ids_slice=input_ids_slice,
                                                                  attention_mask_slice=attn_mask_slice,
                                                                  previous_outputs=outputs,
                                                                  context_length=context_length,
                                                                  sequence_length=sequence_length)

                cur_outputs = model.orig_forward(**prepared_inputs)

                # Explicitly move logits to CPU
                cur_outputs = (change_tensor_device_placement(cur_outputs[0], "cpu"), cur_outputs[1])

                # avoided creating a new tuple of current_key_value to avoid the memory spike, sending slice
                outputs['past_key_values'] = update_kv_cache(unpadded_past_kv=outputs['past_key_values'],
                                                             current_key_values=cur_outputs[-1],
                                                             input_ids_slice=input_ids_slice)

                lm_logits = trim_pad_logits(cur_logits=cur_outputs[0],
                                            input_ids_slice=input_ids_slice)

                bsz, _, dim = lm_logits.shape

                outputs['logits'] = torch.cat(
                    (outputs.get('logits', torch.zeros((bsz, 0, dim), device=lm_logits.device)), lm_logits),
                    dim=1)

            if return_dict:
                return CausalLMOutputWithPast(
                    loss=outputs.get('loss', None),
                    logits=outputs.get('logits', None),
                    past_key_values=outputs.get('past_key_values', None),
                    hidden_states=None,
                    attentions=None,
                )
            return (outputs['logits'], outputs['past_key_values'])

        return static_graph_forward

    # pylint: disable=unused-argument
    # pylint: disable=bad-staticmethod-argument
    @staticmethod
    def static_graph_prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: tuple[tuple[torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        Patched prepare_inputs_for_generation function to enable Huggingface generate() on model with static
        graph constraints
        """

        # "past_key_values is None" indicates that `prepare_inputs_for_generation()` is called first time in `generate()`
        # when model_mode is kvcache. In first inference, we should pass all inputs to get valid kv cache.
        # this is called before `generate()`, so if we get the token ids from the output of generation,
        # then in the next call, we can update the embeddings before returning to `generate()`
        kv_length = past_key_values.get_seq_length() if not isinstance(past_key_values, tuple) else \
            past_key_values[0][1].shape[-2]
        return {
            "input_ids": input_ids[:, kv_length:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask
        }
