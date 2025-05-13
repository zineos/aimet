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
"""Utilities for simulating static graph constraints on HF models. Borrowed from GenAI lib"""

import torch
from typing import Union


def _shift(
    a: Union[torch.Tensor, tuple[torch.Tensor]],
    dim: int,
    shift_size: int,
    shift_to_left: bool = True,
):
    if isinstance(a, tuple):
        return tuple(_shift(ai, dim, shift_size) for ai in a)

    if dim not in (2, 3):
        raise ValueError("Unexpected shift axis")

    if shift_to_left:
        return a[:, :, shift_size:, :] if dim == 2 else a[:, :, :, shift_size:]

    # this fails when the shift size is 0, in that case, it returns empty tensor, which is opposite of what we want
    orig_len = a.shape[-2] if dim == 2 else a.shape[-1]
    keep_size = orig_len - shift_size
    return a[:, :, :keep_size, :] if dim == 2 else a[:, :, :, :keep_size]


def _concat(
    a: Union[torch.Tensor, tuple[torch.Tensor]],
    b: Union[torch.Tensor, tuple[torch.Tensor]],
    dim: int,
):
    if isinstance(a, tuple):
        return tuple(_concat(ai, bi, dim) for ai, bi in zip(a, b))
    if a is None:
        return b
    if b is None:
        return a
    return torch.cat((a, b), dim=dim)


# pylint: disable=too-many-positional-arguments
def slice_inputs_for_inference(
    sequence_length: int,
    input_ids: torch.Tensor = None,
    inputs_embeds: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    past_seen_tokens: tuple[tuple[torch.Tensor, torch.Tensor]] = None,
    hidden_states: torch.Tensor = None,
):
    """
    This function is responsible for slicing the inputs based on the AR and yielding them to the user as a generator.

    :param sequence_length: length of the sequence
    :param input_ids: (optional) input IDs. Callers must submit either input_ids, or input_embeds but not both
    :param inputs_embeds: (optional) input embeddings. Callers must submit either input_ids, or input_embeds but not both.
    :param attention_mask: (optional) attention mask
    :param position_ids: (optional) position ids
    :param past_seen_tokens: (optional) hidden states
    :param hidden_states: (optional) hidden states

    Note: To be able to ingest all the model_context_len tokens, we need to slice using the left padding and chunk the
    input into chunks of max_input_tokens
    """
    input_count = 0
    for inp in (input_ids, inputs_embeds):
        if inp is not None:
            input_count = input_count + 1

    if input_count != 1:
        raise RuntimeError(
            "Inference only works for input ids or input embeddings, not both"
        )

    if input_ids is not None:
        input_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        device = input_ids.device
    else:
        input_length = inputs_embeds.shape[1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, input_length), dtype=torch.long, device=device
        )

    if position_ids is None:
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        if past_seen_tokens is not None:
            position_ids += past_seen_tokens

    for idx in range(0, input_length, sequence_length)[::-1]:
        idx = input_length - idx
        input_slice = {
            "attn_mask_slice": attention_mask[:, max(0, idx - sequence_length) : idx],
            "position_ids_slice": position_ids[:, max(0, idx - sequence_length) : idx],
        }

        if input_ids is not None:
            input_slice["input_ids_slice"] = input_ids[
                :, max(0, idx - sequence_length) : idx
            ]
        else:
            input_slice["inputs_embeds_slice"] = inputs_embeds[
                :, max(0, idx - sequence_length) : idx, :
            ]

        if hidden_states is not None:
            input_slice["hidden_states_slice"] = hidden_states[
                :, max(0, idx - sequence_length) : idx
            ]

        yield input_slice


# pylint: disable=too-many-positional-arguments
def pad_inputs(
    max_input_tokens: int,
    input_ids_slice: torch.Tensor = None,
    inputs_embeds_slice: torch.Tensor = None,
    pad_token: int = 0,
    pad_embeds: torch.Tensor = None,
    pad_to_left: bool = True,
) -> torch.Tensor:
    """
    This function pads the input_ids/ inputs_embeds since slice may return input_ids or inputs_embeds that is shorter
    than what the model accepts (AR len)

    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference
    :param input_ids_slice: (optional) the current input ids slice that is passed into the model in the current invocation
    :param inputs_embeds_slice: (optional) the current input embeds slice that is passed into the model in the current invocation
    :param pad_token: (optional) padding token. Defaults to 0 to avoid impacting the range of values in the activation tensor
    :param pad_embeds: (optional) Tensor with which we pad the inputs_embeds_slice. This is optional and will be used if provided.
        If this is not provided, and we are working with input embeddings, the inputs_embeds_slice tensor will be padded
        with zero and not the pad_token. The reason for this is that the pad_token could be a large non-zero value which
        will impact the range of values in the padded tensor.
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    input_ids_or_embs = (
        input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    )
    device = input_ids_or_embs.device
    input_length = input_ids_or_embs.shape[1]
    batch_size = input_ids_or_embs.shape[0]
    shape = (batch_size, max_input_tokens - input_length)

    if pad_embeds is None:
        if inputs_embeds_slice is not None:
            shape += (input_ids_or_embs.shape[-1],)
            pad_token = 0

        input_extensions = torch.full(
            shape, fill_value=pad_token, dtype=input_ids_or_embs.dtype, device=device
        )
    else:
        assert input_ids_or_embs.shape[-1] == pad_embeds.shape[-1]
        # we only want to extract the embeddings dimension from the passed pad_embeddings
        pad_embeds = pad_embeds[-1]
        input_extensions = (
            pad_embeds.view(1, 1, -1)
            .repeat(batch_size, max_input_tokens - input_length, 1)
            .to(dtype=input_ids_or_embs.dtype, device=device)
        )

    # left padding
    if pad_to_left:
        input_ids_or_embs = torch.cat((input_extensions, input_ids_or_embs), dim=1)
    # right padding
    else:
        input_ids_or_embs = torch.cat((input_ids_or_embs, input_extensions), dim=1)

    return input_ids_or_embs


def pad_hidden_states(
    max_input_tokens: int,
    hidden_states_slice: torch.Tensor,
    pad_token: int = 0,
    pad_to_left: bool = True,
):
    """
    This function pads the hidden states since slice may return hidden states that is shorter than model sequence length

    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference
    :param hidden_states_slice: the current hidden state slice that is passed into the model in the current invocation
    :param pad_token: (optional) padding token. Defaults to 0 to avoid impacting the range of values in the activation tensor
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    pad_shape = list(hidden_states_slice.shape)
    pad_shape[1] = max_input_tokens - hidden_states_slice.shape[1]
    pad = torch.full(
        pad_shape,
        fill_value=pad_token,
        dtype=hidden_states_slice.dtype,
        device=hidden_states_slice.device,
    )

    # left padding
    if pad_to_left:
        padded_hidden_states_slice = torch.cat((pad, hidden_states_slice), dim=1)
    # right padding
    else:
        padded_hidden_states_slice = torch.cat((hidden_states_slice, pad), dim=1)

    return padded_hidden_states_slice


def pad_input_attn_mask(
    attn_mask_slice: torch.Tensor, max_input_tokens: int, pad_to_left: bool = True
):
    """
    This function pads the 1d attention mask to make it of shape (batch_size, max_input_tokens),

    A: padded current input (0s)
    B: current valid input (1s)

    If the pad_to_left argument is set to True, it means we perform left_padding & produce the attention mask as A|B
    else, pad_to_left=False means we do right padding, & produce the attention mask as B|A

    :param attn_mask_slice: attention mask which corresponds to the current slice of inputs
    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (sequence length)
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    batch_size = attn_mask_slice.shape[0]
    input_padding_length = max_input_tokens - attn_mask_slice.shape[1]

    padded_input_attn_mask = torch.zeros(
        (batch_size, input_padding_length),
        dtype=torch.long,
        device=attn_mask_slice.device,
    )

    # left padding
    if pad_to_left:
        attention_mask = torch.cat((padded_input_attn_mask, attn_mask_slice), dim=1)
    # right padding
    else:
        attention_mask = torch.cat((attn_mask_slice, padded_input_attn_mask), dim=1)

    return attention_mask


# pylint: disable=too-many-positional-arguments
def create_kv_attn_mask(
    unpadded_past_kv: tuple[tuple[torch.Tensor, torch.Tensor]],
    model_context_len: int,
    max_input_tokens: int,
    batch_size: int,
    device: str,
    pad_to_left: bool = True,
) -> torch.Tensor:
    """
    This function prepares the 1d attention mask based on the useful past key values seen so far.
    This can be visualized into two sections.
    A | B
    A: padded past kv length (0s)
    B: useful past kv length (1s)

    :param unpadded_past_kv: useful accumulated past kv
    :param model_context_len: model context length
    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (sequence length)
    :param batch_size: batch size
    :param device: torch device to use
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """

    useful_past_kv_length = unpadded_past_kv[0][1].shape[-2] if unpadded_past_kv else 0
    padded_kv_length = (model_context_len - max_input_tokens) - useful_past_kv_length

    useful_past_kv_attn_mask = torch.ones(
        (batch_size, useful_past_kv_length), dtype=torch.long, device=device
    )
    padded_kv_attn_mask = torch.zeros(
        (batch_size, padded_kv_length), dtype=torch.long, device=device
    )

    # left padding
    if pad_to_left:
        attention_mask = torch.cat(
            (padded_kv_attn_mask, useful_past_kv_attn_mask), dim=1
        )
    # right padding
    else:
        attention_mask = torch.cat(
            (useful_past_kv_attn_mask, padded_kv_attn_mask), dim=1
        )
    return attention_mask


def create_1d_attn_mask(
    attn_mask_past_kv: torch.Tensor,
    attn_mask_input: torch.Tensor,
    cache_index: int = None,
):
    """
    This function concatenates the attention mask corresponding to the input ids and the past kv together

    :param attn_mask_past_kv: attention mask corresponding to the past kv
    :param attn_mask_input: attention mask corresponding to the input (max_input tokens that the model takes)
    :param cache_index:cache_index determines where should the attn_mask_input be placed. If None, the input_attention mask
    is placed towards the end (assuming concat in the kv update within attention) else it is placed right after the valid kv mask.
    """
    if cache_index is None:
        attention_mask = torch.cat((attn_mask_past_kv, attn_mask_input), dim=1)
    else:
        attention_mask_post_valid_kv = attn_mask_past_kv[:, cache_index:]
        attention_mask_valid_kv = attn_mask_past_kv[:, :cache_index]
        attention_mask = torch.cat(
            (attention_mask_valid_kv, attn_mask_input, attention_mask_post_valid_kv),
            dim=1,
        )

    return attention_mask


# pylint: disable=too-many-positional-arguments
def pad_past_kv(
    dummy_past_kv: tuple[torch.Tensor],
    past_kv: tuple[tuple[torch.Tensor, torch.Tensor]],
    num_hidden_layers: int,
    key_concat_axis: int = 2,
    value_concat_axis: int = 2,
    pad_to_left: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
    """
    This function is responsible taking in current past kv and pad it using dummy kv to meet the static shape
    requirements for past kv.
    We compute the padding kv length as (Context Length - sequence length) - (valid kv length).
    The shape after we pad past kv is (Context Length - sequence length)

    :param dummy_past_kv: dummy KV cache constructed for a single hidden layer
    :param past_kv: useful accumulated past kv
    :param num_hidden_layers: number of hidden layers
    :param key_concat_axis: axis to concatenate past keys
    :param value_concat_axis: axis to concatenate past values
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    useful_past_kv_length = past_kv[0][1].shape[-2] if past_kv else 0

    # trimmed dummy kv is the final length dummy kv that will be concatenated to the unpadded_past_kv either to the left or to the right.
    trimmed_dummy_kv = (
        _shift(dummy_past_kv[0], key_concat_axis, useful_past_kv_length),
        _shift(dummy_past_kv[1], value_concat_axis, useful_past_kv_length),
    )
    if past_kv:
        if pad_to_left:
            padded_key_values = tuple(
                (
                    _concat(trimmed_dummy_kv[0], past_kv[i][0], key_concat_axis),
                    _concat(trimmed_dummy_kv[1], past_kv[i][1], value_concat_axis),
                )
                for i in range(num_hidden_layers)
            )
        else:
            padded_key_values = tuple(
                (
                    _concat(past_kv[i][0], trimmed_dummy_kv[0], key_concat_axis),
                    _concat(past_kv[i][1], trimmed_dummy_kv[1], value_concat_axis),
                )
                for i in range(num_hidden_layers)
            )
        return padded_key_values
    return tuple(trimmed_dummy_kv for _ in range(num_hidden_layers))


# pylint: disable=too-many-positional-arguments
def get_dummy_kv(
    batch_size: int,
    num_key_value_heads: int,
    head_dim: int,
    device: str,
    cache_len: int = None,
    model_context_len: int = None,
    max_input_tokens: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function determines the shape of the dummy kv using the required arguments which reflect model config
    Returns the dummy kv of fixed size each time (for a single layer). This will be used for padding the passed past kv

    :param batch_size: batch size
    :param num_key_value_heads: number of KV heads
    :param head_dim: dimension at each head
    :param device: torch device to place dummy KV on
    :param cache_len: cache length
    :param model_context_len: model context length
    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (sequence length)
    """

    def _cache(shape):
        return torch.zeros(shape, device=device)

    if cache_len is None:
        cache_len = model_context_len - max_input_tokens

    value = (batch_size, num_key_value_heads, cache_len, head_dim)
    key = tuple(value)
    return (_cache(key), _cache(value))


def _trim_padded_tensor(
    tensor: torch.Tensor, input_length: int, pad_axis: int = 1, pad_to_left: bool = True
) -> torch.Tensor:
    """
    This function is responsible for stripping the non-useful values from the returned tensor (e.g., logits or hidden states)
    since our prepared model returns fixed length tensor

    :param tensor: tensor returned from the model (logits or hidden states)
    :param input_length: length of the valid portion of the tensor
    :param pad_axis: dimension along which padding should be applied
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    # left padding so we remove the logits from the left & return the valid input length from the end
    if pad_to_left:
        trimmed_tensor = torch.narrow(
            tensor, pad_axis, tensor.shape[pad_axis] - input_length, input_length
        )
    # right padding, so we extract the valid input_length from the beginning
    else:
        trimmed_tensor = torch.narrow(tensor, pad_axis, 0, input_length)
    return trimmed_tensor


def trim_pad_logits(
    cur_logits: torch.Tensor,
    input_ids_slice: torch.Tensor = None,
    inputs_embeds_slice: torch.Tensor = None,
    pad_to_left: bool = True,
) -> torch.Tensor:
    """
    This function is responsible for stripping the non-useful logits from the returned logits since our prepared model
    returns fixed length logits

    :param cur_logits: logits returned from the model
    :param input_ids_slice: unpadded input_ids slice. Cannot supply this and input_embeds_slice
    :param inputs_embeds_slice: unpadded input_embeds slice. Cannot supply this and input_ids_slice
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    input_ids_or_embs = (
        input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    )
    input_length = input_ids_or_embs.shape[1]

    return _trim_padded_tensor(
        cur_logits, input_length=input_length, pad_to_left=pad_to_left
    )


def trim_padded_hidden_states(
    hidden_states: torch.Tensor,
    input_ids_slice: torch.Tensor = None,
    inputs_embeds_slice: torch.Tensor = None,
    pad_to_left: bool = True,
) -> torch.Tensor:
    """
    This function is responsible for stripping the non-useful hidden states from the returned hidden states
    since our prepared model returns fixed length hidden states

    :param hidden_states: hidden states returned from the model
    :param input_ids_slice: unpadded input_ids slice. Cannot supply this and input_embeds_slice
    :param inputs_embeds_slice: unpadded input_embeds slice. Cannot supply this and input_ids_slice
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    input_ids_or_embs = (
        input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    )
    input_length = input_ids_or_embs.shape[1]

    return _trim_padded_tensor(
        hidden_states, input_length=input_length, pad_to_left=pad_to_left
    )


def get_position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
    max_input_tokens: int,
    model_context_len: int,
    cache_index: int = None,
) -> torch.Tensor:
    """
    This function computes the position ids for the tokens being fed into the model from the 1d_attn_mask.

    :param attention_mask: prepared attention mask needed to deduce position ids
    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (sequence length)
    :param model_context_len: model context length
    :param cache_index: index for the starting position of KV cache
    """

    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.clip(0, model_context_len - 1)
    if cache_index is None:
        position_ids = position_ids[..., -max_input_tokens:]
    else:
        position_ids = position_ids[..., cache_index : cache_index + max_input_tokens]
    return position_ids


def pad_position_ids(
    position_ids_slice: torch.Tensor,
    max_input_tokens: int,
    pad_value: int = 0,
    pad_to_left: bool = True,
) -> torch.Tensor:
    """
    This function pads the position_ids since slice may return position_ids that is smaller than what the model accepts (AR len)

    :param position_ids_slice: current position_ids slice, to be passed into the model during the current invocation
    :param max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (sequence length)
    :param pad_value: value to pad position_ids with. Defaults to 0.
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    if position_ids_slice.dim() != 2:
        raise ValueError("Position ids slice must be 2D torch Tensor")

    batch_size, pos_ids_len = position_ids_slice.shape

    if pos_ids_len < max_input_tokens:
        pad_pos_ids = torch.full(
            (batch_size, max_input_tokens - pos_ids_len),
            pad_value,
            dtype=position_ids_slice.dtype,
            device=position_ids_slice.device,
        )

        if pad_to_left:
            position_ids = torch.cat((pad_pos_ids, position_ids_slice), dim=-1)
        else:
            position_ids = torch.cat((position_ids_slice, pad_pos_ids), dim=-1)

        return position_ids

    return position_ids_slice


# pylint: disable=too-many-positional-arguments
def update_kv_cache(
    unpadded_past_kv: tuple[tuple[torch.Tensor, torch.Tensor]],
    current_key_values: tuple[tuple[torch.Tensor, torch.Tensor]],
    key_concat_axis: int = 2,
    value_concat_axis: int = 2,
    input_ids_slice: torch.Tensor = None,
    inputs_embeds_slice: torch.Tensor = None,
    pad_to_left: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
    """
    This function concats the KV cache that the model outputs in the current iteration (unpadded_past_kv) with the KV$
    that the model has accumulated so far(unpadded_past_kv)
    1. remove the non-useful padding kv from the current_key_values depending on whether it was padded to left or to right
    2. concatenate the stripped current kv with past useful kv if it exists

    :param unpadded_past_kv: unpadded useful kv that is accumulated from the previous model invocations
    :param current_key_values: current padded kv returned from the model
    :param key_concat_axis: axis to concatenate keys to
    :param value_concat_axis: axis to concatenate values to
    :param input_ids_slice: (optional) the slice of inputs returned from the iterator (before any padding is applied)
    :param inputs_embeds_slice: (optional) the slice of inputs returned from the iterator (before any padding is applied)
    :param pad_to_left: (optional) boolean value indicating whether padding is done towards the left or right.
    """
    input_ids_or_embs = (
        input_ids_slice if input_ids_slice is not None else inputs_embeds_slice
    )
    input_length = input_ids_or_embs.shape[1]
    current_pad_length = current_key_values[0][1].shape[2] - input_length

    # the shift operator takes in the argument shift_to_left which indicates whether to shift to left or to right
    trimmed_current_key_values = tuple(
        (
            _shift(current_key, key_concat_axis, current_pad_length, pad_to_left),
            _shift(current_value, value_concat_axis, current_pad_length, pad_to_left),
        )
        for current_key, current_value in current_key_values
    )

    # slicing in place before sending to concat function to avoid  memory spiking.
    if unpadded_past_kv:
        concatenated_key_values = tuple(
            (
                _concat(unpadded_key, current_key, key_concat_axis),
                _concat(unpadded_value, current_value, value_concat_axis),
            )
            for (unpadded_key, unpadded_value), (current_key, current_value) in zip(
                unpadded_past_kv, trimmed_current_key_values
            )
        )
        return concatenated_key_values

    return trimmed_current_key_values
