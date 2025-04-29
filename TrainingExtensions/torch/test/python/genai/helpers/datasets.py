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
""" Datasets for GenAI testing """

from abc import ABC, abstractmethod
from datasets import load_dataset
import torch
from transformers import PreTrainedTokenizer

class Dataset(ABC):
    """ Generic GenAI Dataset class """
    @staticmethod
    @abstractmethod
    def load_dataset(split: str):
        """ Load dataset from huggingface """

    @classmethod
    @abstractmethod
    def load_encoded_dataset(cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str):
        """ Load encoded and chunked dataset """


class ChunkedDataset(torch.utils.data.Dataset):
    """ Internal helper class to chunk input IDs to static graph context length """
    def __init__(self, tokenized_data: dict[str, torch.Tensor], size_per_chunk: int):
        self.tokenized_data = tokenized_data
        self.size_per_chunk = size_per_chunk

    def __len__(self):
        return len(self.tokenized_data['input_ids'][0]) // self.size_per_chunk

    def __getitem__(self, index: int):
        start_index = index * self.size_per_chunk
        end_index = (index + 1) * self.size_per_chunk
        return {'input_ids': self.tokenized_data['input_ids'][:, start_index:end_index],
                'attention_mask': self.tokenized_data['attention_mask'][:, start_index:end_index]}

class Wikitext(Dataset):
    """ Wikitest dataset """
    @staticmethod
    def load_dataset(split: str):
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    @classmethod
    def load_encoded_dataset(cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str):
        dataset_split = cls.load_dataset(split)
        encoded_dataset_split = tokenizer(
            "\n\n".join(dataset_split["text"]),
            return_tensors="pt",
            add_special_tokens=True
        )

        return ChunkedDataset(encoded_dataset_split, context_length)

class TinyMMLU(Dataset):
    """ TinyMMLU dataset"""
    @staticmethod
    def load_dataset(split: str):
        return load_dataset("tinyBenchmarks/tinyMMLU", split=split)

    @classmethod
    def load_encoded_dataset(cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str):
        dataset_split = cls.load_dataset(split)

        def tokenize(samples):
            tokenized_question = tokenizer(
                samples["input_formatted"],
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), samples["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
                "input_formatted",
            ])
