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
""" Metrics for GenAI testing """

from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .datasets import Wikitext, TinyMMLU

class EvaluationMetric(ABC):
    """ Generic GenAI evaluation metric """
    @staticmethod
    @abstractmethod
    def evaluate(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, context_length: int) -> float:
        """ Perform evaluation on provided model """


class PPL(EvaluationMetric):
    """ PPL evaluation metric """
    @staticmethod
    def _compute_loss_from_logits(output_logits: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """ Helper function to compute loss """

        # Get the outputs and move it to CPU. Assumes that index 0 is logits as
        lm_logits = output_logits.cpu()

        # Trim the last logit off lm_logits, and the first token off input_tokens
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = input_tokens[..., 1:].contiguous().to(shift_logits.device)

        loss_fn = torch.nn.CrossEntropyLoss()
        neg_log_likelihood = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return neg_log_likelihood

    @staticmethod
    @torch.no_grad()
    def evaluate(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, context_length: int) -> float:
        dataset = Wikitext.load_encoded_dataset(tokenizer, context_length , "test")
        dataloader = DataLoader(dataset)

        neg_log_likelihoods = []
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating PPL"):
            batch["input_ids"] = batch["input_ids"].to(model.device)
            outputs = model(input_ids=batch["input_ids"][0])
            neg_log_likelihoods.append(PPL._compute_loss_from_logits(outputs[0], batch["input_ids"]))
            del outputs

        ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
        return float(ppl)

class MMLU(EvaluationMetric):
    """ Generic MMLU evaluation metric. Should work with any MMLU dataset. """
    @staticmethod
    def evaluate(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, context_length: int) -> float:
        dataset = TinyMMLU.load_encoded_dataset(tokenizer, context_length , "test")
        dataloader = DataLoader(dataset)

        def tokenize_letter(letter: str):
            return torch.Tensor(tokenizer(letter, add_special_tokens=False)["input_ids"]).to(dtype=torch.int)
        choices = tuple(tokenize_letter(letter) for letter in ("A", "B", "C", "D"))

        correct_predictions = 0

        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating MMLU"):
            batch["input_ids"] = torch.Tensor(batch["input_ids"]).to(dtype=torch.int, device=model.device).unsqueeze(0)
            outputs = model(input_ids=batch["input_ids"])

            last_logit = outputs[0][..., -1, :].contiguous().to(dtype=torch.float32, device="cpu").flatten()
            last_logit = torch.nn.functional.log_softmax(last_logit, dim=-1)

            scores = tuple(last_logit[choice] for choice in choices)
            index = scores.index(max(scores))
            prediction = choices[index]
            label = torch.Tensor(batch["label"]).to(dtype=torch.int)

            if prediction == label:
                correct_predictions += 1

        return float(correct_predictions / len(dataloader))
