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
""" GenAI test runner """

import pytest
import torch
import gc

from .models import GenAIModel, Llama_32_1B, Qwen_25_15B
from .helpers.datasets import Dataset, Wikitext
from .helpers.quant_recipes import QuantizationTechnique, RemoveQuantization, PCQ, LPBQ, AdaScale
from .helpers.metrics import PPL, MMLU
from .helpers.profiler import ResourceProfiler, write_stats_to_disk

from aimet_torch.utils import place_model

@pytest.mark.skip(reason="This test is currently under development")
@pytest.mark.parametrize("model, sequence_length, context_length", [
    [Llama_32_1B, 2048, 4096],
    [Qwen_25_15B, 2048, 4096],
])
@pytest.mark.parametrize("quant_dataset", [Wikitext])
@pytest.mark.parametrize("quant_technique", [RemoveQuantization, AdaScale, PCQ, LPBQ])
def test_llm_quantization(model: GenAIModel,
                          sequence_length: int,
                          context_length: int,
                          quant_dataset: Dataset,
                          quant_technique: QuantizationTechnique):
    """ GenAI test runner function """

    # Adding this explicitly so that memory is cleared before the test is started, and collected stats are accurate
    gc.collect()
    torch.cuda.empty_cache()

    test_statistics = {}

    quantsim = model.instantiate_quantsim(context_length, sequence_length)
    tokenizer = model.instantiate_tokenizer()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        with ResourceProfiler() as profiler:
            train_dataset = quant_dataset.load_encoded_dataset(tokenizer, sequence_length, 'train')
            quant_technique.apply(quantsim, train_dataset)
        test_statistics[f"{quant_technique.__name__}+{quant_dataset.__name__}"] = profiler.as_dict()

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for metric in (PPL, MMLU):
                with ResourceProfiler() as profiler:
                    result = metric.evaluate(quantsim.model, tokenizer, context_length)
                    print(f"{metric.__name__} result: {result}")
                test_statistics[f"{metric.__name__}"] = {"result": result} | profiler.as_dict()

    write_stats_to_disk("../genai_scorecard_data.json",
                        model,
                        (sequence_length, context_length),
                        quant_technique,
                        test_statistics)
