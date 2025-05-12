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
import os
from pathlib import Path

from aimet_torch.utils import place_model

from .helpers.profiler import ResourceProfiler, write_stats_to_disk

@pytest.mark.skipif(lambda test_parameters: test_parameters is None, reason="No GenAI test parameters provided.")
def test_llm_quantization(test_parameters):
    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id", None)

    dataset_kwargs = test_parameters.pop("dataset")
    dataset_cls = dataset_kwargs.pop("class")

    recipe_kwargs = test_parameters.pop("recipe")
    recipe_cls = recipe_kwargs.pop("class")

    profiler_kwargs = test_parameters.pop("profiler")

    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    test_statistics = {}

    quantsim = model_cls.instantiate_quantsim(model_id, context_length, sequence_length, **model_kwargs)
    tokenizer = model_cls.instantiate_tokenizer(model_id)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with place_model(quantsim.model, device):
        with ResourceProfiler(**profiler_kwargs) as profiler:
            train_dataset = dataset_cls.load_encoded_dataset(tokenizer, sequence_length, **dataset_kwargs)
            recipe_cls.apply(quantsim, train_dataset, **recipe_kwargs)
        test_statistics[f"{recipe_cls.__name__}+{dataset_cls.__name__}"] = profiler.as_dict()

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for metric_kwargs in metrics:
                metric_cls = metric_kwargs.pop("class")
                with ResourceProfiler(**profiler_kwargs, disable_constant_sampling=True) as profiler:
                    result = metric_cls.evaluate(quantsim.model, tokenizer, context_length, **metric_kwargs)
                    print(f"{metric_cls.__name__} result: {result}")
                test_statistics[f"{metric_cls.__name__}"] = {"result": result} | profiler.as_dict()

    model_kwargs["context_length"] = context_length
    model_kwargs["sequence_length"] = sequence_length
    model_kwargs["model_id"] = model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID

    output_folder = Path(os.getcwd()) / "genai_test_artifacts"
    output_folder.mkdir(parents=True, exist_ok=True)
    write_stats_to_disk(str(output_folder / "profiling_data.json"),
                        model_cls,
                        model_kwargs,
                        recipe_cls,
                        recipe_kwargs,
                        test_statistics)
