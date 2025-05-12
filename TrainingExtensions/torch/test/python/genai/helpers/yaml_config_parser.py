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
""" General utils for GenAI model testing """

import inspect
import yaml

from . import datasets as genai_datasets
from . import metrics as genai_metrics
from . import quant_recipes as genai_quant_recipes
from .. import models as genai_models

genai_dataset_classes = [obj for name, obj in inspect.getmembers(genai_datasets) if inspect.isclass(obj) and issubclass(obj, genai_datasets.Dataset)]
genai_metric_classes = [obj for name, obj in inspect.getmembers(genai_metrics) if inspect.isclass(obj) and issubclass(obj, genai_metrics.EvaluationMetric)]
genai_quant_recipe_classes = [obj for name, obj in inspect.getmembers(genai_quant_recipes) if inspect.isclass(obj) and issubclass(obj, genai_quant_recipes.QuantizationTechnique)]
genai_model_classes = [obj for name, obj in inspect.getmembers(genai_models) if inspect.isclass(obj) and issubclass(obj, genai_models.GenAIModel)]

class YAMLConfigParser:
    recipe_lookup: dict = {}
    model_lookup: dict = {}
    dataset_lookup: dict = {}
    metrics_lookup: dict = {}

    @classmethod
    def register_model(cls, model_cls):
        cls.model_lookup[model_cls.__name__] = model_cls

    @classmethod
    def register_recipe(cls, recipe_cls):
        cls.recipe_lookup[recipe_cls.__name__] = recipe_cls

    @classmethod
    def register_dataset(cls, dataset_cls):
        cls.dataset_lookup[dataset_cls.__name__] = dataset_cls

    @classmethod
    def register_metrics(cls, metrics_cls):
        cls.metrics_lookup[metrics_cls.__name__] = metrics_cls

    @classmethod
    def get_model(cls, model_name):
        return cls.model_lookup[model_name]

    @classmethod
    def get_recipe(cls, recipe_name):
        return cls.recipe_lookup[recipe_name]

    @classmethod
    def get_dataset(cls, dataset_name):
        return cls.dataset_lookup[dataset_name]

    @classmethod
    def get_metrics(cls, metrics_name):
        return cls.metrics_lookup[metrics_name]

    @classmethod
    def validate_config(cls, doc):
        if "model" not in doc:
            raise RuntimeError("Model not specified.")
        if "dataset" not in doc:
            raise RuntimeError("Dataset not specified.")
        if "recipe" not in doc:
            raise RuntimeError("Recipe not specified.")
        if "metrics" not in doc:
            raise RuntimeError("Metrics not specified.")

        if not isinstance(doc["model"], dict):
            raise RuntimeError("Multiple models cannot be specified in a single document.")
        if not isinstance(doc["dataset"], dict):
            raise RuntimeError("Multiple datasets cannot be specified in a single document.")
        if not isinstance(doc["recipe"], dict):
            raise RuntimeError("Multiple recipes cannot be specified in a single document.")

        if "name" not in doc["model"]:
            raise RuntimeError("Model name not specified.")
        if "name" not in doc["dataset"]:
            raise RuntimeError("Dataset name not specified.")
        if "name" not in doc["recipe"]:
            raise RuntimeError("Quantization recipe name not specified.")

        if "sequence_length" not in doc["model"]:
            raise RuntimeError("Sequence length not specified.")
        if "context_length" not in doc["model"]:
            raise RuntimeError("Context length not specified.")

        metrics = doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
        for metric in metrics:
            if "name" not in metric:
                raise RuntimeError("Metric name not specified.")

    @classmethod
    def parse_document(cls, doc):
        cls.validate_config(doc)
        task_params = {}

        model_name = doc["model"]["name"]
        try:
            model_cls = cls.get_model(model_name)
            task_params["model"] = doc.pop("model")
            task_params["model"]["class"] = model_cls
            del task_params["model"]["name"]
        except LookupError as exc:
            raise LookupError(f"Specified model name ({model_name}) not found.") from exc

        dataset_name = doc["dataset"]["name"]
        try:
            dataset_cls = cls.get_dataset(dataset_name)
            task_params["dataset"] = doc.pop("dataset")
            task_params["dataset"]["class"] = dataset_cls
            del task_params["dataset"]["name"]
        except LookupError as exc:
            raise LookupError(f"Specified dataset name ({dataset_name}) not found.") from exc

        recipe_name = doc["recipe"]["name"]
        try:
            recipe_cls = cls.get_recipe(recipe_name)
            task_params["recipe"] = doc.pop("recipe")
            task_params["recipe"]["class"] = recipe_cls
            del task_params["recipe"]["name"]
        except LookupError as exc:
            raise LookupError(f"Specified quantization recipe name ({recipe_name}) not found.") from exc

        metrics = doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
        task_params["metrics"] = []
        for metric in metrics:
            metric_name = metric["name"]
            try:
                metric_cls = cls.get_metrics(metric_name)
                task_params["metrics"].append(metric)
                task_params["metrics"][-1]["class"] = metric_cls
                del task_params["metrics"][-1]["name"]
            except LookupError as exc:
                raise LookupError(f"Specified metric name ({metric_name}) not found.") from exc
        del doc["metrics"]

        task_params["profiler"] = doc.pop("profiler", {})

        if len(doc) > 0:
            raise ValueError(f"Unrecognized sections in config: {doc.keys()}")

        return task_params

    @classmethod
    def parse(cls, filename):
        print(filename)
        with open(filename, 'r') as file:
            docs = yaml.safe_load_all(file)
            for doc in docs:
                yield cls.parse_document(doc)

for dataset in genai_dataset_classes:
    YAMLConfigParser.register_dataset(dataset)
for metric in genai_metric_classes:
    YAMLConfigParser.register_metrics(metric)
for quant_recipe in genai_quant_recipe_classes:
    YAMLConfigParser.register_recipe(quant_recipe)
for model in genai_model_classes:
    YAMLConfigParser.register_model(model)
