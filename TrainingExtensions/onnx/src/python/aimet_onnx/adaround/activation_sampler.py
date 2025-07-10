# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

"""Sample output from original module for Adaround feature"""

import copy
from typing import Tuple, List, Dict, Union

import numpy as np
import onnxruntime as ort
import onnx
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import (
    add_hook_to_get_activation,
    remove_activation_hooks,
    create_input_dict,
)

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import ModelProto
else:
    from onnx.onnx_pb import ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


# TODO: Remove redundant session build/teardown
#       and and always use "EXHUASTIVE" (ORT default policy)
if ort.__version__ < "1.20":
    _cudnn_conv_algo_search = "DEFAULT"
else:
    _cudnn_conv_algo_search = "HEURISTIC"


class ActivationSampler:
    """
    For a module in the original model and the corresponding module in the weight quantized QuantSim model,
    collect the module's output and input activation data respectively
    """

    def __init__(
        self,
        module_input_fp: str,
        module_output_quant: str,
        quant_model: QuantizationSimModel,
        use_cuda: bool,
        device: int = 0,
    ):
        """
        :param module_input_fp: FP input tensor name of the module to retrieve
        :param module_output_quant: Quant output tensor name of the module to retrieve
        :param quant_model: Session with the model with quantization simulations ops
        :param use_cuda: If we should use cuda
        :param device: CUDA device ID
        :return: Input data to quant op, Output data from original op
        """
        self._quant_model = quant_model

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.warning(
                "CUDAExecutionProvider not in ort available providers. use_cuda is set to False"
            )
            use_cuda = False
        if use_cuda:
            self.providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device,
                        "cudnn_conv_algo_search": _cudnn_conv_algo_search,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            self.providers = ["CPUExecutionProvider"]

        orig_model = copy.deepcopy(quant_model.model)
        orig_model = QuantizationSimModel.remove_quantizers(orig_model)

        self._orig_module_collector = ModuleData(
            orig_model, module_input_fp, self.providers, quant_model._user_onnx_libs
        )
        self._quant_module_collector = ModuleData(
            quant_model.model,
            module_output_quant,
            self.providers,
            quant_model._user_onnx_libs,
        )

    def sample_and_place_all_acts_on_cpu(self, dataset) -> Tuple:
        """
        From the original module, collect output activations and input activations
        to corresponding quantized module.

        NOTE: Keeps collected activation data on CPU memory so this function should only be invoked
        if collected activation data can be fit entirely in CPU memory.

        :param dataset: Cached dataset.
        :return: Input data, output data
        """
        all_inp_data = []
        all_out_data = []

        iterator = iter(dataset)
        for batch_index in range(len(dataset)):
            model_inputs = next(iterator)
            inp_data, out_data = self.sample_acts(
                create_input_dict(self._quant_model.model.model, model_inputs)
            )

            all_inp_data.append(inp_data[0])
            all_out_data.append(out_data[0])

            if batch_index == len(dataset) - 1:
                break

        return all_inp_data, all_out_data

    def sample_acts(
        self, model_inputs: Dict[str, List[np.ndarray]]
    ) -> Tuple[List, List]:
        """
        For given model_inputs, collect input activations data to quant module and
        output activations data from original module.

        :param model_inputs: Model inputs.
        :return: Input and output activations data.
        """
        # Collect input activation data to quantized wrapper module
        # (with all preceding weight modules quantized)
        inp_data = self._quant_module_collector.collect_activation(model_inputs)
        # Collect output activation data from original module
        out_data = self._orig_module_collector.collect_activation(model_inputs)
        return inp_data, out_data


class ModuleData:
    """
    Collect activation tensor for the given model and model input
    """

    def __init__(
        self,
        model: ModelProto,
        activation_name: str,
        providers: List,
        user_onnx_libs: List[str] = None,
    ):
        """
        :param model: ONNX model
        :param activation_name: tensor corresponding to activation name to fetch
        :param providers: CPU/GPU execution providers
        :param user_onnx_libs: List of paths to all compiled ONNX custom ops libraries
        """
        self._model = model
        self._activation_name = activation_name
        self._providers = providers
        self._user_onnx_libs = user_onnx_libs

    def collect_activation(self, model_input: Dict[str, List[np.ndarray]]) -> List:
        """
        Collect activation using the model_input

        :param model_input: Input to model
        :return: Activation corresponding to the model_input passed
        """

        handle = add_hook_to_get_activation(self._model.model, self._activation_name)
        sess = QuantizationSimModel.build_session(
            self._model.model, self._providers, self._user_onnx_libs
        )
        if self._activation_name in model_input:
            # Workaround memory corruption bug in onnxruntime >= 1.19 when a graph output is also a graph input
            # https://github.com/microsoft/onnxruntime/issues/21922
            outputs = [model_input[self._activation_name]]
        else:
            outputs = sess.run([self._activation_name], model_input)
        remove_activation_hooks(self._model.model, handle)

        return outputs
