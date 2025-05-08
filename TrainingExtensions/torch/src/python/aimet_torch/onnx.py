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
""" Defines onnx export API """
import contextlib
import io
import os
import tempfile
from typing import Any, Mapping, Tuple, Union

import onnx
import torch

from aimet_common.onnx._utils import _add_onnx_qdq_nodes

from .nn import QuantizationMixin
from .quantization import DequantizedTensor
from .quantization.base import EncodingBase
from .quantization.affine import AffineQuantizerBase
from .quantization.float import FloatQuantizeDequantize
from .quantsim import QuantizationSimModel
from .v2.experimental import onnx as _onnx


def export(model: Union[torch.nn.Module, QuantizationSimModel],
           args: Union[Tuple[Any, ...], torch.Tensor],
           f: Union[str, io.BytesIO],
           *posargs,
           export_int32_bias: bool = True,
           **kwargs):
    """
    Export QuantizationSimModel to onnx model with
    QuantizeLinear/DequantizeLinear embedded in the graph.

    This function takes set of same arguments as torch.onnx.export()
    """
    if isinstance(model, QuantizationSimModel):
        model = model.model

    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(
            f"aimet_torch.export only supports torch.nn.Module or QuantizationSimModel; got {type(model)}"
        )

    with contextlib.ExitStack() as stack:
        # Unfold all param quantizers to incorporate QuantizeLinear/DequantizeLinear
        # of those parameters in tracing time
        stack.enter_context(_temporarily_unfold_param_quantizers(model))

        if export_int32_bias:
            # Temoprarily instantiate int32 bias quantizers
            stack.enter_context(_concretize_int32_bias_quantizers(model, args))

        # Export quantize-dequantized weight
        # pylint: disable=protected-access
        stack.enter_context(QuantizationSimModel._apply_qdq_to_model_parameters(model))

        # Remove [b]float16 quantizers
        stack.enter_context(_remove_fp16_quantizers(model))

        onnx_model, tensor_to_encoding_map = _to_onnx(model, args, *posargs, **kwargs)

    onnx_qdq_model = _to_onnx_qdq(onnx_model, tensor_to_encoding_map)
    onnx.save(onnx_qdq_model, f)


def _to_onnx(model: torch.nn.Module,
             args: Union[Tuple[Any, ...], torch.Tensor],
             *posargs, **kwargs):
    _check_unsupported_quantizers(model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx_path = os.path.join(tmp_dir, "quantized_model.onnx")
        _onnx.export(model, args, tmp_onnx_path, *posargs, **kwargs)
        onnx_model = onnx.load(tmp_onnx_path)

        param_names = {
            f"{layer_name}.{param_name}"
            for layer_name, layer in model.named_modules()
            if isinstance(layer, QuantizationMixin)
            for param_name, quantizer in layer.param_quantizers.items()
            if quantizer
        }

    tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]]
    tensor_to_encoding_map = {
        name: (encoding, name in param_names)
        for name, encoding in _onnx.remove_quantization_nodes_from_onnx_graph(onnx_model).items()
    }
    return onnx_model, tensor_to_encoding_map


@contextlib.contextmanager
def _concretize_int32_bias_quantizers(model, args):
    if not isinstance(args, (tuple, list)):
        args = (args,)

    handles = []
    orig_bias_quantizers = {
        qmodule: qmodule.param_quantizers["bias"]
        for qmodule in model.modules()
        if isinstance(qmodule, QuantizationMixin)
           and "bias" in qmodule.param_quantizers
           and qmodule.bias is not None
    }

    try:
        for qmodule, qtzr in orig_bias_quantizers.items():
            if qtzr is not None:
                # Bias quantizer already exists.
                # This means the user created bias quantizer by him/herself
                # In this case, we honor the custom bias quantizer defined by the user
                continue

            if "weight" in qmodule.param_quantizers and \
                    isinstance(qmodule.param_quantizers["weight"], AffineQuantizerBase):
                # pylint: disable=protected-access
                handle = qmodule.register_forward_hook(type(qmodule)._create_int32_bias_quantizer)
                handles.append(handle)
        try:
            model(*args)
        finally:
            for handle in handles:
                handle.remove()
        yield
    finally:
        for qmodule, qtzr in orig_bias_quantizers.items():
            qmodule.param_quantizers["bias"] = qtzr


@contextlib.contextmanager
def _temporarily_unfold_param_quantizers(model: torch.nn.Module):
    # pylint: disable=protected-access
    """
    Temporarily re-instantiate param quantizers for ease of export
    """
    modules_with_folded_parameters = [
        qmodule for qmodule in model.modules()
        if isinstance(qmodule, QuantizationMixin) and
           any(isinstance(param, DequantizedTensor) for param in qmodule.parameters())
    ]

    try:
        for qmodule in modules_with_folded_parameters:
            qmodule._unfold_param_quantizers()
        yield
    finally:
        for qmodule in modules_with_folded_parameters:
            qmodule._fold_param_quantizers()


@contextlib.contextmanager
def _remove_fp16_quantizers(model: torch.nn.Module):
    """
    Temporarily remove [b]float16 quantizers for sim.onnx.export,
    as sim.onnx.export does NOT support exporting [b]float16 quantizers.
    """
    original_containers = {}

    try:
        for qmodule in model.modules():
            if not isinstance(qmodule, QuantizationMixin):
                continue

            for name, qtzr in qmodule.param_quantizers.items():
                if isinstance(qtzr, FloatQuantizeDequantize) and \
                        (qtzr.is_float16() or qtzr.is_bfloat16()):
                    original_containers[(qmodule.param_quantizers, name)] = qtzr
                    qmodule.param_quantizers[name] = None

            for i, qtzr in enumerate(qmodule.input_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and \
                        (qtzr.is_float16() or qtzr.is_bfloat16()):
                    original_containers[(qmodule.input_quantizers, i)] = qtzr
                    qmodule.input_quantizers[i] = None

            for i, qtzr in enumerate(qmodule.output_quantizers):
                if isinstance(qtzr, FloatQuantizeDequantize) and \
                        (qtzr.is_float16() or qtzr.is_bfloat16()):
                    original_containers[(qmodule.output_quantizers, i)] = qtzr
                    qmodule.output_quantizers[i] = None

        yield

    finally:
        for (container, key), qtzr in original_containers.items():
            container[key] = qtzr


def _to_onnx_qdq(onnx_model: onnx.ModelProto,
                 tensor_to_encoding_map: Mapping[str, Tuple[EncodingBase, bool]]) -> onnx.ModelProto:
    qnn_encodings = {
        name: encoding.to_qnn_encoding_dict("2.0.0.beta")
        for name, (encoding, _) in tensor_to_encoding_map.items()
    }
    qnn_encodings = {
        name: encoding for name, encoding in qnn_encodings.items() if encoding
    }

    qdq_tensor_names = {
        fp_tensor_name: f"{fp_tensor_name}_qdq"
        for fp_tensor_name in qnn_encodings
    }

    onnx_opset_version = next(opset.version for opset in onnx_model.opset_import if opset.domain == "")

    # Add onnx QDQ nodes in batch
    _add_onnx_qdq_nodes(onnx_model,
                        input_names=qnn_encodings.keys(),
                        output_names=qdq_tensor_names.values(),
                        node_name_prefixes=qnn_encodings.keys(),
                        encodings=qnn_encodings.values(),
                        onnx_opset=onnx_opset_version)

    # Restore model output names from "{output}_qdq" to "{output}"
    _restore_model_output_names(onnx_model, qdq_tensor_names)

    return onnx_model


def _check_unsupported_quantizers(module: torch.nn.Module):
    for qtzr in module.modules():
        if isinstance(qtzr, FloatQuantizeDequantize):
            if not qtzr.is_float16() and not qtzr.is_bfloat16():
                msg = " ".join([
                    "sim.onnx.export doesn't support exporting floating point encodings",
                    f"except [b]float16. Got {qtzr.bitwidth}-bit float encoding",
                ])
                raise RuntimeError(msg)


def _rename_inputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
    for node in onnx_model.graph.node:
        for i, old_name in enumerate(node.input):
            new_name = new_names.get(old_name, None)
            if new_name is not None:
                node.input[i] = new_name


def _rename_outputs(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
    for node in onnx_model.graph.node:
        for i, old_name in enumerate(node.output):
            new_name = new_names.get(old_name, None)
            if new_name is not None:
                node.output[i] = new_name


def _restore_model_output_names(onnx_model: onnx.ModelProto, new_names: Mapping[str, str]):
    """
    Rename model outputs. Assuming "output" is the model output,

    before:
        Softmax ----> output -------> QDQ -------> output_qdq

    after:
        Softmax ----> output__ -----> QDQ -------> output
    """
    _new_names = {
        output.name: f"{output.name}__"
        for output in onnx_model.graph.output
        if output.name in new_names
    }
    _rename_inputs(onnx_model, _new_names)

    _new_names.update({
        new_names[output.name]: output.name
        for output in onnx_model.graph.output
        if output.name in new_names
    })
    _rename_outputs(onnx_model, _new_names)
