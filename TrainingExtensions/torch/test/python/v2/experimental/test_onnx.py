# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import os
import json
import onnxruntime as ort
import pytest
import contextlib
import numpy as np
import torch
import onnx
import tempfile
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_common import quantsim as quantsim_common
import aimet_torch.v2 as aimet
import aimet_torch.v2.quantization as Q
from aimet_torch.v2.quantsim import QuantizationSimModel
from torchvision.models import resnet18, mobilenet_v3_small
from aimet_torch.v2.experimental.onnx._export import export as _export
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.utils import get_all_quantizers
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights


@pytest.fixture(autouse=True, params=range(1))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)


@contextlib.contextmanager
def set_encoding_version(version):
    try:
        old_version = quantsim_common.encoding_version
        quantsim_common.encoding_version = version
        yield
    finally:
        quantsim_common.encoding_version = old_version


@pytest.mark.parametrize("qtzr_cls", [Q.affine.Quantize, Q.affine.QuantizeDequantize])
@pytest.mark.parametrize("input_shape, scale_shape, block_size", [
                         ([],          [],          None      ), # per-tensor
                         ((100, 100),  (1,),        None      ), # per-tensor
                         ((100, 100),  [],          None      ), # per-tensor
                         ((100, 100),  (100, 1),    None      ), # per-channel
                         ((100, 100),  (100, 1),    (1, 100)  ), # per-channel
                         ((100, 100),  (100, 50),   (1, 2)    ), # blockwise
                         ((100, 100),  (50, 100),   (2, 1)    ), # blockwise
                         ((100, 100),  (50, 50),    (2, 2)    ), # blockwise
                         ((100, 100),  (50, 50),    (-1, -1)  ), # blockwise
])
@pytest.mark.parametrize("symmetric", [True, False])
def test_quantize_torch_ort_equal(qtzr_cls, input_shape, scale_shape, block_size, symmetric):
    """
    When: Export a quantizer with torch.onnx.export
    """
    x = torch.randn(input_shape)
    qtzr = qtzr_cls(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        _ = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            _export(qtzr, x, f, input_names=['input'], output_names=['output'])

        with torch.no_grad():
            y = qtzr(x)

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == 'aimet']
        assert len(nodes) == 1
        node, = nodes

        assert node.name == '/quantize' if qtzr_cls is Q.affine.Quantize else '/quantize_dequantize'
        assert node.attribute[0].name == 'block_size'
        assert node.attribute[0].ints == ([1] if block_size is None else list(np.array(input_shape) // np.array(scale_shape)))
        assert node.attribute[1].name == 'qmax'
        assert node.attribute[1].i == (127 if symmetric else 255)
        assert node.attribute[2].name == 'qmin'
        assert node.attribute[2].i == (-128 if symmetric else 0)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper scale and offset values
        """
        const_map = {node.output[0]: node for node in model.graph.node if node.op_type == "Constant"}
        assert node.input[1] in const_map
        assert node.input[2] in const_map
        onnx_scale = torch.tensor(onnx.numpy_helper.to_array(const_map[node.input[1]].attribute[0].t))
        onnx_offset = torch.tensor(onnx.numpy_helper.to_array(const_map[node.input[2]].attribute[0].t))
        if scale_shape == []:
            onnx_scale.squeeze_(0)
            onnx_offset.squeeze_(0)
        assert torch.equal(onnx_scale, qtzr.get_scale())
        assert torch.equal(onnx_offset, qtzr.get_offset())

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)


@pytest.mark.parametrize("input_shape, scale_shape, block_size", [
                         ([],          [],          None      ), # per-tensor
                         ((100, 100),  (1,),        None      ), # per-tensor
                         ((100, 100),  [],          None      ), # per-tensor
                         ((100, 100),  (100, 1),    None      ), # per-channel
                         ((100, 100),  (100, 1),    (1, 100)  ), # per-channel
                         ((100, 100),  (100, 50),   (1, 2)    ), # blockwise
                         ((100, 100),  (50, 100),   (2, 1)    ), # blockwise
                         ((100, 100),  (50, 50),    (2, 2)    ), # blockwise
                         ((100, 100),  (50, 50),    (-1, -1)  ), # blockwise
])
@pytest.mark.parametrize("symmetric", [True, False])
def test_dequantize_torch_ort_equal(input_shape, scale_shape, block_size, symmetric):
    """
    When: Export dequantize with torch.onnx.export
    """

    class Dequantize(torch.nn.Module):
        def forward(self, x: Q.QuantizedTensor):
            return x.dequantize()

    x = torch.randn(input_shape)
    qtzr = Q.affine.Quantize(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        x = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            _export(Dequantize(), x, f, input_names=['input'], output_names=['output'])

        with torch.no_grad():
            y = x.dequantize()

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == 'aimet']
        assert len(nodes) == 1
        node, = nodes

        assert node.name == '/dequantize'
        assert node.attribute[0].name == 'block_size'
        assert node.attribute[0].ints == ([1] if block_size is None else list(np.array(input_shape) // np.array(scale_shape)))

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)



@torch.no_grad()
@pytest.mark.parametrize(
    "model_factory,      input_shape", [
    (resnet18,           (1, 3, 224, 224)),
    (mobilenet_v3_small, (1, 3, 224, 224)),
])
def test_export_torchvision_models(model_factory, input_shape):
    """
    When: Export quantized torchvision model
    """
    x = torch.randn(input_shape)
    model = model_factory().eval()
    model = prepare_model(model)
    model = QuantizationSimModel(model, x, config_file=get_path_for_per_channel_config()).model

    with aimet.nn.compute_encodings(model):
        model(x)

    y = model(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "torchvision_model.onnx")

        with open(full_path, "wb") as f:
            _export(model, x, f, input_names=['input'], output_names=['output'])

        """
        Then: The saved onnx model should pass onnx model checker
        """
        onnx_model = onnx.load_model(full_path)
        onnx.checker.check_model(onnx_model)

        """
        Then: The onnx model should have the same number of quant nodes
              as the number of quantizers in the original pytorch model
        """
        nodes = [node for node in onnx_model.graph.node if node.domain == 'aimet']
        quantizers_in_model = [qtzr for qtzr_group in get_all_quantizers(model) for qtzr in qtzr_group if qtzr]
        assert len(nodes) == len(quantizers_in_model)

        """
        Then: The quant nodes in the onnx model should have constant scale and offset values
        """
        const_map = {node.output[0]: node for node in onnx_model.graph.node if node.op_type == "Constant"}
        for node in nodes:
            assert node.input[1] in const_map
            assert node.input[2] in const_map

        """
        Then: The onnx model should produce output close enough to the original pytorch model
        """
        sess = ort.InferenceSession(full_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {'input': x.numpy()})

        # Allow off-by-3 error
        atol = 3 * y.encoding.scale.item()
        assert torch.allclose(torch.from_numpy(out), y, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("encoding_version", ["0.6.1", "1.0.0", "2.0.0.beta"])
@pytest.mark.parametrize("lpbq", [False, True])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
@pytest.mark.parametrize(
    "weight_dtype,  activation_dtype", [
    (torch.int8,    torch.uint8),
    (torch.int8,    torch.float16),
    (torch.float16, torch.float16),
])
def test_quantsim_export_resnet18(encoding_version, lpbq: bool, fold_param_quantizers: bool,
                                  weight_dtype: torch.dtype, activation_dtype: torch.dtype):
    """
    When: Export quantized torchvision model using quantsim.export
    """
    x = torch.randn(1, 3, 224, 224)
    model = resnet18().eval()
    model = prepare_model(model)
    fold_all_batch_norms(model, None, x)
    sim = QuantizationSimModel(model, x,
                               default_param_bw=weight_dtype.itemsize * 8,
                               default_output_bw=activation_dtype.itemsize * 8)

    if lpbq:
        set_grouped_blockwise_quantization_for_weights(sim,
                                                       [sim.model.fc],
                                                       bitwidth=4,
                                                       symmetric=True,
                                                       decompressed_bw=8,
                                                       block_size=64)

    if weight_dtype.is_floating_point:
        for qmodule in sim.qmodules():
            for name, qtzr in qmodule.param_quantizers.items():
                if not qtzr:
                    continue
                qmodule.param_quantizers[name] = Q.float.FloatQuantizeDequantize(dtype=weight_dtype)

    if activation_dtype.is_floating_point:
        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.input_quantizers):
                if not qtzr:
                    continue
                qmodule.input_quantizers[i] = Q.float.FloatQuantizeDequantize(dtype=activation_dtype)

        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.output_quantizers):
                if not qtzr:
                    continue
                qmodule.output_quantizers[i] = Q.float.FloatQuantizeDequantize(dtype=activation_dtype)

    sim.compute_encodings(lambda model: model(x))

    # Compute original pytorch model output with qdq weights
    with sim._concretize_int32_bias_quantizers(x):
        expected_param_encodings = {
            f"{module_name}.{param_name}": qtzr.get_encodings().to_qnn_encoding_dict(encoding_version)
            for module_name, qmodule in sim.named_qmodules()
            for param_name, qtzr in qmodule.param_quantizers.items()
            if isinstance(qtzr, Q.affine.AffineQuantizerBase)
        }
        expected_activation_encodings = {}
        expected_activation_encodings.update({
            f"{module_name}.input_quantizers.{i}": qtzr.get_encodings().to_qnn_encoding_dict(encoding_version)
            for module_name, qmodule in sim.named_qmodules()
            for i, qtzr in enumerate(qmodule.input_quantizers)
            if isinstance(qtzr, Q.affine.AffineQuantizerBase)
        })
        expected_activation_encodings.update({
            f"{module_name}.output_quantizers.{i}": qtzr.get_encodings().to_qnn_encoding_dict(encoding_version)
            for module_name, qmodule in sim.named_qmodules()
            for i, qtzr in enumerate(qmodule.output_quantizers)
            if isinstance(qtzr, Q.affine.AffineQuantizerBase)
        })

        with remove_activation_quantizers(sim.model):
            expected_out = sim.model(x)

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    with tempfile.TemporaryDirectory() as dirname:
        onnx_path = os.path.join(dirname, "torchvision_model.onnx")
        encodings_path = os.path.join(dirname, "torchvision_model.encodings")

        with set_encoding_version(encoding_version):
            sim.onnx.export(x, onnx_path, input_names=["input"], output_names=["output"])

        """
        Then: The saved onnx model should pass onnx model checker
        """
        onnx_model = onnx.load_model(onnx_path)
        onnx.checker.check_model(onnx_model)

        """
        Then: Input/Output names should be strictly honored
        """
        assert list(x.name for x in onnx_model.graph.input) == ["input"]
        assert list(y.name for y in onnx_model.graph.output) == ["output"]

        with open(encodings_path) as f:
            onnx_encodings = json.load(f)

        """
        Then: The onnx encodings should have the same number of encodings
              as the number of quantizers in the original pytorch model
        """
        if encoding_version < "2.0.0":
            assert len(onnx_encodings['param_encodings']) == len(expected_param_encodings)
            assert len(onnx_encodings['activation_encodings']) == len(expected_activation_encodings)
        else:
            assert len(onnx_encodings["encodings"]) == len(expected_param_encodings) + \
                                                       len(expected_activation_encodings)

        """
        Then: The onnx encodings should have the same scale and offset value
              as the values of quantizers in the original pytorch model
        """
        if encoding_version == '0.6.1':
            assert onnx_encodings['param_encodings'] == expected_param_encodings

            for e in onnx_encodings['activation_encodings'].values():
                assert any(
                    e[0]["scale"] == expected[0]["scale"] and
                    e[0]["offset"] == expected[0]["offset"] and
                    e[0]["bitwidth"] == expected[0]["bitwidth"]
                    for expected in expected_activation_encodings.values()
                )
        elif encoding_version == "1.0.0":
            for e in onnx_encodings['param_encodings']:
                name = e.pop("name")
                assert e == expected_param_encodings[name]

            for e in onnx_encodings['activation_encodings']:
                assert any(
                    e["scale"] == expected["scale"] and
                    e["offset"] == expected["offset"] and
                    e["bw"] == expected["bw"]
                    for expected in expected_activation_encodings.values()
                )
        elif encoding_version == "2.0.0.beta":
            expected_encodings = expected_param_encodings | expected_activation_encodings

            for e in onnx_encodings["encodings"]:
                name = e.pop("name")
                if name in expected_encodings:
                    assert e == expected_encodings[name]
                    continue

                assert any(
                    e.get("output_dtype") == expected.get("output_dtype") and
                    e.get("y_scale")      == expected.get("y_scale") and
                    e.get("y_zero_point") == expected.get("y_zero_point") and
                    e.get("per_channel_float_scale") == expected.get("per_channel_float_scale") and
                    e.get("per_block_int_scale")     == expected.get("per_block_int_scale")
                    for expected in expected_encodings.values()
                )
        else:
            raise RuntimeError(f"Unexpected encoding veresion: {encoding_version}")

        """
        Then: The exported onnx model should produce output close enough to
              the original pytorch model with qdq weights
        """
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        out, = sess.run(None, {"input": x.numpy()})

        assert torch.allclose(torch.from_numpy(out), expected_out, atol=1e-5)
