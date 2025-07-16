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

"""Unit tests for Adaround Weights"""

import copy
import os
import json
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import pytest
from onnxsim import simplify
import onnx

from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_onnx import apply_adaround, QuantizationSimModel
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_onnx.adaround.utils import AdaroundSupportedModules
from aimet_onnx.utils import make_dummy_input, ParamUtils, build_session
from .models import models_for_tests


class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        np.random.seed(0)
        torch.manual_seed(0)
        model = models_for_tests.single_residual_model()
        data_loader = dataloader(input_shape=(1, 3, 32, 32))
        dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        sess = build_session(model.model, providers)
        out_before_ada = sess.run(None, dummy_input)

        def callback(session, args):
            in_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(
                model,
                params,
                tempdir,
                "dummy",
                use_cuda="CUDAExecutionProvider" in providers,
            )
            sess = build_session(ada_rounded_model.model, providers)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, "dummy.encodings")) as json_file:
                encoding_data = json.load(json_file)

            param_names = {encoding["name"] for encoding in encoding_data}
            params = {
                node.input[1]
                for node in model.nodes()
                if node.op_type in AdaroundSupportedModules
            }
            assert params.issubset(param_names)

    @pytest.mark.parametrize("pre_calibrate", [True, False])
    @pytest.mark.parametrize(
        "model",
        [
            models_for_tests.single_residual_model().model,
            models_for_tests.depthwise_conv_model().model,
            models_for_tests.add_matmul_model(),
            models_for_tests.model_with_split_matmul(),
        ],
    )
    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround_2(self, providers, model, pre_calibrate):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(copy.deepcopy(model), providers=providers)
        graph_outputs = sim.model.graph().output
        is_enabled = {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}

        adaroundable_ops = [
            op
            for op in sim.connected_graph.ordered_ops
            if op.type in ("Conv", "MatMul", "Gemm")
        ]
        weight_tensors = {
            t.name: copy.deepcopy(onnx.numpy_helper.to_array(t.tensor))
            for op in adaroundable_ops
            for t, param_type in op.parameters.values()
            if param_type == "weight"
        }

        assert weight_tensors

        if pre_calibrate:
            sim.compute_encodings(inputs)

        apply_adaround(sim, inputs, num_iterations=5)

        # Quantizer enabled state must be restored after adaround
        for name, q in sim.qc_quantize_op_dict.items():
            assert q.enabled == is_enabled[name]

        # Only optimized weights should have frozen encodings
        for name, quantizer in sim.qc_quantize_op_dict.items():
            assert quantizer.is_encoding_frozen() == (name in weight_tensors)

        # Optimized weight should not be equal to original weight
        for name, old_weight in weight_tensors.items():
            new_weight = onnx.numpy_helper.to_array(
                ParamUtils.get_param_by_name(sim.model.model, name)
            )
            assert not np.all(old_weight == new_weight)

        assert graph_outputs == sim.model.graph().output

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround_for_custom_op(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        from onnxruntime_extensions import get_library_path

        onnx_library = get_library_path()

        np.random.seed(0)
        torch.manual_seed(0)
        model = models_for_tests.custom_add_model()
        data_loader = dataloader(input_shape=(1, 3, 64, 64))
        dummy_input = {"input": np.random.rand(1, 3, 64, 64).astype(np.float32)}
        sess = build_session(model.model, providers, [onnx_library])
        out_before_ada = sess.run(None, dummy_input)

        def callback(session, args):
            in_tensor = {"input": np.random.rand(1, 3, 64, 64).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(
                model,
                params,
                tempdir,
                "dummy",
                user_onnx_libs=[onnx_library],
                use_cuda="CUDAExecutionProvider" in providers,
            )
            sess = build_session(ada_rounded_model.model, providers, [onnx_library])
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, "dummy.encodings")) as json_file:
                encoding_data = json.load(json_file)

            param_keys = {enc["name"] for enc in encoding_data}
            params = {
                node.input[1]
                for node in model.nodes()
                if node.op_type in AdaroundSupportedModules
            }
            assert params.issubset(param_keys)

    @pytest.mark.parametrize(
        "model, input_shape",
        [
            (models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),
            (models_for_tests.weight_gemm_model(10, 20, False), (1, 10)),
            (models_for_tests.weight_matmul_model(10, 20), (1, 10, 10)),
        ],
    )
    def test_adaround_matmul_gemm(self, model, input_shape, tmpdir):
        data_loader = dataloader(input_shape, input_shape[0])

        def callback(session, args):
            in_tensor = {"input": np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        Adaround.apply_adaround(model, params, tmpdir, "dummy", use_cuda=False)

        with open(os.path.join(tmpdir, "dummy.encodings")) as json_file:
            encoding_data = json.load(json_file)

        param_names = {encoding["name"] for encoding in encoding_data}
        assert "weight" in param_names

    @pytest.mark.parametrize(
        "model, input_shape",
        [
            (models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),
        ],
    )
    def test_adaround_with_dict_input(self, model, input_shape, tmpdir):
        class DictDataLoader:
            """
            Example of a Dataloader which can be used for running AMPv2
            """

            def __init__(self, input_shape: tuple, input_name):
                """
                :param batch_size: batch size for data loader
                """
                self.input_shape = input_shape
                self.input_name = input_name

            def __iter__(self):
                """Iterates over dataset"""
                dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
                yield {self.input_name: dummy_input}

            def __len__(self):
                return 4

        data_loader = DictDataLoader((1, 10), "input")

        def callback(session, args):
            in_tensor = {"input": np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        Adaround.apply_adaround(model, params, tmpdir, "dummy", use_cuda=False)

        with open(os.path.join(tmpdir, "dummy.encodings")) as json_file:
            encoding_data = json.load(json_file)

        param_names = {encoding["name"] for encoding in encoding_data}
        assert "weight" in param_names

    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.dynamic_matmul_model(1), (1, 10))]
    )
    def test_adaround_dynamic_matmul(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out if there is a dynamic matmul
        """
        data_loader = dataloader(input_shape, input_shape[0])

        def callback(session, args):
            in_tensor = {"input": np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        Adaround.apply_adaround(model, params, tmpdir, "dummy", use_cuda=False)

    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.simplifiable_model(1), (1, 10))]
    )
    def test_adaround_simplifiable_model(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out for models which need simplification
        """
        data_loader = dataloader(input_shape, input_shape[0])

        def callback(session, args):
            in_tensor = {"input": np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        model, _ = simplify(model)
        Adaround.apply_adaround(model, params, tmpdir, "dummy", use_cuda=False)

    @pytest.mark.parametrize(
        "model_factory, input_shape",
        [
            (models_for_tests.pointwise_conv1d, (1, 10, 32)),
            (models_for_tests.pointwise_conv3d, (1, 10, 8, 8, 8)),
            (models_for_tests.pointwise_convtranspose1d, (1, 10, 32)),
            (models_for_tests.pointwise_convtranspose3d, (1, 10, 8, 4, 3)),
            (models_for_tests.padded_convtranspose2d, (1, 10, 32, 32)),
        ],
    )
    def test_adaround_convNd_model(self, model_factory, input_shape, tmpdir):
        """
        AdaRound should not error-out for non-2d Conv/ConvTranspose layers
        """
        model = model_factory(input_shape)
        data_loader = dataloader(input_shape, input_shape[0])

        def callback(session, args):
            in_tensor = {"input": np.random.rand(*input_shape).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )

        Adaround.apply_adaround(model, params, tmpdir, "dummy", use_cuda=False)

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround_per_channel(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        np.random.seed(0)
        torch.manual_seed(0)
        model = models_for_tests.single_residual_model()
        data_loader = dataloader(input_shape=(1, 3, 32, 32))
        dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
        sess = build_session(model.model, providers)
        out_before_ada = sess.run(None, dummy_input)

        def callback(session, args):
            in_tensor = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}
            session.run(None, in_tensor)

        params = AdaroundParameters(
            data_loader=data_loader,
            num_batches=1,
            default_num_iterations=5,
            forward_fn=callback,
            forward_pass_callback_args=None,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            ada_rounded_model = Adaround.apply_adaround(
                model,
                params,
                tempdir,
                "dummy",
                use_cuda="CUDAExecutionProvider" in providers,
                default_config_file=get_path_for_per_channel_config(),
            )
            sess = build_session(ada_rounded_model.model, providers)
            out_after_ada = sess.run(None, dummy_input)
            assert not np.array_equal(out_before_ada[0], out_after_ada[0])

            with open(os.path.join(tempdir, "dummy.encodings")) as json_file:
                encoding_data = json.load(json_file)

                param_encodings = {
                    encoding["name"]: encoding for encoding in encoding_data
                }
                assert (
                    len(param_encodings["conv3.weight"]["scale"]) == 8
                )  # out_channels
                assert (
                    len(param_encodings["conv4.weight"]["scale"]) == 8
                )  # out_channels

    @pytest.mark.parametrize(
        "config",
        [
            # model, ops_to_optimize input to pass, expected result
            (
                models_for_tests.single_residual_model().model,
                [],
                [
                    "/conv1/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                    "/conv4/Conv",
                    "/fc/Gemm",
                ],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv1/Conv"],
                ["/conv1/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv2/Conv"],
                ["/conv2/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                [
                    "/conv1/Conv",
                    "/conv4/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                ],
                ["/conv1/Conv", "/conv4/Conv", "/conv2/Conv", "/conv3/Conv"],
            ),
        ],
    )
    def test_whitelist_functionality(self, config):
        model, whitelist_ops, expected = config
        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
        )

        param_to_op_name_dict = {}
        for cg_op in sim.connected_graph.get_all_ops().values():
            if cg_op.type in AdaroundSupportedModules:
                param_to_op_name_dict[cg_op.inputs[1].name] = cg_op.name

        ops_processed = []

        def mock_adaround_module(module, *args, **kwargs):
            ops_processed.append(param_to_op_name_dict[module.params["weight"].name])

        with patch(
            "aimet_onnx.adaround.adaround_optimizer.AdaroundOptimizer.adaround_module",
            mock_adaround_module,
        ):
            apply_adaround(
                sim, inputs, num_iterations=5, node_names_to_optimize=whitelist_ops
            )

            print([name for name in ops_processed])
            assert ops_processed.sort() == expected.sort()


def dataloader(input_shape: tuple, batch_size=2):
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """

        def __init__(self, batch_size: int, input_shape: tuple):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size
            self.input_shape = input_shape

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=batch_size, input_shape=input_shape)
    return dummy_dataloader
