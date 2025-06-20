# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import tempfile
import math
from functools import partial

import numpy as np
import onnxruntime

from aimet_common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import analyze_per_layer_sensitivity
from aimet_onnx import int8, int16, float16
from aimet_onnx.utils import make_dummy_input
from aimet_onnx.lite_mp import flip_layers_to_higher_precision

from .models import models_for_tests


def _compute_snr(expected: np.array, actual: np.array):
    """
    Computes the SNR for two signals where the noise is defined as expected - actual
    """
    data_range = np.abs(expected).max()
    noise_pw = np.sum(np.power(expected - actual, 2))
    noise_pw /= actual.size
    noise = np.sqrt(noise_pw)
    noise = max(noise, 1e-10)
    return 20 * np.log10(data_range / noise)


def _collect_inputs_and_fp_outputs(model):
    model_bytes = model.SerializeToString()
    fp_session = onnxruntime.InferenceSession(
        model_bytes, providers=["CUDAExecutionProvider"]
    )

    fp_inputs = np.random.randn(1, 3, 32, 32).astype(np.float32)
    fp_outputs = fp_session.run(None, {"input": fp_inputs})

    return fp_inputs, fp_outputs


def _eval_accuracy(session, args):
    fp_inputs, fp_outputs = args
    quantized_outputs = session.run(None, {"input": fp_inputs})
    snr = _compute_snr(fp_outputs[0], quantized_outputs[0])

    return snr if not math.isnan(snr) else 0.0


class TestLiteMp:
    @pytest.mark.parametrize("percent_flip", [30, 50, 80])
    def test_flip_to_float(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_inputs, fp_outputs = _collect_inputs_and_fp_outputs(model)
        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        with tempfile.TemporaryDirectory() as tempdir:
            layer_sensitivity_dict = analyze_per_layer_sensitivity(
                sim, eval_fn=partial(_eval_accuracy, args=(fp_inputs, fp_outputs))
            )
            flip_layers_to_higher_precision(
                sim, layer_sensitivity_dict, percent_flip, override_precision=float16
            )

        new_int8_count = sum(1 for op in q_ops if op.bitwidth == 8 and op.enabled)
        assert new_int8_count < int8_count
        assert new_int8_count <= (100 - percent_flip) / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])

    @pytest.mark.parametrize("percent_flip", [30, 50, 80])
    def test_flip_to_int16(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_inputs, fp_outputs = _collect_inputs_and_fp_outputs(model)
        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        with tempfile.TemporaryDirectory() as tempdir:
            layer_sensitivity_dict = analyze_per_layer_sensitivity(
                sim, eval_fn=partial(_eval_accuracy, args=(fp_inputs, fp_outputs))
            )
            flip_layers_to_higher_precision(
                sim, layer_sensitivity_dict, percent_flip, override_precision=int16
            )

        int16_count = sum(1 for op in q_ops if op.bitwidth == 16 and op.enabled)
        assert int16_count >= percent_flip / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])
