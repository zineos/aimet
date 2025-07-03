# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for exporting models from ONNX to Torch"""

import os
import torch
import onnx
import glob


def get_onnx_model(
    checkpoint: str | os.PathLike,
    fp_model: torch.nn.Module,
    context_length: int,
    sample_input: tuple[torch.Tensor],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> onnx.ModelProto:
    # Create the checkpoint directory if it does not exist.
    os.makedirs(checkpoint, exist_ok=True)
    onnx_model_path = os.path.join(checkpoint, f"model_cl{context_length}.onnx")

    fp_model.eval()
    fp_model.train(False)

    if not os.path.exists(onnx_model_path):
        print("Exporting model to ONNX...")
        fp_model.to(torch.device("cpu"))

        with torch.no_grad():
            torch.onnx.export(
                fp_model,
                sample_input,
                onnx_model_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
            )

        print("Loading ONNX model...")
        model = onnx.load(onnx_model_path)

        # Clean up multiple weights files
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "*.weight")
        ):
            os.remove(file)
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "onnx__*")
        ):
            os.remove(file)
        for file in glob.glob(
            os.path.join(os.path.dirname(onnx_model_path), "*__value")
        ):
            os.remove(file)

        onnx.save_model(
            model,
            onnx_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

        onnx.external_data_helper.load_external_data_for_model(
            model, os.path.dirname(onnx_model_path)
        )
    else:
        print("Loading cached ONNX model...")
        model = onnx.load(onnx_model_path)

    return model
