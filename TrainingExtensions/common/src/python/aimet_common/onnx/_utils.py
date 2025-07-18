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
"""Collection of onnx-related util functions that can be shared across aimet-onnx and aimet-torch"""

# pylint: disable=no-member

from collections import deque
from typing import Iterable, Optional, Sequence, Dict, List

import os
import tempfile

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto
from onnx.numpy_helper import from_array, to_array

from aimet_common.onnx import opset10, opset13, opset21
from aimet_common.utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


def _add_onnx_qdq_node(
    model: ModelProto,
    input_name: str,
    output_name: str,
    node_name_prefix: str,
    encodings: dict,
    onnx_opset: int,
    prequantize_constants: bool,
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encoding, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    _add_onnx_qdq_nodes(
        model,
        [input_name],
        [output_name],
        [node_name_prefix],
        [encodings],
        onnx_opset,
        prequantize_constants,
    )


def _add_onnx_qdq_nodes(
    model: ModelProto,
    input_names: Iterable[str],
    output_names: Iterable[str],
    node_name_prefixes: Iterable[str],
    encodings: Iterable[dict],
    onnx_opset: int,
    prequantize_constants: bool,
):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_q)                             (output)


    except for int32 bias encodings, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_q)                             (bias_qdq)

    """
    if onnx_opset < 10:
        raise RuntimeError(
            "ONNX opset {} cannot represent QuantizeLinear and DequantizeLinear nodes."
            "So not able to export model as ONNX QDQ graph"
        )

    if onnx_opset < 13:
        opset = opset10
    elif onnx_opset < 21:
        opset = opset13
    else:
        opset = opset21

    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding in zip(
        input_names, output_names, node_name_prefixes, encodings
    ):
        inputs_to_rename[input_name] = output_name
        output_dtype = encoding["output_dtype"]
        axis = encoding.get("axis", None)
        block_size = encoding.get("block_size", None)
        y_zero_point = encoding.get("y_zero_point", None)

        y_scale = np.array(
            encoding.get("y_scale") or encoding.get("per_channel_float_scale")
        ).astype(np.float32)
        per_block_int_scale = (
            np.array(encoding["per_block_int_scale"])
            if "per_block_int_scale" in encoding
            else None
        )

        if y_zero_point is not None:
            y_zero_point = np.array(encoding["y_zero_point"], dtype=np.int64)
        elif per_block_int_scale is not None:
            y_zero_point = np.zeros(per_block_int_scale.shape, dtype=np.int64)
        else:
            y_zero_point = np.zeros(y_scale.shape, dtype=np.int64)

        tensors_to_add.append(
            opset.DequantizeLinear.make_zero_point(
                y_zero_point, dtype=output_dtype, name=f"{input_name}_zero_point"
            )
        )

        if per_block_int_scale is None:
            tensors_to_add.append(from_array(y_scale, name=f"{input_name}_scale"))
        else:
            # Export LPBQ.
            #
            # Strategy: Derive y_scale from per_channel_float_scale and per_block_uint_scale
            #
            #           (FLOAT)
            # per_channel_float_scale -----+
            #                              +--> DequantizeLinear -----+ (blockwise scale)
            #    per_block_uint_scale -----+                          |
            #           (UINT8)                               +-------+---------+
            #                                                 V                 V
            #                              weight ---> QuantizeLinear -> DequantizeLinear -> ...
            if output_dtype != "int4":
                raise RuntimeError(
                    f"LPBQ can be only exported with int4; got {output_dtype}"
                )

            try:
                weight_dims = next(
                    init for init in model.graph.initializer if init.name == input_name
                ).dims
            except StopIteration:
                weight_dims = None

            if weight_dims is not None and len(weight_dims) != 2:
                raise RuntimeError(
                    "LPBQ can be only applied to 2D matrices when exported to onnx QDQ. "
                    f'Got "{input_name}" with shape {weight_dims}'
                )

            if axis not in (-2, -1, 0, 1):
                raise RuntimeError(
                    "LPBQ can be only applied to 2D matrices when exported to onnx QDQ. "
                    f"Got axis {axis}"
                )

            tensors_to_add.extend(
                [
                    from_array(
                        y_scale.flatten(), name=f"{input_name}_per_channel_float_scale"
                    ),
                    from_array(
                        per_block_int_scale.astype(np.uint8),
                        name=f"{input_name}_per_block_uint_scale",
                    ),
                ]
            )
            nodes_to_add.extend(
                [
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_scale_dq",
                        inputs=[
                            f"{input_name}_per_block_uint_scale",
                            f"{input_name}_per_channel_float_scale",
                        ],
                        output=f"{input_name}_scale",
                        dtype="uint8",
                        axis=0 if axis in (-1, 1) else 1,  # == channel axis
                    )
                ]
            )

        input_q = None
        if prequantize_constants or output_dtype in ("int32", "uint32"):
            input_q = _quantize_const(
                model,
                input_name,
                y_scale,
                y_zero_point,
                axis,
                block_size,
                output_dtype,
            )

        if input_q:
            nodes_to_add.append(
                opset.DequantizeLinear.make_node(
                    name=f"{node_name_prefix}_dq",
                    inputs=[
                        input_q.name,
                        f"{input_name}_scale",
                        f"{input_name}_zero_point",
                    ],
                    output=output_name,
                    dtype=output_dtype,
                    axis=axis,
                    block_size=block_size,
                )
            )
            tensors_to_remove[input_name] = True
            tensors_to_add.append(input_q)

        else:
            nodes_to_add.extend(
                [
                    opset.QuantizeLinear.make_node(
                        name=f"{node_name_prefix}_q",
                        inputs=[
                            input_name,
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=f"{input_name}_q",
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                    opset.DequantizeLinear.make_node(
                        name=f"{node_name_prefix}_dq",
                        inputs=[
                            f"{input_name}_q",
                            f"{input_name}_scale",
                            f"{input_name}_zero_point",
                        ],
                        output=output_name,
                        dtype=output_dtype,
                        axis=axis,
                        block_size=block_size,
                    ),
                ]
            )

    _finalize_graph_changes(
        model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove
    )


def _quantize_const(
    model: ModelProto,
    name: str,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    axis: Optional[int],
    block_size: Optional[int],
    output_dtype: str,
) -> Optional[TensorProto]:
    const = _ParamUtils.get_param_by_name(model, name)

    if not const:
        return None

    const = to_array(const).astype(np.float32)
    unsigned, bitwidth = output_dtype.split("int")
    bitwidth = int(bitwidth)

    if unsigned:
        clip_min = 0
        clip_max = 2**bitwidth - 1
    else:
        clip_min = -(2 ** (bitwidth - 1))
        clip_max = -clip_min - 1

    if axis is not None:
        axis = (const.ndim + axis) % const.ndim  # Make positive
        if block_size is None:
            channel_axis = axis
            broadcast_shape = tuple(
                -1 if axis == channel_axis else 1 for axis in range(const.ndim)
            )
            y_scale = y_scale.reshape(broadcast_shape)
            y_zero_point = y_zero_point.reshape(broadcast_shape)
        else:
            block_axis = axis
            y_scale = y_scale.repeat(block_size, axis=block_axis)
            y_zero_point = y_zero_point.repeat(block_size, axis=block_axis)

    y_scale = y_scale.astype(np.float32)
    const_q = (const / y_scale + y_zero_point).round()
    const_q = const_q.clip(clip_min, clip_max)
    return opset10.DequantizeLinear.make_int_arr(
        const_q, dtype=output_dtype, name=f"{name}_q"
    )


def _finalize_graph_changes(
    model: ModelProto,
    nodes_to_add: Iterable,
    inputs_to_rename: Dict,
    tensors_to_add: List[TensorProto],
    tensors_to_remove: Dict,
):
    # Remove dangling tensors/nodes
    initializers = [
        init
        for init in model.graph.initializer
        if not tensors_to_remove.pop(init.name, None)
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)

    nodes = [
        node
        for node in model.graph.node
        if not (
            node.op_type == "Constant" and tensors_to_remove.pop(node.output[0], None)
        )
    ]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)

    # Redirect consumers that took the removed biases to take qdq bias instead
    # before:
    #     bias --------------------> consumer
    # after:
    #     bias_int32 --> DQ -------> consumer
    for node in model.graph.node:
        for i, old_name in enumerate(node.input):
            new_name = inputs_to_rename.get(old_name, None)
            if new_name is not None:
                node.input[i] = new_name

    # Add new tensors
    for t in tensors_to_add:
        model.graph.initializer.append(t)

    # Insert new nodes in a topologically order
    original_nodes = deque(list(model.graph.node))
    new_nodes = {node.input[0]: node for node in nodes_to_add if node.input}
    queue = deque([node for node in nodes_to_add if not node.input])

    queue.extend(
        [new_nodes.pop(inp.name) for inp in model.graph.input if inp.name in new_nodes]
    )
    queue.extend(
        [
            new_nodes.pop(init.name)
            for init in model.graph.initializer
            if init.name in new_nodes
        ]
    )

    if not queue and original_nodes:
        queue.append(original_nodes.popleft())

    model.graph.ClearField("node")

    while queue:
        node = queue.popleft()
        model.graph.node.append(node)

        qdq_nodes = [
            new_nodes.pop(output_name)
            for output_name in node.output
            if output_name in new_nodes
        ]
        if qdq_nodes:
            queue.extend(qdq_nodes)

        if not queue and original_nodes:
            queue.append(original_nodes.popleft())

    model.graph.node.extend(new_nodes.values())


class _ParamUtils:
    """Param utilities"""

    @staticmethod
    def get_shape(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[Sequence[int]]:
        """
        Returns a list of shape for the param specifies
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        param = _ParamUtils.get_param(model, node, param_index)
        if param:
            return param.dims
        return None

    @staticmethod
    def get_param(
        model: ModelProto, node: NodeProto, param_index: int
    ) -> Optional[TensorProto]:
        """
        Returns the param tensor
        :param model: ONNX model
        :param node: ONNX node to which the param feeds to
        :param param_index: Index at which param feeds to the ONNX node
        """
        if len(node.input) >= param_index + 1:
            param_name = node.input[param_index]
            return _ParamUtils.get_param_by_name(model, param_name)
        return None

    @staticmethod
    def get_param_by_name(model: ModelProto, param_name: str) -> Optional[TensorProto]:
        """
        Returns the param tensor

        :param model: ONNX model
        :param param_name: Name of parameter to retrieve
        """

        def find_param_in_model_initializers(param_name: str, model: ModelProto):
            for param in model.graph.initializer:
                if param.name == param_name:
                    return param
            return None

        def find_param_in_model_constants(param_name: str, model: ModelProto):
            for node in model.graph.node:
                if node.op_type == "Constant" and param_name in node.output:
                    for attribute in node.attribute:
                        if attribute.name == "value":
                            param = attribute.t
                            param.name = param_name
                            return param
                if node.op_type == "Identity" and param_name == node.output[0]:
                    return _ParamUtils.get_param(model, node, 0)
            return None

        param = find_param_in_model_initializers(param_name, model)
        if param is None:
            param = find_param_in_model_constants(param_name, model)
        return param


_all_op_schemas = {schema.name: schema for schema in onnx.defs.get_all_schemas()}


def _is_float_output(op_type: str) -> bool:
    """
    Returns True if op_type can return float output
    """
    schema = _all_op_schemas[op_type]
    type_str = schema.outputs[0].type_str
    type_constraint = next(
        type_constraint
        for type_constraint in schema.type_constraints
        if type_constraint.type_param_str == type_str
    )
    return any(
        t in ("tensor(float)", "tensor(double)", "tensor(float16)")
        for t in type_constraint.allowed_type_strs
    )


def _is_grid_preserving_op(op_type: str) -> bool:
    """
    Returns True if op_type can be considered a grid-preserving op.
    Data movement op is defined as a reshape or indexing operator
    whose output strictly preserves the quantization grid of the input

    Formally put,
    function `f` is a grid-preserving op if and only if y == y'
    where
        * x_q = quantize(x, scale_x, zp_x)
        * y  = f(x_q)
        * y' = requantize(y, scale_x, zp_x)
    """
    return op_type in (
        "BatchToSpace",
        "Col2Im",
        "Compress",
        "DepthToSpace",
        "Dropout",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "MaxPool",
        "MaxRoiPool",
        "NonZero",
        "Pad",
        "ReduceMax",
        "ReduceMin",
        "Reshape",
        "Slice",
        "SpaceToBatch",
        "SpaceToDepth",
        "Split",
        "SplitToSequence",
        "Squeeze",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
    )


def _is_htp_interpolation_op(op_type: str) -> bool:
    """
    Returns True if op_type can be considered an interpolation op in HTP.
    Although these operators aren't strictly data movement ops,
    HTP reuses the same quantization encoding for both input and output of
    the interpolation ops
    """
    # TODO: Absorb this function into redesigned config file
    return op_type in (
        "CropAndResize",
        "Resize",
        "ScatterElements",
        "Upsample",
    )


def _convert_version(
    model: onnx.ModelProto, target_opset_version: int
) -> onnx.ModelProto:
    try:
        model = onnx.version_converter.convert_version(model, target_opset_version)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # convert_version throws an exception on model > 2GB the observed exception was a
        # RuntimeError exception about ir_version, but possible other exceptions could be
        # triggerd. leaving this very generic for now.
        logger.warning(
            "onnx.version_converter.convert_version failed with exception: %s. Retrying with external data",
            str(e),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_file = os.path.join(tmp_dir, "model.onnx")
            onnx.save_model(
                model,
                onnx_file,
                save_as_external_data=True,
                location="model.data",
            )
            model = onnx.load_model(onnx_file, load_external_data=False)
            model = onnx.version_converter.convert_version(model, target_opset_version)
            onnx.external_data_helper.load_external_data_for_model(model, tmp_dir)

    logger.info("The opset of the onnx model is updated to %s.", target_opset_version)
    return model
