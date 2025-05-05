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
import numpy as np
from onnx import helper, ModelProto, NodeProto, TensorProto
from onnx.numpy_helper import from_array, to_array


def _add_onnx_qdq_node(model: ModelProto,
                       input_name: str,
                       output_name: str,
                       node_name_prefix: str,
                       encodings: dict,
                       onnx_opset: int):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_int)                           (output)


    except for int32 bias encoding, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_int)                           (bias_qdq)

    """
    _add_onnx_qdq_nodes(model, [input_name], [output_name], [node_name_prefix], [encodings], onnx_opset)


def _add_onnx_qdq_nodes(model: ModelProto,
                        input_names: Iterable[str],
                        output_names: Iterable[str],
                        node_name_prefixes: Iterable[str],
                        encodings: Iterable[dict],
                        onnx_opset: int):
    """
    Add onnx::QuantizeLinear and/or onnx::DequantizeLinear as below

     -------> onnx::QuantizeLinear -------------> onnx::DequantizeLinear ----->
    (input)                        (input_int)                           (output)


    except for int32 bias encodings, for which we take alternative representation as below
    since onnx::QuantizeLinear doesn't allow int32 outputs.

    -------------> onnx::DequantizeLinear ----->
    (bias_int)                           (bias_qdq)

    """
    if onnx_opset < 10:
        raise RuntimeError('ONNX opset {} cannot represent QuantizeLinear and DequantizeLinear nodes.'
                           'So not able to export model as ONNX QDQ graph')

    if onnx_opset < 13:
        _add_onnx_qdq_nodes_opset_10(model, input_names, output_names, node_name_prefixes, encodings)

    elif onnx_opset < 21:
        _add_onnx_qdq_nodes_opset_13(model, input_names, output_names, node_name_prefixes, encodings)

    else:
        _add_onnx_qdq_nodes_opset_21(model, input_names, output_names, node_name_prefixes, encodings)


def _replace_bias_with_quantized_bias(model: ModelProto,
                                      bias_name: str,
                                      y_scale: np.array,
                                      output_dtype: str,
                                      tensors_to_add: List[TensorProto],
                                      tensors_to_remove: Dict):

    bias = _ParamUtils.get_param_by_name(model, bias_name)
    tensors_to_remove[bias_name] = True

    bias_int32 = (to_array(bias) / y_scale).round()
    if output_dtype == "int32":
        bias_int32 = bias_int32.clip(-2 ** 31, 2 ** 31 - 1)
    else:
        bias_int32 = bias_int32.clip(0, 2 ** 32 - 1)

    tensors_to_add.append(from_array(bias_int32.astype(output_dtype), name=f"{bias_name}_int"))


def _finalize_graph_changes(model: ModelProto,
                            nodes_to_add: Iterable,
                            inputs_to_rename: Dict,
                            tensors_to_add: List[TensorProto],
                            tensors_to_remove: Dict):
    # Remove dangling tensors/nodes
    initializers = [
        init for init in model.graph.initializer
        if not tensors_to_remove.pop(init.name, None)
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)

    nodes = [
        node for node in model.graph.node
        if not (node.op_type == "Constant" and tensors_to_remove.pop(node.output[0], None))
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
    new_nodes = {
        node.input[0]: node for node in nodes_to_add
    }
    queue = deque([])

    queue.extend([
        new_nodes.pop(inp.name)
        for inp in model.graph.input
        if inp.name in new_nodes
    ])
    queue.extend([
        new_nodes.pop(init.name)
        for init in model.graph.initializer
        if init.name in new_nodes
    ])

    if not queue and original_nodes:
        queue.append(original_nodes.popleft())

    model.graph.ClearField("node")

    while queue:
        node = queue.popleft()
        model.graph.node.append(node)

        qdq_nodes = [
            new_nodes.pop(output_name) for output_name in node.output
            if output_name in new_nodes
        ]
        if qdq_nodes:
            queue.extend(qdq_nodes)

        if not queue and original_nodes:
            queue.append(original_nodes.popleft())

    model.graph.node.extend(new_nodes.values())

def _add_onnx_qdq_nodes_opset_10(model: ModelProto,
                                 input_names: Iterable[str],
                                 output_names: Iterable[str],
                                 node_name_prefixes: Iterable[str],
                                 encodings: Iterable[dict]):
    """
    Specialized code for creating QDQ nodes for Opset 10
    Special rules for Opset 10
    - Only support for INT8 and UINT8 precision
    - No support for per-channel quantization
    """
    # pylint: disable=too-many-branches, too-many-locals
    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding in zip(input_names, output_names, node_name_prefixes,
                                                                   encodings):

        inputs_to_rename[input_name] = output_name

        output_dtype = encoding["output_dtype"]
        dtype_map_to_onnx = {'int8': TensorProto.INT8,
                             'uint8': TensorProto.UINT8,
                             'int32': TensorProto.INT32,
                            }

        if output_dtype not in dtype_map_to_onnx:
            raise RuntimeError("ONNX QDQ in opset 10 cannot represent {} precision. "
                               "Hence cannot export to ONNX QDQ".format(output_dtype))

        block_size = encoding.get("block_size", None)

        y_scale = np.array(encoding["y_scale"]).astype(np.float32)
        if encoding.get("y_zero_point", None):
            y_zero_point = np.array(encoding["y_zero_point"]).astype(output_dtype)
        else:
            y_zero_point = np.zeros(y_scale.shape).astype(output_dtype)

        if output_dtype in ("int32", "uint32"):
            new_node = helper.make_node('DequantizeLinear',
                                        name=f"{node_name_prefix}_dq",
                                        inputs=[f"{input_name}_int", f"{input_name}_scale",
                                                f"{input_name}_zero_point"],
                                        outputs=[output_name],
                                        block_size=block_size)

            nodes_to_add.append(new_node)
            tensors_to_add.extend([
                from_array(y_scale, name=f"{input_name}_scale"),
                from_array(y_zero_point, name=f"{input_name}_zero_point"),
            ])

            _replace_bias_with_quantized_bias(model, input_name, y_scale, output_dtype, tensors_to_add, tensors_to_remove)

        else:

            q_node = helper.make_node('QuantizeLinear',
                                      name=f"{node_name_prefix}_q",
                                      inputs=[input_name, f"{input_name}_scale", f"{input_name}_zero_point"],
                                      outputs=[f"{input_name}_int"],
                                      block_size=block_size)

            dq_node = helper.make_node('DequantizeLinear',
                                       name=f"{node_name_prefix}_dq",
                                       inputs=[f"{input_name}_int", f"{input_name}_scale", f"{input_name}_zero_point"],
                                       outputs=[output_name],
                                       block_size=block_size)

            nodes_to_add.extend([q_node, dq_node])
            tensors_to_add.extend([
                from_array(y_scale, name=f"{input_name}_scale"),
                from_array(y_zero_point, name=f"{input_name}_zero_point"),
            ])

    _finalize_graph_changes(model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove)



def _add_onnx_qdq_nodes_opset_13(model: ModelProto,
                                 input_names: Iterable[str],
                                 output_names: Iterable[str],
                                 node_name_prefixes: Iterable[str],
                                 encodings: Iterable[dict]):
    """
    Specialized code for creating QDQ nodes for Opset 13
    Special rules for Opset 13
    - QuantizeLinear and DequantizeLinear have an axis field for per-channel quantization
    - Only support for INT8 and UINT8 precision
    """

    # pylint: disable=too-many-branches, too-many-locals
    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding in zip(input_names, output_names, node_name_prefixes,
                                                                   encodings):

        inputs_to_rename[input_name] = output_name

        output_dtype = encoding["output_dtype"]
        dtype_map_to_onnx = {'int8': TensorProto.INT8,
                             'uint8': TensorProto.UINT8,
                             'int32': TensorProto.INT32,
                             }

        if output_dtype not in dtype_map_to_onnx:
            raise RuntimeError("ONNX QDQ in opset 13 cannot represent {} precision. "
                               "Hence cannot export to ONNX QDQ".format(output_dtype))

        axis = encoding.get("axis", None)
        block_size = encoding.get("block_size", None)

        y_scale = np.array(encoding["y_scale"]).astype(np.float32)
        if encoding.get("y_zero_point", None):
            y_zero_point = np.array(encoding["y_zero_point"]).astype(output_dtype)
        else:
            y_zero_point = np.zeros(y_scale.shape).astype(output_dtype)

        if output_dtype in ("int32", "uint32"):
            new_node = helper.make_node('DequantizeLinear',
                                        name=f"{node_name_prefix}_dq",
                                        inputs=[f"{input_name}_int", f"{input_name}_scale",
                                                f"{input_name}_zero_point"],
                                        outputs=[output_name],
                                        axis=axis,
                                        block_size=block_size)

            nodes_to_add.append(new_node)
            tensors_to_add.extend([
                from_array(y_scale, name=f"{input_name}_scale"),
                from_array(y_zero_point, name=f"{input_name}_zero_point"),
            ])

            _replace_bias_with_quantized_bias(model, input_name, y_scale, output_dtype, tensors_to_add, tensors_to_remove)

        else:

            q_node = helper.make_node('QuantizeLinear',
                                      name=f"{node_name_prefix}_q",
                                      inputs=[input_name, f"{input_name}_scale", f"{input_name}_zero_point"],
                                      outputs=[f"{input_name}_int"],
                                      axis=axis,
                                      block_size=block_size)

            dq_node = helper.make_node('DequantizeLinear',
                                       name=f"{node_name_prefix}_dq",
                                       inputs=[f"{input_name}_int", f"{input_name}_scale", f"{input_name}_zero_point"],
                                       outputs=[output_name],
                                       axis=axis,
                                       block_size=block_size)

            nodes_to_add.extend([q_node, dq_node])
            tensors_to_add.extend([
                from_array(y_scale, name=f"{input_name}_scale"),
                from_array(y_zero_point, name=f"{input_name}_zero_point"),
            ])

    _finalize_graph_changes(model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove)


def _add_onnx_qdq_nodes_opset_21(model: ModelProto,
                                 input_names: Iterable[str],
                                 output_names: Iterable[str],
                                 node_name_prefixes: Iterable[str],
                                 encodings: Iterable[dict]):
    """
    Specialized code for creating QDQ nodes for Opset 21
    Special rules for Opset 21
    - QuantizeLinear.output_dtype set to quantized data type
    - QuantizeLinear.y_zero_point data type set to quantized data type
    - Support for INT8, INT16 and INT4 precisions
    - Special handling for y_zero_point 4 bits - we are currently not supporting asymmetric 4-bit quantization
    """
    # pylint: disable=too-many-branches, too-many-locals
    nodes_to_add = []
    tensors_to_add = []
    tensors_to_remove = {}
    inputs_to_rename = {}

    for input_name, output_name, node_name_prefix, encoding in zip(input_names, output_names, node_name_prefixes,
                                                                   encodings):

        inputs_to_rename[input_name] = output_name

        output_dtype = encoding["output_dtype"]
        dtype_map_to_onnx = {'int4': TensorProto.INT4,
                             'uint4': TensorProto.UINT4,
                             'int8': TensorProto.INT8,
                             'uint8': TensorProto.UINT8,
                             'int16': TensorProto.INT16,
                             'uint16': TensorProto.UINT16,
                             'int32': TensorProto.INT32,
                            }

        if output_dtype not in dtype_map_to_onnx:
            raise RuntimeError("ONNX QDQ in opset 21 cannot represent {} precision. "
                               "Hence cannot export to ONNX QDQ".format(output_dtype))

        output_dtype_onnx = dtype_map_to_onnx[output_dtype]

        axis = encoding.get("axis", None)
        block_size = encoding.get("block_size", None)

        y_scale = np.array(encoding["y_scale"]).astype(np.float32)
        tensors_to_add.append(from_array(y_scale, name=f"{input_name}_scale"))

        # No zero point specified if not needed - since output_dtype indicates quantized data_type
        y_zero_point = encoding.get("y_zero_point", None)
        if y_zero_point:
            if output_dtype in ("int4", "uint4"):
                raise NotImplementedError("No support for asymmetric 4-bit quantization when exporting to QDQ")
            else:
                y_zero_point = np.array(encoding["y_zero_point"]).astype(output_dtype)
                tensors_to_add.append(from_array(y_zero_point, name=f"{input_name}_zero_point"))


        if output_dtype in ("int32", "uint32"):
            params = {'name':     f"{node_name_prefix}_dq",
                      'outputs': [output_name],
                      'axis': axis,
                      'block_size': block_size}

            if y_zero_point:
                params['inputs'] = [f"{input_name}_int", f"{input_name}_scale", f"{input_name}_zero_point"]
            else:
                params['inputs'] = [f"{input_name}_int", f"{input_name}_scale"]

            new_node = helper.make_node('DequantizeLinear',
                                        **params)

            nodes_to_add.append(new_node)

            _replace_bias_with_quantized_bias(model, input_name, y_scale, output_dtype, tensors_to_add, tensors_to_remove)

        else:
            params = {'name':     f"{node_name_prefix}_q",
                      'outputs': [f"{input_name}_int"],
                      'output_dtype': output_dtype_onnx,
                      'axis': axis,
                      'block_size': block_size}
            if y_zero_point:
                params['inputs'] = [input_name, f"{input_name}_scale", f"{input_name}_zero_point"]
            else:
                params['inputs'] = [input_name, f"{input_name}_scale"]

            q_node = helper.make_node('QuantizeLinear',
                                      **params)

            params = {'name':     f"{node_name_prefix}_dq",
                      'outputs': [output_name],
                      'axis': axis,
                      'block_size': block_size}
            if y_zero_point:
                params['inputs'] = [f"{input_name}_int", f"{input_name}_scale", f"{input_name}_zero_point"]
            else:
                params['inputs'] = [f"{input_name}_int", f"{input_name}_scale"]

            dq_node = helper.make_node('DequantizeLinear',
                                       **params)

            nodes_to_add.extend([q_node, dq_node])

    _finalize_graph_changes(model, nodes_to_add, inputs_to_rename, tensors_to_add, tensors_to_remove)


class _ParamUtils:
    """ Param utilities """

    @staticmethod
    def get_shape(model: ModelProto, node: NodeProto, param_index: int) -> Optional[Sequence[int]]:
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
    def get_param(model: ModelProto, node: NodeProto, param_index: int) -> Optional[TensorProto]:
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
                if node.op_type == 'Constant' and param_name in node.output:
                    for attribute in node.attribute:
                        if attribute.name == 'value':
                            param = attribute.t
                            param.name = param_name
                            return param
                if node.op_type == 'Identity' and param_name == node.output[0]:
                    return _ParamUtils.get_param(model, node, 0)
            return None

        param = find_param_in_model_initializers(param_name, model)
        if param is None:
            param = find_param_in_model_constants(param_name, model)
        return param
