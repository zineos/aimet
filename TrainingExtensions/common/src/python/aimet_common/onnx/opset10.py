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
# pylint: disable=no-member
from abc import ABC, abstractmethod
import sys
from typing import Iterable, Mapping, Optional
import numpy as np
from onnx import helper, numpy_helper, TensorProto

class _QdqNodeFactory(ABC):
    OPSET: int
    SUPPORTED_DTYPES: Mapping[str, "TensorProto.DataType"]

    @classmethod
    @abstractmethod
    def make_node(cls, name: str, inputs: Iterable[str], output: str,
                  dtype: str, axis: Optional[int] = None,
                  block_size: Optional[int] = None):
        ...

    @classmethod
    def _check_dtype(cls, dtype: str):
        if dtype in cls.SUPPORTED_DTYPES:
            return

        raise RuntimeError(
            f"Unsupported dtype {dtype}; "
            f"opset {cls.OPSET} expects one of {list(cls.SUPPORTED_DTYPES.keys())}"
        )

    @classmethod
    def make_zero_point(cls, zero_point: np.ndarray, dtype: str, name: str):
        cls._check_dtype(dtype)

        if (dtype == "int32" or dtype.startswith("float")) and not np.all(zero_point == 0):
            raise RuntimeError(
                "DequantizeLinear with type int32 or float8 should have "
                "no zero point or all zero points should be 0"
            )

        if dtype not in ("int4", "uint4"):
            zero_point = zero_point.astype(dtype)
            return numpy_helper.from_array(zero_point, name=name)

        target_shape = zero_point.shape

        # Numpy doesn't support int4/uint4.
        # Do bitshift operations to pack int4 array into int8 array
        zero_point = zero_point.astype("int8" if dtype == "int4" else "uint8").flatten()
        if zero_point.size % 2 == 1:
            # Add 0 padding to enable int4x2 packing
            zero_point = np.concatenate((zero_point, np.array([0], dtype=zero_point.dtype)))

        if sys.byteorder == "little":
            # Little endian:
            #
            #       zp[n+1]     zp[n]
            #       <-----> | <----->
            # bit:  7 6 5 4   3 2 1 0
            #     (MSB)           (LSB)
            MSB = zero_point[1::2] << 4
            LSB = zero_point[::2] & 0x0F
        else:
            # Big endian:
            #
            #       zp[n]     zp[n+1]
            #       <-----> | <----->
            # bit:  7 6 5 4   3 2 1 0
            #     (MSB)           (LSB)
            MSB = zero_point[::2] << 4
            LSB = zero_point[1::2] & 0x0F

        zero_point_int4x2 = MSB | LSB
        tensor = numpy_helper.from_array(zero_point_int4x2, name=name)

        # Restore data_type to INT4/UINT4
        tensor.data_type = TensorProto.INT4 if dtype == "int4" else TensorProto.UINT4
        tensor.ClearField("dims")
        tensor.dims.extend(target_shape)

        return tensor


class QuantizeLinear(_QdqNodeFactory):
    OPSET = 10
    SUPPORTED_DTYPES = {
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
    }

    @classmethod
    def make_node(cls, name: str, inputs: Iterable[str], output: str,
                  dtype: str, axis: Optional[int] = None,
                  block_size: Optional[int] = None):
        if axis is not None:
            raise RuntimeError(
                f"Per-channel quantization is not supported in opset {cls.OPSET}"
            )

        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node("QuantizeLinear",
                                name=name,
                                inputs=list(inputs),
                                outputs=[output])


class DequantizeLinear(_QdqNodeFactory):
    OPSET = 10
    SUPPORTED_DTYPES = {
        "int8": TensorProto.INT8,
        "uint8": TensorProto.UINT8,
        "int32": TensorProto.INT32,
    }

    @classmethod
    def make_node(cls, name: str, inputs: Iterable[str], output: str,
                  dtype: str, axis: Optional[int] = None,
                  block_size: Optional[int] = None):
        if axis is not None:
            raise RuntimeError(
                f"Per-channel quantization is not supported in opset {cls.OPSET}"
            )

        if block_size is not None:
            raise RuntimeError(
                f"Blockwise quantization is not supported in opset {cls.OPSET}"
            )

        cls._check_dtype(dtype)

        return helper.make_node("DequantizeLinear",
                                name=name,
                                inputs=list(inputs),
                                outputs=[output])
