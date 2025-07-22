.. include:: ../abbreviation.txt

.. _techniques-lpbq:

################################
Low-Power Blockwise Quantization
################################

|qnn| supports an alternative to blockwise quantization called *Low-Power Blockwise Quantization* (*LPBQ*).

.. note::
   To read about generic blockwise quantization, see :ref:`Blockwise Quantization <techniques-blockwise>`

In LPBQ, blockwise encodings at a lower bit width are adjusted such that they lie on a common higher-bit-width per-channel grid. This enables models to run on existing per channel kernels and still benefit from blockwise quantization. 

LPBQ encodings require less storage than blockwise quantization encodings because only the low bit-width integer scale expansion factors need to be stored blockwise (the floating point encoding scales are stored per-channel).

LPBQ quantization is part of the :class:`aimet_torch.quantization.affine.GroupedBlockQuantizeDequantize` class.

Besides the  `block_size` argument described in the  blockwise Quantization section, LPBQ requires two additional arguments:

`decompressed_bw`
  The higher bit-width value for the per channel grid that the lower bit-width blockwise encodings are expanded to. The `decompressed_bw` value must be greater than or equal to the  bit width of the quantizer.

`block_grouping`
  The number of blocks for each dimension that are grouped when expanding the lower bit-width blockwise encodings. The block grouping for a particular dimension must be divisible by the number of blocks in that dimension.

  As with block size, a block grouping value of -1 is automatically interpreted as the number of blocks in that dimension.

To allow for experimentation, the `GroupedBlockQuantizeDequantize` object supports arbitrary block sizes. However, the Qualcomm runtime imposes the following restrictions on LPBQ:

* Blockwise quantization runs on weight (not activation) quantizers only.
* Block size must be set to one for the output channel dimension.
* Block size may take an arbitrary value for the input channel dimension (it must still divide evenly into the input channel tensor shape).
*  Block size must be equal to the tensor size for all other dimensions.
* Block groupings must be set to one for all dimensions, except for the input channels dimension which should be
  set to the number of blocks in that dimension.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            from aimet_torch.quantization.affine import GroupedBlockQuantizeDequantize

            # Assume sim.model.conv_1 refers to a QuantizedConv2d layer with weight param shape of (16, 64, 2, 2)
            # Below settings equate to a block size of 16 in the input channels dimension.
            sim.model.conv_1.param_quantizers['weight'] = GroupedBlockQuantizeDequantize(shape=(16, 4, 1, 1),
                                                                                         bitwidth=4,
                                                                                         symmetric=True,
                                                                                         block_size=(1, 16, 2, 2),
                                                                                         decompressed_bw=8,
                                                                                         block_grouping=(1, 4, 1, 1))   # (1, -1, 1, 1) works too

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Exporting blockwise-quantized models
====================================

Blockwise quantization generates a larger number of encodings than per tensor or per channel quantization. To reduce the size of the exported encodings JSON file and the time needed to write the file, blockwise quantization uses an improved file format, designated 1.0.0, for the export.

The 1.0.0 encoding format is supported by the Qualcomm runtime and can be used to export per tensor, per channel, blockwise, and LPBQ quantizer encodings.

.. important::
    If  blockwise and/or LPBQ quantizers are present in the model, the 1.0.0 format *must* be used when exporting encodings for the Qualcomm runtime.

The following code snippet shows how to export encodings in the 1.0.0 format:

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. code-block:: Python

            from aimet_common import quantsim

            # Assume 'sim' is a QuantizationSimModel object imported from aimet_torch.quantsim

            # Set encoding_version to 1.0.0
            quantsim.encoding_version = '1.0.0'
            sim.export('./data', 'exported_model', dummy_input)

    .. tab-item:: TensorFlow
        :sync: tf

        Not supported.

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

See the :ref:`Encoding specifications <quantsim-encoding-spec>` page, which describes encodings specifications in detail.

API
===

**Top-level API to configure LPBQ quantization**

.. autofunction:: aimet_torch.v2.quantsim.config_utils.set_grouped_blockwise_quantization_for_weights
    :noindex:

This utility enables you to configure quantized layers to use grouped blockwise quantization by supplying a `decompressed_bw`,  `block_size`, and `block_grouping`. Similar to :func:`set_blockwise_quantization_for_weights`, `block_grouping` can be a single value. In this case the input_channel's dimension is assigned the value, and all other dimensions are assigned a value of one.

Different layers can have different numbers of blocks for the input channels dimension for the same block size. If you assign -1 as the single `block_grouping` value, the input channels dimension automatically uses a `block_grouping` value equal to the number of blocks in any affected layer. This enbles you to configure all affected layers to LPBQ quantization with a single API call.
