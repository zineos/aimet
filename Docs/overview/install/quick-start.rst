.. _install-quick-start:

#####################
Quick Start
#####################

This page describes how to quickly install the latest version of AIMET for the ONNX and PyTorch framework.

For all the framework variants and compute platforms, see :ref:`Installation <install-index>`.

.. _install-quick-start-platform:

Tested platform
===============

aimet-torch and aimet-onnx have been validated on the following platform:

* 64-bit Intel x86-compatible processor
* Python 3.10
* Ubuntu 22.04
* For GPU variants:
    * Nvidia GPU card (Compute capability 5.2 or later)
    * Nvidia driver version 455 or later (using the latest driver is recommended; both CUDA and cuDNN are supported)

Installing AIMET
================

Install AIMET from PyPI

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
      :sync: onnx

      .. code-block:: bash

        pip install aimet-onnx

    .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: Bash

        pip install aimet-torch

Verifying the installation
==========================

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
      :sync: onnx

      .. code-block:: Python

        import aimet_onnx
        print(aimet_onnx.__version__)

    .. tab-item:: PyTorch
      :sync: torch

      .. code-block:: Python

        import aimet_torch
        print(aimet_torch.__version__)


Quantize a small model quickly with AIMET
=========================================

Create a :class:`QuantizationSimModel`, perform calibration,
and then evaluate it.

**Step 1:** Handle imports and other setup.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_1]
      :end-before: # End of [step_1]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_1]
      :end-before: # End of [step_1]


**Step 2:** Create a :class:`QuantizationSimModel` and ensure the model contains quantization operations.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_2]
      :end-before: # End of [step_2]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_2]
      :end-before: # End of [step_2]

**Step 3:** Calibrate the model. This example uses random values as input. In real-world cases, calibration should be performed using a representative dataset.


.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_3]
      :end-before: # End of [step_3]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_3]
      :end-before: # End of [step_3]

**Step 4:** Evaluate the model.

Infer directly on QuantSim model to check quantized model's accuracy.

.. tab-set::
  :sync-group: platform

  .. tab-item:: ONNX
    :sync: onnx

    .. literalinclude:: ../../snippets/onnx/installation_verification.py
      :language: python
      :start-after: # [step_4]
      :end-before: # End of [step_4]

  .. tab-item:: PyTorch
    :sync: torch

    .. literalinclude:: ../../snippets/torch/installation_verification.py
      :language: python
      :start-after: # [step_4]
      :end-before: # End of [step_4]

Sample output of QuantSim model is shown below:

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. rst-class:: script-output

          .. code-block:: none

            [array([[-0.4599525 ,  0.35107604,  0.43178225, ..., -0.45040053,
                      0.1450607 ,  0.23799022],
                    [-0.4132449 ,  0.20722957,  0.60808927, ..., -0.5315115 ,
                      -0.01675645,  0.22884297],
                    [-0.4677236 ,  0.3576329 ,  0.5317543 , ..., -0.50366503,
                      -0.01392324, -0.0897725 ],
                    ...,
                    [-0.4503196 ,  0.3851556 ,  0.56810045, ..., -0.6998855 ,
                      0.03513189,  0.36678016],
                    [-0.27045077,  0.28065038,  0.46723792, ..., -0.24665177,
                      -0.11899511,  0.03658897],
                    [-0.43477735,  0.35536635,  0.62274104, ..., -0.5091695 ,
                      -0.11446196,  0.10984787]], dtype=float32)]

    .. tab-item:: PyTorch
      :sync: torch
        
      .. rst-class:: script-output

        .. code-block:: none

          DequantizedTensor([[-1.7466,  0.8405,  1.8606,  ..., -0.9714,  0.8366, 2.2363],
                            [-1.6091,  1.0449,  1.7788,  ..., -0.9904,  1.0861, 2.2431],
                            [-1.5307,  0.8442,  1.5157,  ..., -0.7793,  0.6327, 2.3861],
                            ...,
                            [-1.3610,  1.4499,  2.2068,  ..., -0.8188,  1.1155, 2.5962],
                            [-1.1619,  1.2217,  2.1050,  ..., -0.5301,  0.9150, 2.1458],
                            [-1.6340,  0.9826,  2.2459,  ..., -1.0769,  0.9054, 2.2315]],
                            device='cuda:0', grad_fn=<AliasBackward0>)

