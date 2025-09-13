# 张量量化器 (TensorQuantizer) 设计文档

## 1. 模块概述

### 1.1 职责定义
TensorQuantizer是AIMET系统中负责单个张量量化操作的核心模块，主要职责包括：
- 执行张量的量化和反量化操作
- 管理量化编码参数（scale, offset）
- 收集统计信息用于编码计算
- 支持多种量化数据类型和位宽

### 1.2 设计目标
- **高性能**：优化的量化计算实现
- **灵活性**：支持多种量化方案和参数配置
- **精确性**：保持量化过程的数值精度
- **可扩展性**：支持新的量化算法接入

## 2. 架构设计

### 2.1 类层次结构
**查看TensorQuantizer类层次结构图**: 在 [architecture_diagrams.html](./architecture_diagrams.html) 中的"TensorQuantizer 类层次结构"部分展示了：

- 🟣 **抽象基类**: QuantizerBase用紫色虚线框表示抽象类
- 🔴 **主实现类**: TensorQuantizer用红色突出显示
- 🔵 **基类分支**: AffineQuantizerBase和FloatQuantizerBase
- 🟢 **具体实现**: 各种具体的量化器实现
- 🟠 **组合关系**: EncodingAnalyzer的组合关系
- 📊 **继承层次**: 清晰的继承关系可视化

### 2.2 核心组件关系
**组件交互关系**: TensorQuantizer与其他组件的交互关系在可视化图表中清晰展示：

- **EncodingAnalyzer**: 负责统计信息收集和编码计算
- **QuantizationEncoding**: 存储量化参数（scale, offset等）
- **QuantizationBackend**: 提供具体的量化计算实现
- **多种分析器**: MinMax、TF-Enhanced、Percentile等不同算法
- **多种后端**: PyTorch内置、CUDA加速等实现方式

图表中使用不同的颜色和线条类型来区分继承关系、组合关系和关联关系。

## 3. 详细设计

### 3.1 核心类定义

#### 3.1.1 基础量化器类
```python
import torch
from torch import nn
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

class QuantizerBase(ABC, torch.nn.Module):
    """量化器抽象基类"""
    
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.encoding_computed = False
        
    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """量化前向传播"""
        pass
    
    @abstractmethod
    def compute_encoding(self) -> None:
        """计算量化编码参数"""
        pass
    
    @abstractmethod
    def get_encoding(self) -> 'QuantizationEncoding':
        """获取量化编码"""
        pass
    
    @abstractmethod
    def set_encoding(self, encoding: 'QuantizationEncoding') -> None:
        """设置量化编码"""
        pass


class TensorQuantizer(QuantizerBase):
    """张量量化器主类"""
    
    def __init__(self,
                 bitwidth: int = 8,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                 use_symmetric_encodings: bool = True,
                 enabled: bool = True,
                 data_type: QuantizationDataType = QuantizationDataType.int):
        """
        初始化张量量化器
        
        Args:
            bitwidth: 量化位宽
            quant_scheme: 量化方案
            use_symmetric_encodings: 是否使用对称编码
            enabled: 是否启用量化器
            data_type: 量化数据类型
        """
        super().__init__()
        
        self.bitwidth = bitwidth
        self.quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = enabled
        self.data_type = data_type
        
        # 量化参数
        self.register_parameter('scale', nn.Parameter(torch.tensor(1.0), requires_grad=False))
        self.register_parameter('offset', nn.Parameter(torch.tensor(0.0), requires_grad=False))
        
        # 编码分析器
        self.encoding_analyzer = self._create_encoding_analyzer()
        
        # 状态标志
        self.stats_collection_mode = False
        self.encoding_computed = False
        
        # 量化范围
        self.qmin, self.qmax = self._compute_quantization_range()
    
    def _create_encoding_analyzer(self):
        """根据量化方案创建编码分析器"""
        if self.quant_scheme == QuantScheme.post_training_tf:
            return MinMaxEncodingAnalyzer()
        elif self.quant_scheme == QuantScheme.post_training_tf_enhanced:
            return TfEnhancedEncodingAnalyzer()
        elif self.quant_scheme == QuantScheme.post_training_percentile:
            return PercentileEncodingAnalyzer()
        else:
            raise ValueError(f"Unsupported quantization scheme: {self.quant_scheme}")
    
    def _compute_quantization_range(self):
        """计算量化范围"""
        if self.data_type == QuantizationDataType.int:
            if self.use_symmetric_encodings:
                # 对称量化：[-2^(n-1), 2^(n-1)-1]
                qmax = 2**(self.bitwidth - 1) - 1
                qmin = -qmax
            else:
                # 非对称量化：[0, 2^n-1]
                qmin = 0
                qmax = 2**self.bitwidth - 1
        else:
            # 浮点量化的范围由具体实现决定
            qmin, qmax = self._compute_float_quantization_range()
        
        return qmin, qmax
    
    def _compute_float_quantization_range(self):
        """计算浮点量化范围"""
        if self.bitwidth == 16:
            # FP16范围
            return -65504.0, 65504.0
        elif self.bitwidth == 8:
            # 自定义FP8范围
            return -240.0, 240.0
        else:
            raise ValueError(f"Unsupported float bitwidth: {self.bitwidth}")
```

#### 3.1.2 前向传播实现
```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    张量量化器前向传播
    
    Args:
        input_tensor: 输入张量
        
    Returns:
        量化后的张量（如果启用）或原始张量
    """
    if not self.enabled:
        return input_tensor
    
    if self.stats_collection_mode:
        # 统计收集模式：更新统计信息但不量化
        self.encoding_analyzer.update_stats(input_tensor)
        return input_tensor
    else:
        # 推理模式：执行量化
        if not self.encoding_computed:
            # 如果编码未计算，发出警告并返回原始张量
            logger.warning(f"Encoding not computed for quantizer. Returning original tensor.")
            return input_tensor
        
        return self.quantize_dequantize(input_tensor)

def quantize_dequantize(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    量化后立即反量化
    
    Args:
        input_tensor: 输入张量
        
    Returns:
        量化-反量化后的张量
    """
    if self.data_type == QuantizationDataType.int:
        return self._quantize_dequantize_int(input_tensor)
    else:
        return self._quantize_dequantize_float(input_tensor)

def _quantize_dequantize_int(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """整数量化-反量化"""
    # 1. 量化：x_q = clamp(round(x / scale) + offset, qmin, qmax)
    quantized = torch.clamp(
        torch.round(input_tensor / self.scale) + self.offset,
        self.qmin, self.qmax
    )
    
    # 2. 反量化：x_dq = (x_q - offset) * scale
    dequantized = (quantized - self.offset) * self.scale
    
    return dequantized

def _quantize_dequantize_float(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """浮点量化-反量化"""
    if self.bitwidth == 16:
        # FP16量化
        return input_tensor.half().float()
    elif self.bitwidth == 8:
        # 自定义FP8量化
        return self._custom_fp8_quantize_dequantize(input_tensor)
    else:
        raise ValueError(f"Unsupported float bitwidth: {self.bitwidth}")

def _custom_fp8_quantize_dequantize(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """自定义FP8量化-反量化"""
    # 简化的FP8实现
    # 实际实现需要考虑指数位和尾数位的分配
    clamped = torch.clamp(input_tensor, -240.0, 240.0)
    
    # 模拟FP8的精度损失
    scale_factor = 240.0 / 127.0  # 映射到8位整数范围
    quantized_int = torch.round(clamped / scale_factor)
    dequantized = quantized_int * scale_factor
    
    return dequantized
```

### 3.2 编码计算

#### 3.2.1 编码计算主流程
```python
def compute_encoding(self) -> None:
    """计算量化编码参数"""
    if not self.encoding_analyzer.stats_updated:
        raise RuntimeError("No statistics collected for encoding computation")
    
    try:
        # 从编码分析器获取编码参数
        encoding = self.encoding_analyzer.compute_encoding(
            self.bitwidth, 
            self.use_symmetric_encodings
        )
        
        # 验证编码参数的有效性
        self._validate_encoding(encoding)
        
        # 更新量化参数
        self._update_quantization_parameters(encoding)
        
        # 设置编码计算完成标志
        self.encoding_computed = True
        
        logger.debug(f"Computed encoding: min={encoding.min:.6f}, max={encoding.max:.6f}, "
                    f"scale={encoding.scale:.6f}, offset={encoding.offset}")
        
    except Exception as e:
        logger.error(f"Failed to compute encoding: {str(e)}")
        raise

def _validate_encoding(self, encoding):
    """验证编码参数的有效性"""
    # 检查scale是否为正数
    if encoding.scale <= 0:
        raise ValueError(f"Invalid scale value: {encoding.scale}. Scale must be positive.")
    
    # 检查scale是否过小（可能导致数值问题）
    min_scale = 1e-10
    if encoding.scale < min_scale:
        logger.warning(f"Scale value {encoding.scale} is very small, may cause numerical issues")
    
    # 检查offset是否在合理范围内
    if not (self.qmin <= encoding.offset <= self.qmax):
        raise ValueError(f"Offset {encoding.offset} is outside quantization range [{self.qmin}, {self.qmax}]")

def _update_quantization_parameters(self, encoding):
    """更新量化参数"""
    self.scale.data.fill_(encoding.scale)
    self.offset.data.fill_(encoding.offset)
    
    # 存储原始编码信息用于导出
    self._encoding = encoding
```

#### 3.2.2 统计信息管理
```python
def reset_encoding_stats(self) -> None:
    """重置编码统计信息"""
    self.encoding_analyzer.reset_stats()
    self.encoding_computed = False
    logger.debug("Reset encoding statistics")

def set_stats_collection_mode(self, enabled: bool) -> None:
    """设置统计收集模式"""
    self.stats_collection_mode = enabled
    logger.debug(f"Stats collection mode set to: {enabled}")

def get_stats_summary(self) -> dict:
    """获取统计信息摘要"""
    if hasattr(self.encoding_analyzer, 'get_stats_summary'):
        return self.encoding_analyzer.get_stats_summary()
    else:
        return {
            'stats_updated': self.encoding_analyzer.stats_updated,
            'min_val': getattr(self.encoding_analyzer, 'min_val', None),
            'max_val': getattr(self.encoding_analyzer, 'max_val', None)
        }
```

### 3.3 编码管理

#### 3.3.1 编码获取和设置
```python
def get_encoding(self) -> 'QuantizationEncoding':
    """获取当前的量化编码"""
    if not self.encoding_computed:
        return None
    
    return QuantizationEncoding(
        min=self._encoding.min,
        max=self._encoding.max,
        scale=self.scale.item(),
        offset=self.offset.item(),
        bitwidth=self.bitwidth,
        symmetric=self.use_symmetric_encodings,
        data_type=self.data_type
    )

def set_encoding(self, encoding: 'QuantizationEncoding') -> None:
    """设置量化编码"""
    # 验证编码兼容性
    self._validate_encoding_compatibility(encoding)
    
    # 更新量化参数
    self.scale.data.fill_(encoding.scale)
    self.offset.data.fill_(encoding.offset)
    
    # 更新内部状态
    self._encoding = encoding
    self.encoding_computed = True
    
    logger.debug(f"Set encoding: scale={encoding.scale:.6f}, offset={encoding.offset}")

def _validate_encoding_compatibility(self, encoding):
    """验证编码兼容性"""
    if encoding.bitwidth != self.bitwidth:
        raise ValueError(f"Encoding bitwidth {encoding.bitwidth} does not match "
                        f"quantizer bitwidth {self.bitwidth}")
    
    if encoding.data_type != self.data_type:
        raise ValueError(f"Encoding data type {encoding.data_type} does not match "
                        f"quantizer data type {self.data_type}")

def export_encoding_dict(self) -> dict:
    """导出编码为字典格式"""
    if not self.encoding_computed:
        return None
    
    encoding_dict = {
        'min': self._encoding.min,
        'max': self._encoding.max,
        'scale': self.scale.item(),
        'offset': self.offset.item(),
        'bitwidth': self.bitwidth,
        'symmetric': self.use_symmetric_encodings,
        'data_type': self.data_type.name
    }
    
    return encoding_dict

def load_encoding_dict(self, encoding_dict: dict) -> None:
    """从字典加载编码"""
    encoding = QuantizationEncoding(
        min=encoding_dict['min'],
        max=encoding_dict['max'],
        scale=encoding_dict['scale'],
        offset=encoding_dict['offset'],
        bitwidth=encoding_dict['bitwidth'],
        symmetric=encoding_dict['symmetric'],
        data_type=QuantizationDataType[encoding_dict['data_type']]
    )
    
    self.set_encoding(encoding)
```

### 3.4 高级功能

#### 3.4.1 量化感知训练支持
```python
def enable_param_quantization(self) -> None:
    """启用参数量化（用于QAT）"""
    self.scale.requires_grad = True
    self.offset.requires_grad = True

def disable_param_quantization(self) -> None:
    """禁用参数量化"""
    self.scale.requires_grad = False
    self.offset.requires_grad = False

def get_quantization_error(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """计算量化误差"""
    if not self.encoding_computed:
        raise RuntimeError("Encoding must be computed before calculating quantization error")
    
    quantized_output = self.quantize_dequantize(input_tensor)
    error = torch.abs(input_tensor - quantized_output)
    
    return error

def get_quantization_noise_stats(self, input_tensor: torch.Tensor) -> dict:
    """获取量化噪声统计信息"""
    error = self.get_quantization_error(input_tensor)
    
    return {
        'mean_error': error.mean().item(),
        'std_error': error.std().item(),
        'max_error': error.max().item(),
        'snr_db': self._calculate_snr(input_tensor, error)
    }

def _calculate_snr(self, signal: torch.Tensor, noise: torch.Tensor) -> float:
    """计算信噪比"""
    signal_power = torch.mean(signal**2)
    noise_power = torch.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()
```

#### 3.4.2 Per-channel量化支持
```python
class PerChannelTensorQuantizer(TensorQuantizer):
    """Per-channel张量量化器"""
    
    def __init__(self, 
                 channel_axis: int,
                 num_channels: int,
                 **kwargs):
        """
        初始化per-channel量化器
        
        Args:
            channel_axis: 通道轴索引
            num_channels: 通道数量
        """
        super().__init__(**kwargs)
        
        self.channel_axis = channel_axis
        self.num_channels = num_channels
        
        # 创建per-channel参数
        scale_shape = [1] * 4  # 假设4D张量
        scale_shape[channel_axis] = num_channels
        
        self.scale = nn.Parameter(torch.ones(scale_shape), requires_grad=False)
        self.offset = nn.Parameter(torch.zeros(scale_shape), requires_grad=False)
    
    def _quantize_dequantize_int(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Per-channel整数量化-反量化"""
        # 确保scale和offset的形状与输入兼容
        scale = self.scale.expand_as(input_tensor)
        offset = self.offset.expand_as(input_tensor)
        
        # 量化
        quantized = torch.clamp(
            torch.round(input_tensor / scale) + offset,
            self.qmin, self.qmax
        )
        
        # 反量化
        dequantized = (quantized - offset) * scale
        
        return dequantized
```

### 3.5 性能优化

#### 3.5.1 CUDA加速支持
```python
def _quantize_dequantize_cuda(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """CUDA加速的量化-反量化"""
    if not torch.cuda.is_available() or not input_tensor.is_cuda:
        return self._quantize_dequantize_int(input_tensor)
    
    # 使用CUDA内核进行优化计算
    try:
        from aimet_torch.quantization_cuda import quantize_dequantize_cuda
        return quantize_dequantize_cuda(
            input_tensor, self.scale, self.offset, self.qmin, self.qmax
        )
    except ImportError:
        # 回退到CPU实现
        return self._quantize_dequantize_int(input_tensor)

def _optimize_for_inference(self):
    """推理优化"""
    # 预计算常用值
    self._inv_scale = 1.0 / self.scale
    
    # 融合操作
    self._fused_quantize_params = {
        'scale': self.scale.item(),
        'offset': self.offset.item(),
        'inv_scale': self._inv_scale.item(),
        'qmin': self.qmin,
        'qmax': self.qmax
    }
```

#### 3.5.2 内存优化
```python
def cleanup_intermediate_results(self):
    """清理中间结果以节省内存"""
    if hasattr(self, '_intermediate_stats'):
        del self._intermediate_stats
    
    # 清理编码分析器的中间结果
    if hasattr(self.encoding_analyzer, 'cleanup'):
        self.encoding_analyzer.cleanup()

def get_memory_footprint(self) -> dict:
    """获取内存占用情况"""
    footprint = {
        'parameters': sum(p.numel() * p.element_size() for p in self.parameters()),
        'buffers': sum(b.numel() * b.element_size() for b in self.buffers())
    }
    
    if hasattr(self.encoding_analyzer, 'get_memory_footprint'):
        footprint['encoding_analyzer'] = self.encoding_analyzer.get_memory_footprint()
    
    return footprint
```

## 4. 量化编码数据结构

### 4.1 编码类定义
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantizationEncoding:
    """量化编码参数"""
    min: float
    max: float
    scale: float
    offset: float
    bitwidth: int
    symmetric: bool = True
    data_type: QuantizationDataType = QuantizationDataType.int
    
    def __post_init__(self):
        """后处理验证"""
        if self.scale <= 0:
            raise ValueError("Scale must be positive")
        
        if self.bitwidth < 1 or self.bitwidth > 32:
            raise ValueError("Bitwidth must be between 1 and 32")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'min': self.min,
            'max': self.max,
            'scale': self.scale,
            'offset': self.offset,
            'bitwidth': self.bitwidth,
            'symmetric': self.symmetric,
            'data_type': self.data_type.name
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'QuantizationEncoding':
        """从字典创建"""
        return cls(
            min=data['min'],
            max=data['max'],
            scale=data['scale'],
            offset=data['offset'],
            bitwidth=data['bitwidth'],
            symmetric=data['symmetric'],
            data_type=QuantizationDataType[data['data_type']]
        )
```

## 5. 测试和验证

### 5.1 单元测试
```python
import unittest

class TestTensorQuantizer(unittest.TestCase):
    def setUp(self):
        self.quantizer = TensorQuantizer(
            bitwidth=8,
            quant_scheme=QuantScheme.post_training_tf_enhanced
        )
    
    def test_quantization_basic(self):
        """测试基本量化功能"""
        input_tensor = torch.randn(10, 10)
        
        # 收集统计信息
        self.quantizer.set_stats_collection_mode(True)
        self.quantizer(input_tensor)
        
        # 计算编码
        self.quantizer.compute_encoding()
        
        # 测试量化
        self.quantizer.set_stats_collection_mode(False)
        output = self.quantizer(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertTrue(self.quantizer.encoding_computed)
    
    def test_encoding_persistence(self):
        """测试编码的持久化"""
        # 设置编码
        encoding = QuantizationEncoding(
            min=-1.0, max=1.0, scale=0.1, offset=0, bitwidth=8
        )
        self.quantizer.set_encoding(encoding)
        
        # 获取编码并验证
        retrieved_encoding = self.quantizer.get_encoding()
        self.assertAlmostEqual(retrieved_encoding.scale, 0.1)
        self.assertEqual(retrieved_encoding.bitwidth, 8)
```

这个TensorQuantizer设计提供了完整的张量量化功能，支持多种量化方案、数据类型和优化特性，是AIMET量化系统的核心计算模块。