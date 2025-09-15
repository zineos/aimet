# AIMET API 参考文档

## 1. 概述

本文档提供了AIMET系统所有公共API的详细参考信息，包括类、方法、参数和使用示例。

## 2. 核心API

### 2.1 QuantizationSimModel

量化仿真模型的主要接口。

#### 2.1.1 类定义
```python
class QuantizationSimModel:
    """
    量化仿真模型主类
    
    Args:
        model (torch.nn.Module): 要量化的PyTorch模型
        dummy_input (torch.Tensor): 用于模型分析的虚拟输入
        quant_scheme (QuantScheme): 量化方案，默认为post_training_tf_enhanced
        default_output_bw (int): 默认输出位宽，默认为8
        default_param_bw (int): 默认参数位宽，默认为8
        config_file (str, optional): 量化配置文件路径
    
    Example:
        >>> model = torchvision.models.resnet18()
        >>> dummy_input = torch.randn(1, 3, 224, 224)
        >>> sim = QuantizationSimModel(model, dummy_input)
    """
```

#### 2.1.2 主要方法

##### compute_encodings()
```python
def compute_encodings(self, 
                     forward_pass_callback: Callable,
                     forward_pass_callback_args: Any = None) -> None:
    """
    计算量化编码参数
    
    Args:
        forward_pass_callback: 前向传播回调函数，用于收集统计信息
        forward_pass_callback_args: 回调函数的参数
    
    Raises:
        RuntimeError: 当没有可量化的层时
        
    Example:
        >>> def calibration_fn(model, args=None):
        ...     for batch in calibration_data:
        ...         model(batch)
        >>> sim.compute_encodings(calibration_fn)
    """
```

##### export()
```python
def export(self, 
           path: str, 
           filename_prefix: str, 
           dummy_input: torch.Tensor,
           export_format: str = 'onnx') -> Tuple[str, str]:
    """
    导出量化模型
    
    Args:
        path: 导出路径
        filename_prefix: 文件名前缀
        dummy_input: 用于导出的虚拟输入
        export_format: 导出格式，支持'onnx', 'torchscript'
    
    Returns:
        Tuple[模型文件路径, 编码文件路径]
        
    Example:
        >>> model_path, encoding_path = sim.export('./output', 'quantized_model', dummy_input)
    """
```

##### set_and_freeze_param_encodings()
```python
def set_and_freeze_param_encodings(self, encoding_path: str) -> None:
    """
    设置并冻结参数编码
    
    Args:
        encoding_path: 编码文件路径
        
    Example:
        >>> sim.set_and_freeze_param_encodings('./encodings.json')
    """
```

### 2.2 TensorQuantizer

张量量化器类。

#### 2.2.1 类定义
```python
class TensorQuantizer(torch.nn.Module):
    """
    张量量化器
    
    Args:
        bitwidth (int): 量化位宽，默认为8
        quant_scheme (QuantScheme): 量化方案
        use_symmetric_encodings (bool): 是否使用对称编码，默认为True
        enabled (bool): 是否启用量化器，默认为True
        data_type (QuantizationDataType): 量化数据类型，默认为int
    
    Example:
        >>> quantizer = TensorQuantizer(bitwidth=8, quant_scheme=QuantScheme.post_training_tf)
    """
```

#### 2.2.2 主要方法

##### forward()
```python
def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    前向传播
    
    Args:
        input_tensor: 输入张量
    
    Returns:
        量化后的张量（如果启用）或原始张量
        
    Example:
        >>> quantized_tensor = quantizer(input_tensor)
    """
```

##### compute_encoding()
```python
def compute_encoding(self) -> None:
    """
    计算量化编码参数
    
    Raises:
        RuntimeError: 当没有收集到统计信息时
        
    Example:
        >>> quantizer.set_stats_collection_mode(True)
        >>> quantizer(data)
        >>> quantizer.compute_encoding()
    """
```

##### get_encoding()
```python
def get_encoding(self) -> Optional[QuantizationEncoding]:
    """
    获取量化编码
    
    Returns:
        量化编码对象，如果编码未计算则返回None
        
    Example:
        >>> encoding = quantizer.get_encoding()
        >>> if encoding:
        ...     print(f"Scale: {encoding.scale}, Offset: {encoding.offset}")
    """
```

##### set_encoding()
```python
def set_encoding(self, encoding: QuantizationEncoding) -> None:
    """
    设置量化编码
    
    Args:
        encoding: 量化编码对象
        
    Example:
        >>> from aimet.common.defs import QuantizationEncoding
        >>> encoding = QuantizationEncoding(min=-1.0, max=1.0, scale=0.1, offset=0, bitwidth=8)
        >>> quantizer.set_encoding(encoding)
    """
```

### 2.3 EncodingAnalyzer

编码分析器基类和具体实现。

#### 2.3.1 基类定义
```python
class EncodingAnalyzer(ABC):
    """
    编码分析器抽象基类
    
    Attributes:
        stats_updated (bool): 统计信息是否已更新
    """
```

#### 2.3.2 MinMaxEncodingAnalyzer
```python
class MinMaxEncodingAnalyzer(EncodingAnalyzer):
    """
    最小最大值编码分析器
    
    Example:
        >>> analyzer = MinMaxEncodingAnalyzer()
        >>> analyzer.update_stats(tensor)
        >>> encoding = analyzer.compute_encoding(bitwidth=8, use_symmetric=True)
    """
```

##### update_stats()
```python
def update_stats(self, tensor: torch.Tensor) -> None:
    """
    更新统计信息
    
    Args:
        tensor: 输入张量
        
    Example:
        >>> analyzer.update_stats(torch.randn(10, 10))
    """
```

##### compute_encoding()
```python
def compute_encoding(self, bitwidth: int, use_symmetric: bool) -> QuantizationEncoding:
    """
    计算编码参数
    
    Args:
        bitwidth: 量化位宽
        use_symmetric: 是否使用对称编码
    
    Returns:
        量化编码对象
        
    Example:
        >>> encoding = analyzer.compute_encoding(8, True)
    """
```

### 2.4 ConnectedGraph

连接图类，用于模型结构分析。

#### 2.4.1 类定义
```python
class ConnectedGraph:
    """
    连接图类
    
    Args:
        model (torch.nn.Module): PyTorch模型
        dummy_input (torch.Tensor, optional): 虚拟输入用于分析
    
    Example:
        >>> graph = ConnectedGraph(model, dummy_input)
    """
```

#### 2.4.2 主要方法

##### get_all_ops()
```python
def get_all_ops(self) -> Dict[str, Operation]:
    """
    获取所有操作节点
    
    Returns:
        操作节点字典，键为操作名称，值为Operation对象
        
    Example:
        >>> ops = graph.get_all_ops()
        >>> for name, op in ops.items():
        ...     print(f"Operation: {name}, Type: {op.type}")
    """
```

##### get_ordered_ops()
```python
def get_ordered_ops(self) -> List[Operation]:
    """
    获取拓扑排序的操作列表
    
    Returns:
        按执行顺序排列的操作列表
        
    Example:
        >>> ordered_ops = graph.get_ordered_ops()
        >>> for op in ordered_ops:
        ...     print(op.name)
    """
```

## 3. 数据类型和枚举

### 3.1 QuantScheme
```python
class QuantScheme(Enum):
    """
    量化方案枚举
    
    Attributes:
        min_max: 最小最大值量化
        post_training_tf: TensorFlow风格量化（等同于min_max）
        post_training_tf_enhanced: TensorFlow增强量化（基于KL散度）
        post_training_percentile: 百分位量化
    
    Example:
        >>> scheme = QuantScheme.post_training_tf_enhanced
    """
```

### 3.2 QuantizationDataType
```python
class QuantizationDataType(Enum):
    """
    量化数据类型枚举
    
    Attributes:
        int: 整数量化
        float: 浮点量化
    
    Example:
        >>> dtype = QuantizationDataType.int
    """
```

### 3.3 QuantizationEncoding
```python
@dataclass
class QuantizationEncoding:
    """
    量化编码参数
    
    Attributes:
        min (float): 量化范围最小值
        max (float): 量化范围最大值
        scale (float): 量化缩放因子
        offset (float): 量化偏移量
        bitwidth (int): 量化位宽
        symmetric (bool): 是否对称编码
        data_type (QuantizationDataType): 数据类型
    
    Example:
        >>> encoding = QuantizationEncoding(
        ...     min=-1.0, max=1.0, scale=0.1, offset=0, 
        ...     bitwidth=8, symmetric=True
        ... )
    """
```

## 4. 工具函数

### 4.1 量化工具
```python
def compute_quantization_range(bitwidth: int, signed: bool = True) -> Tuple[int, int]:
    """
    计算量化范围
    
    Args:
        bitwidth: 量化位宽
        signed: 是否有符号
    
    Returns:
        (qmin, qmax) 量化范围元组
        
    Example:
        >>> qmin, qmax = compute_quantization_range(8, True)  # (-128, 127)
    """
```

### 4.2 模型工具
```python
def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    获取模型大小信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        包含参数数量和内存大小的字典
        
    Example:
        >>> size_info = get_model_size(model)
        >>> print(f"Parameters: {size_info['num_params']}")
    """
```

## 5. 配置API

### 5.1 量化配置
```python
class QuantSimConfig:
    """
    量化仿真配置类
    
    Example:
        >>> config = QuantSimConfig.from_json('config.json')
        >>> sim = QuantizationSimModel(model, dummy_input, config=config)
    """
```

#### 5.1.1 配置文件格式
```json
{
  "defaults": {
    "ops": {
      "is_output_quantized": "True"
    },
    "params": {
      "is_quantized": "True",
      "is_symmetric": "True"
    },
    "strict_symmetric": "False",
    "unsigned_symmetric": "True",
    "per_channel_quantization": "False"
  },
  "params": {
    "bias": {
      "is_quantized": "False"
    }
  },
  "op_type": {
    "Conv": {
      "is_input_quantized": "True",
      "is_output_quantized": "True",
      "params": {
        "weight": {
          "is_quantized": "True"
        }
      }
    }
  }
}
```

## 6. 异常类

### 6.1 AIMET异常层次
```python
class AimetError(Exception):
    """AIMET基础异常类"""
    pass

class QuantizationError(AimetError):
    """量化相关异常"""
    pass

class EncodingComputationError(QuantizationError):
    """编码计算异常"""
    pass

class ModelExportError(AimetError):
    """模型导出异常"""
    pass
```

## 7. 使用模式和最佳实践

### 7.1 基本量化流程
```python
# 1. 创建量化仿真模型
sim = QuantizationSimModel(model, dummy_input)

# 2. 准备校准数据
def calibration_fn(model, args=None):
    for batch in calibration_data:
        with torch.no_grad():
            model(batch)

# 3. 计算编码
sim.compute_encodings(calibration_fn)

# 4. 评估量化效果
with torch.no_grad():
    quantized_output = sim(test_input)

# 5. 导出模型
sim.export('./output', 'quantized_model', dummy_input)
```

### 7.2 自定义量化器
```python
# 创建自定义量化器
custom_quantizer = TensorQuantizer(
    bitwidth=4,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    use_symmetric_encodings=False
)

# 手动设置编码
encoding = QuantizationEncoding(
    min=-2.0, max=2.0, scale=0.2, offset=10, bitwidth=4
)
custom_quantizer.set_encoding(encoding)
```

### 7.3 Per-channel量化
```python
# 启用per-channel量化
config = {
    "defaults": {
        "per_channel_quantization": "True"
    },
    "op_type": {
        "Conv": {
            "per_channel_quantization": "True"
        }
    }
}

sim = QuantizationSimModel(model, dummy_input, config=config)
```

## 8. 性能优化API

### 8.1 内存优化
```python
def optimize_memory_usage(sim: QuantizationSimModel) -> None:
    """
    优化内存使用
    
    Args:
        sim: 量化仿真模型
        
    Example:
        >>> optimize_memory_usage(sim)
    """
```

### 8.2 并行计算
```python
def enable_parallel_encoding_computation(sim: QuantizationSimModel, 
                                       num_workers: int = 4) -> None:
    """
    启用并行编码计算
    
    Args:
        sim: 量化仿真模型
        num_workers: 工作进程数
        
    Example:
        >>> enable_parallel_encoding_computation(sim, num_workers=8)
    """
```

## 9. 调试和监控API

### 9.1 量化统计
```python
def get_quantization_stats(sim: QuantizationSimModel) -> Dict[str, Any]:
    """
    获取量化统计信息
    
    Args:
        sim: 量化仿真模型
    
    Returns:
        包含量化统计信息的字典
        
    Example:
        >>> stats = get_quantization_stats(sim)
        >>> print(f"Total quantizers: {stats['total_quantizers']}")
    """
```

### 9.2 可视化API
```python
def visualize_quantization_effects(original_model: torch.nn.Module,
                                 quantized_model: torch.nn.Module,
                                 test_data: torch.Tensor) -> None:
    """
    可视化量化效果
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        test_data: 测试数据
        
    Example:
        >>> visualize_quantization_effects(model, sim.model, test_data)
    """
```

## 10. 版本兼容性

### 10.1 API版本
- **v1.0**: 基础量化功能
- **v1.1**: 增加per-channel量化支持
- **v1.2**: 增加浮点量化支持
- **v2.0**: 重构API，增加自动量化功能

### 10.2 向后兼容性
```python
# 检查API版本兼容性
def check_api_compatibility(required_version: str) -> bool:
    """
    检查API版本兼容性
    
    Args:
        required_version: 需要的最低版本
    
    Returns:
        是否兼容
        
    Example:
        >>> if check_api_compatibility('1.1'):
        ...     # 使用per-channel量化功能
        ...     pass
    """
```

这个API参考文档提供了AIMET系统所有主要接口的详细说明，包括参数、返回值、异常处理和使用示例，为开发者提供了完整的API使用指南。