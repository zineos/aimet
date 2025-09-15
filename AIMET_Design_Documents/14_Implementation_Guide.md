# AIMET 实现指南

## 1. 概述

本文档提供了从零开始实现AIMET系统的详细步骤指南，适合新手开发者按照此指南逐步构建一个功能完整的量化工具包。

## 2. 开发环境准备

### 2.1 系统要求
- **操作系统**: Linux (Ubuntu 20.04+) 或 macOS
- **Python版本**: 3.8+
- **内存**: 至少16GB RAM
- **存储**: 至少50GB可用空间
- **GPU**: NVIDIA GPU (可选，用于CUDA加速)

### 2.2 开发工具安装
```bash
# 安装Python依赖管理工具
pip install pipenv poetry

# 安装开发工具
pip install black flake8 pytest pytest-cov sphinx

# 安装C++编译工具
sudo apt-get install build-essential cmake

# 安装CUDA (如果需要GPU支持)
# 参考NVIDIA官方文档安装CUDA Toolkit
```

### 2.3 项目初始化
```bash
# 创建项目目录
mkdir aimet-implementation
cd aimet-implementation

# 初始化Git仓库
git init

# 创建Python虚拟环境
python -m venv aimet-env
source aimet-env/bin/activate

# 创建项目结构
mkdir -p aimet/{common,torch,onnx,tests,docs,examples}
```

## 3. 第一阶段：基础框架搭建 (第1-3周)

### 3.1 步骤1：创建项目结构和基础定义

#### 3.1.1 项目目录结构
```
aimet-implementation/
├── aimet/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── defs.py
│   │   ├── connected_graph/
│   │   │   ├── __init__.py
│   │   │   ├── connectedgraph.py
│   │   │   ├── operation.py
│   │   │   └── product.py
│   │   ├── quantsim.py
│   │   └── utils.py
│   ├── torch/
│   │   ├── __init__.py
│   │   ├── quantsim.py
│   │   ├── tensor_quantizer.py
│   │   ├── encoding_analyzer.py
│   │   └── utils.py
│   ├── onnx/
│   │   ├── __init__.py
│   │   └── quantsim.py
│   └── tests/
│       ├── __init__.py
│       ├── test_common/
│       ├── test_torch/
│       └── test_onnx/
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── docs/
```

#### 3.1.2 实现基础定义
```python
# aimet/common/defs.py
from enum import Enum
from dataclasses import dataclass
from typing import Union, Optional

class QuantScheme(Enum):
    """量化方案枚举"""
    min_max = 1
    post_training_tf = min_max
    post_training_tf_enhanced = 2
    post_training_percentile = 3

class QuantizationDataType(Enum):
    """量化数据类型"""
    int = 1
    float = 2

class ActivationType(Enum):
    """激活函数类型"""
    no_activation = 0
    relu = 1
    relu6 = 2

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
        if self.scale <= 0:
            raise ValueError("Scale must be positive")
        if not 1 <= self.bitwidth <= 32:
            raise ValueError("Bitwidth must be between 1 and 32")

# 导出常用类型
__all__ = [
    'QuantScheme', 
    'QuantizationDataType', 
    'ActivationType', 
    'QuantizationEncoding'
]
```

#### 3.1.3 创建setup.py
```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aimet-implementation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Model Efficiency Toolkit Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aimet-implementation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "sphinx"],
        "gpu": ["cupy"],
    },
)
```

### 3.2 步骤2：实现连接图基础框架

#### 3.2.1 实现Product类
```python
# aimet/common/connected_graph/product.py
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .operation import Operation

class Product:
    """数据流边 - 表示操作间的数据传递"""
    
    def __init__(self, name: str, shape: tuple):
        self.name = name
        self.shape = shape
        self.producer: Optional['Operation'] = None
        self.consumers: List['Operation'] = []
        
        # 产品属性
        self.is_param = False
        self.is_model_input = False
        self.is_const = False
        self.param_name = None
    
    def __repr__(self):
        return f"Product(name='{self.name}', shape={self.shape})"
    
    def add_consumer(self, op: 'Operation'):
        """添加消费者操作"""
        if op not in self.consumers:
            self.consumers.append(op)
    
    def remove_consumer(self, op: 'Operation'):
        """移除消费者操作"""
        if op in self.consumers:
            self.consumers.remove(op)
    
    def set_producer(self, op: 'Operation'):
        """设置生产者操作"""
        self.producer = op
```

#### 3.2.2 实现Operation类
```python
# aimet/common/connected_graph/operation.py
from typing import List, Optional, Any
import torch.nn as nn

class Operation:
    """操作节点 - 表示模型中的一个操作"""
    
    def __init__(self, name: str, op_type: str, module: Optional[nn.Module] = None):
        self.name = name
        self.type = op_type
        self.module = module
        
        self.inputs: List['Product'] = []
        self.outputs: List['Product'] = []
        
        # 操作属性
        self.attributes = {}
        self.quantization_enabled = True
    
    def __repr__(self):
        return f"Operation(name='{self.name}', type='{self.type}')"
    
    def add_input(self, product: 'Product'):
        """添加输入产品"""
        if product not in self.inputs:
            self.inputs.append(product)
            product.add_consumer(self)
    
    def add_output(self, product: 'Product'):
        """添加输出产品"""
        if product not in self.outputs:
            self.outputs.append(product)
            product.set_producer(self)
    
    @property
    def input_ops(self) -> List['Operation']:
        """获取输入操作列表"""
        ops = []
        for product in self.inputs:
            if product.producer and product.producer not in ops:
                ops.append(product.producer)
        return ops
    
    @property
    def output_ops(self) -> List['Operation']:
        """获取输出操作列表"""
        ops = []
        for product in self.outputs:
            for consumer in product.consumers:
                if consumer not in ops:
                    ops.append(consumer)
        return ops
```

#### 3.2.3 实现ConnectedGraph类
```python
# aimet/common/connected_graph/connectedgraph.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from .operation import Operation
from .product import Product

class ConnectedGraph(ABC):
    """连接图抽象基类"""
    
    def __init__(self, model: nn.Module, dummy_input: Optional[torch.Tensor] = None):
        self.model = model
        self.dummy_input = dummy_input
        
        self._ops: Dict[str, Operation] = {}
        self._products: Dict[str, Product] = {}
        self._input_ops: List[Operation] = []
        self._output_ops: List[Operation] = []
        
        # 构建图
        self._build_graph()
    
    def _build_graph(self):
        """构建连接图"""
        # 1. 创建操作节点
        self._create_operations()
        
        # 2. 分析数据流
        if self.dummy_input is not None:
            self._analyze_data_flow()
        
        # 3. 建立连接关系
        self._establish_connections()
    
    def _create_operations(self):
        """创建操作节点"""
        for name, module in self.model.named_modules():
            if self._is_leaf_module(module):
                op_type = type(module).__name__
                operation = Operation(name, op_type, module)
                self._ops[name] = operation
    
    def _is_leaf_module(self, module: nn.Module) -> bool:
        """判断是否为叶子模块"""
        # 叶子模块：没有子模块或只有参数的模块
        children = list(module.children())
        parameters = list(module.parameters())
        
        return len(children) == 0 and len(parameters) > 0
    
    def _analyze_data_flow(self):
        """分析数据流"""
        # 使用hook机制分析数据流向
        hooks = []
        activation_map = {}
        
        def create_hook(name):
            def hook_fn(module, input_data, output_data):
                activation_map[name] = {
                    'input': input_data,
                    'output': output_data,
                    'module': module
                }
            return hook_fn
        
        # 注册forward hooks
        for name, module in self.model.named_modules():
            if name in self._ops:
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # 执行前向传播
        try:
            with torch.no_grad():
                self.model.eval()
                _ = self.model(self.dummy_input)
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()
        
        # 从激活映射构建连接
        self._build_connections_from_activations(activation_map)
    
    def _build_connections_from_activations(self, activation_map: Dict[str, Any]):
        """从激活映射构建连接"""
        tensor_to_producer = {}
        
        for op_name, activation_info in activation_map.items():
            if op_name not in self._ops:
                continue
                
            operation = self._ops[op_name]
            
            # 处理输入
            inputs = activation_info['input']
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            
            for i, input_tensor in enumerate(inputs):
                if isinstance(input_tensor, torch.Tensor):
                    tensor_id = id(input_tensor)
                    product_name = f"{op_name}_input_{i}"
                    
                    if tensor_id in tensor_to_producer:
                        # 找到生产者
                        producer_op = tensor_to_producer[tensor_id]
                        product = Product(f"{producer_op.name}_to_{op_name}", input_tensor.shape)
                        
                        producer_op.add_output(product)
                        operation.add_input(product)
                        
                        self._products[product.name] = product
                    else:
                        # 可能是模型输入
                        product = Product(product_name, input_tensor.shape)
                        product.is_model_input = True
                        operation.add_input(product)
                        self._products[product.name] = product
            
            # 处理输出
            output = activation_info['output']
            if isinstance(output, torch.Tensor):
                tensor_to_producer[id(output)] = operation
    
    def _establish_connections(self):
        """建立连接关系"""
        # 识别输入和输出操作
        for op in self._ops.values():
            # 如果操作没有输入或输入都是模型输入，则为输入操作
            if not op.inputs or all(p.is_model_input for p in op.inputs):
                if op not in self._input_ops:
                    self._input_ops.append(op)
            
            # 如果操作没有输出消费者，则为输出操作
            if not any(p.consumers for p in op.outputs):
                if op not in self._output_ops:
                    self._output_ops.append(op)
    
    # 公共接口方法
    def get_all_ops(self) -> Dict[str, Operation]:
        """获取所有操作"""
        return self._ops
    
    def get_all_products(self) -> Dict[str, Product]:
        """获取所有产品"""
        return self._products
    
    def get_op_by_name(self, name: str) -> Optional[Operation]:
        """根据名称获取操作"""
        return self._ops.get(name)
    
    def get_product_by_name(self, name: str) -> Optional[Product]:
        """根据名称获取产品"""
        return self._products.get(name)
    
    def get_input_ops(self) -> List[Operation]:
        """获取输入操作列表"""
        return self._input_ops
    
    def get_output_ops(self) -> List[Operation]:
        """获取输出操作列表"""
        return self._output_ops
    
    def get_ordered_ops(self) -> List[Operation]:
        """获取拓扑排序的操作列表"""
        visited = set()
        result = []
        
        def dfs(op: Operation):
            if op in visited:
                return
            visited.add(op)
            
            # 先访问所有输入操作
            for input_op in op.input_ops:
                dfs(input_op)
            
            result.append(op)
        
        # 从输入操作开始DFS
        for input_op in self._input_ops:
            dfs(input_op)
        
        return result
```

### 3.3 步骤3：实现基础工具函数

#### 3.3.1 通用工具函数
```python
# aimet/common/utils.py
import logging
import torch
from typing import Any, Dict, List, Optional, Union

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def validate_bitwidth(bitwidth: int) -> None:
    """验证位宽参数"""
    if not isinstance(bitwidth, int):
        raise TypeError("Bitwidth must be an integer")
    
    if not 1 <= bitwidth <= 32:
        raise ValueError("Bitwidth must be between 1 and 32")

def compute_quantization_range(bitwidth: int, signed: bool = True) -> tuple:
    """计算量化范围"""
    if signed:
        qmin = -(2**(bitwidth - 1))
        qmax = 2**(bitwidth - 1) - 1
    else:
        qmin = 0
        qmax = 2**bitwidth - 1
    
    return qmin, qmax

def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """计算张量统计信息"""
    return {
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'shape': list(tensor.shape),
        'numel': tensor.numel()
    }

def save_dict_to_json(data: Dict[str, Any], filepath: str) -> None:
    """保存字典到JSON文件"""
    import json
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_dict_from_json(filepath: str) -> Dict[str, Any]:
    """从JSON文件加载字典"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
```

## 4. 第二阶段：量化核心实现 (第4-7周)

### 4.1 步骤4：实现编码分析器

#### 4.1.1 基础编码分析器
```python
# aimet/common/encoding_analyzer.py
from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any
from .defs import QuantizationEncoding
from .utils import setup_logger

logger = setup_logger(__name__)

class EncodingAnalyzer(ABC):
    """编码分析器抽象基类"""
    
    def __init__(self):
        self.stats_updated = False
        self._reset_stats()
    
    def _reset_stats(self):
        """重置统计信息"""
        self.stats_updated = False
    
    @abstractmethod
    def update_stats(self, tensor: torch.Tensor) -> None:
        """更新统计信息"""
        pass
    
    @abstractmethod
    def compute_encoding(self, bitwidth: int, use_symmetric: bool) -> QuantizationEncoding:
        """计算编码参数"""
        pass
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._reset_stats()


class MinMaxEncodingAnalyzer(EncodingAnalyzer):
    """最小最大值编码分析器"""
    
    def __init__(self):
        super().__init__()
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def _reset_stats(self):
        super()._reset_stats()
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update_stats(self, tensor: torch.Tensor) -> None:
        """更新最小最大值"""
        current_min = tensor.min().item()
        current_max = tensor.max().item()
        
        self.min_val = min(self.min_val, current_min)
        self.max_val = max(self.max_val, current_max)
        self.stats_updated = True
    
    def compute_encoding(self, bitwidth: int, use_symmetric: bool) -> QuantizationEncoding:
        """计算MinMax编码"""
        if not self.stats_updated:
            raise RuntimeError("No statistics available")
        
        # 确保范围包含零点
        min_val = min(self.min_val, 0.0)
        max_val = max(self.max_val, 0.0)
        
        # 确保最小范围
        if abs(max_val - min_val) < 1e-7:
            max_val = min_val + 1e-7
        
        if use_symmetric:
            max_abs = max(abs(min_val), abs(max_val))
            num_steps = 2**(bitwidth - 1) - 1
            scale = max_abs / num_steps
            offset = -num_steps
            actual_min = -max_abs
            actual_max = max_abs
        else:
            num_steps = 2**bitwidth - 1
            scale = (max_val - min_val) / num_steps
            offset = round(min_val / scale)
            actual_min = offset * scale
            actual_max = actual_min + num_steps * scale
        
        return QuantizationEncoding(
            min=actual_min,
            max=actual_max,
            scale=scale,
            offset=offset,
            bitwidth=bitwidth,
            symmetric=use_symmetric
        )
```

### 4.2 步骤5：实现张量量化器

#### 4.2.1 基础张量量化器
```python
# aimet/torch/tensor_quantizer.py
import torch
import torch.nn as nn
from typing import Optional
from ..common.defs import QuantScheme, QuantizationDataType, QuantizationEncoding
from ..common.encoding_analyzer import MinMaxEncodingAnalyzer
from ..common.utils import setup_logger, validate_bitwidth

logger = setup_logger(__name__)

class TensorQuantizer(nn.Module):
    """张量量化器"""
    
    def __init__(self,
                 bitwidth: int = 8,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf,
                 use_symmetric_encodings: bool = True,
                 enabled: bool = True):
        super().__init__()
        
        validate_bitwidth(bitwidth)
        
        self.bitwidth = bitwidth
        self.quant_scheme = quant_scheme
        self.use_symmetric_encodings = use_symmetric_encodings
        self.enabled = enabled
        
        # 量化参数
        self.register_parameter('scale', nn.Parameter(torch.tensor(1.0), requires_grad=False))
        self.register_parameter('offset', nn.Parameter(torch.tensor(0.0), requires_grad=False))
        
        # 编码分析器
        self.encoding_analyzer = self._create_encoding_analyzer()
        
        # 状态
        self.stats_collection_mode = False
        self.encoding_computed = False
        
        # 量化范围
        self.qmin, self.qmax = self._compute_quantization_range()
    
    def _create_encoding_analyzer(self):
        """创建编码分析器"""
        if self.quant_scheme in [QuantScheme.post_training_tf, QuantScheme.min_max]:
            return MinMaxEncodingAnalyzer()
        else:
            raise ValueError(f"Unsupported quantization scheme: {self.quant_scheme}")
    
    def _compute_quantization_range(self):
        """计算量化范围"""
        if self.use_symmetric_encodings:
            qmax = 2**(self.bitwidth - 1) - 1
            qmin = -qmax
        else:
            qmin = 0
            qmax = 2**self.bitwidth - 1
        
        return qmin, qmax
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if not self.enabled:
            return input_tensor
        
        if self.stats_collection_mode:
            # 统计收集模式
            self.encoding_analyzer.update_stats(input_tensor)
            return input_tensor
        else:
            # 推理模式
            if not self.encoding_computed:
                logger.warning("Encoding not computed, returning original tensor")
                return input_tensor
            
            return self.quantize_dequantize(input_tensor)
    
    def quantize_dequantize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """量化后立即反量化"""
        # 量化: q = clamp(round(x / scale) + offset, qmin, qmax)
        quantized = torch.clamp(
            torch.round(input_tensor / self.scale) + self.offset,
            self.qmin, self.qmax
        )
        
        # 反量化: x_dq = (q - offset) * scale
        dequantized = (quantized - self.offset) * self.scale
        
        return dequantized
    
    def compute_encoding(self) -> None:
        """计算编码参数"""
        if not self.encoding_analyzer.stats_updated:
            raise RuntimeError("No statistics collected")
        
        encoding = self.encoding_analyzer.compute_encoding(
            self.bitwidth, self.use_symmetric_encodings
        )
        
        # 更新参数
        self.scale.data.fill_(encoding.scale)
        self.offset.data.fill_(encoding.offset)
        
        self.encoding_computed = True
        self._encoding = encoding
        
        logger.debug(f"Computed encoding: scale={encoding.scale:.6f}, offset={encoding.offset}")
    
    def set_stats_collection_mode(self, enabled: bool) -> None:
        """设置统计收集模式"""
        self.stats_collection_mode = enabled
    
    def reset_encoding_stats(self) -> None:
        """重置编码统计"""
        self.encoding_analyzer.reset_stats()
        self.encoding_computed = False
    
    def get_encoding(self) -> Optional[QuantizationEncoding]:
        """获取编码"""
        if not self.encoding_computed:
            return None
        return self._encoding
    
    def set_encoding(self, encoding: QuantizationEncoding) -> None:
        """设置编码"""
        self.scale.data.fill_(encoding.scale)
        self.offset.data.fill_(encoding.offset)
        self._encoding = encoding
        self.encoding_computed = True
```

### 4.3 步骤6：实现量化包装器

#### 4.3.1 量化包装器实现
```python
# aimet/torch/quantization_wrapper.py
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
from .tensor_quantizer import TensorQuantizer
from ..common.defs import QuantScheme
from ..common.utils import setup_logger

logger = setup_logger(__name__)

class QuantizationWrapper(nn.Module):
    """量化包装器"""
    
    def __init__(self,
                 original_module: nn.Module,
                 module_name: str,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf,
                 output_bitwidth: int = 8,
                 param_bitwidth: int = 8):
        super().__init__()
        
        self.original_module = original_module
        self.module_name = module_name
        self.quant_scheme = quant_scheme
        self.output_bitwidth = output_bitwidth
        self.param_bitwidth = param_bitwidth
        
        # 量化器
        self.input_quantizers = nn.ModuleList()
        self.output_quantizers = nn.ModuleList()
        self.param_quantizers = nn.ModuleDict()
        
        # 创建量化器
        self._create_quantizers()
    
    def _create_quantizers(self):
        """创建量化器"""
        # 输入量化器（假设单输入）
        input_quantizer = TensorQuantizer(
            bitwidth=self.output_bitwidth,
            quant_scheme=self.quant_scheme
        )
        self.input_quantizers.append(input_quantizer)
        
        # 输出量化器（假设单输出）
        output_quantizer = TensorQuantizer(
            bitwidth=self.output_bitwidth,
            quant_scheme=self.quant_scheme
        )
        self.output_quantizers.append(output_quantizer)
        
        # 参数量化器
        for param_name, param in self.original_module.named_parameters():
            param_quantizer = TensorQuantizer(
                bitwidth=self.param_bitwidth,
                quant_scheme=self.quant_scheme
            )
            self.param_quantizers[param_name] = param_quantizer
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        # 量化输入
        quantized_inputs = []
        for i, input_tensor in enumerate(args):
            if i < len(self.input_quantizers):
                quantized_inputs.append(self.input_quantizers[i](input_tensor))
            else:
                quantized_inputs.append(input_tensor)
        
        # 量化参数
        self._quantize_parameters()
        
        # 执行原始模块
        output = self.original_module(*quantized_inputs, **kwargs)
        
        # 量化输出
        if isinstance(output, torch.Tensor) and len(self.output_quantizers) > 0:
            output = self.output_quantizers[0](output)
        
        return output
    
    def _quantize_parameters(self):
        """量化参数"""
        for param_name, param in self.original_module.named_parameters():
            if param_name in self.param_quantizers:
                quantizer = self.param_quantizers[param_name]
                if quantizer.enabled and quantizer.encoding_computed:
                    # 使用straight-through estimator
                    with torch.no_grad():
                        quantized_param = quantizer.quantize_dequantize(param.data)
                        param.data.copy_(quantized_param)
    
    def compute_encodings(self):
        """计算编码"""
        for quantizer in self.input_quantizers:
            if quantizer.encoding_analyzer.stats_updated:
                quantizer.compute_encoding()
        
        for quantizer in self.output_quantizers:
            if quantizer.encoding_analyzer.stats_updated:
                quantizer.compute_encoding()
        
        for quantizer in self.param_quantizers.values():
            if quantizer.encoding_analyzer.stats_updated:
                quantizer.compute_encoding()
    
    def set_encoding_computation_mode(self, enabled: bool):
        """设置编码计算模式"""
        for quantizer in self.input_quantizers:
            quantizer.set_stats_collection_mode(enabled)
        
        for quantizer in self.output_quantizers:
            quantizer.set_stats_collection_mode(enabled)
        
        for quantizer in self.param_quantizers.values():
            quantizer.set_stats_collection_mode(enabled)
    
    def reset_encoding_stats(self):
        """重置编码统计"""
        for quantizer in self.input_quantizers:
            quantizer.reset_encoding_stats()
        
        for quantizer in self.output_quantizers:
            quantizer.reset_encoding_stats()
        
        for quantizer in self.param_quantizers.values():
            quantizer.reset_encoding_stats()
```

## 5. 第三阶段：量化仿真模型 (第8-10周)

### 5.1 步骤7：实现量化仿真模型

#### 5.1.1 量化仿真模型主类
```python
# aimet/torch/quantsim.py
import torch
import torch.nn as nn
from typing import Callable, Any, Dict, Optional, List
from ..common.connected_graph.connectedgraph import ConnectedGraph
from ..common.defs import QuantScheme
from ..common.utils import setup_logger
from .quantization_wrapper import QuantizationWrapper

logger = setup_logger(__name__)

class QuantizationSimModel:
    """量化仿真模型"""
    
    def __init__(self,
                 model: nn.Module,
                 dummy_input: torch.Tensor,
                 quant_scheme: QuantScheme = QuantScheme.post_training_tf,
                 default_output_bw: int = 8,
                 default_param_bw: int = 8):
        
        self.model = model
        self.dummy_input = dummy_input
        self.quant_scheme = quant_scheme
        self.default_output_bw = default_output_bw
        self.default_param_bw = default_param_bw
        
        # 核心组件
        self.connected_graph = None
        self.quantization_wrappers: Dict[str, QuantizationWrapper] = {}
        
        # 状态
        self._encoding_computation_mode = False
        self._encodings_computed = False
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化系统"""
        logger.info("Initializing quantization simulation model...")
        
        # 创建连接图
        self.connected_graph = ConnectedGraph(self.model, self.dummy_input)
        
        # 创建量化包装器
        self._create_quantization_wrappers()
        
        logger.info(f"Created {len(self.quantization_wrappers)} quantization wrappers")
    
    def _create_quantization_wrappers(self):
        """创建量化包装器"""
        quantizable_ops = self._identify_quantizable_operations()
        
        for op in quantizable_ops:
            wrapper = QuantizationWrapper(
                original_module=op.module,
                module_name=op.name,
                quant_scheme=self.quant_scheme,
                output_bitwidth=self.default_output_bw,
                param_bitwidth=self.default_param_bw
            )
            
            self.quantization_wrappers[op.name] = wrapper
            self._replace_module_with_wrapper(op.name, wrapper)
    
    def _identify_quantizable_operations(self):
        """识别可量化的操作"""
        quantizable_ops = []
        
        for op in self.connected_graph.get_all_ops().values():
            if self._is_quantizable_operation(op):
                quantizable_ops.append(op)
        
        return quantizable_ops
    
    def _is_quantizable_operation(self, op):
        """判断操作是否可量化"""
        # 支持的操作类型
        quantizable_types = {
            'Conv1d', 'Conv2d', 'Conv3d',
            'Linear',
            'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'
        }
        
        return op.type in quantizable_types
    
    def _replace_module_with_wrapper(self, module_name: str, wrapper: QuantizationWrapper):
        """用包装器替换模块"""
        path_parts = module_name.split('.')
        parent_module = self.model
        
        # 导航到父模块
        for part in path_parts[:-1]:
            parent_module = getattr(parent_module, part)
        
        # 替换模块
        setattr(parent_module, path_parts[-1], wrapper)
    
    def compute_encodings(self,
                         forward_pass_callback: Callable,
                         forward_pass_callback_args: Any = None) -> None:
        """计算编码"""
        logger.info("Computing quantization encodings...")
        
        try:
            # 设置编码计算模式
            self._set_encoding_computation_mode(True)
            
            # 重置统计信息
            self._reset_all_encoding_stats()
            
            # 执行前向传播收集统计
            self._collect_statistics(forward_pass_callback, forward_pass_callback_args)
            
            # 计算编码
            self._compute_all_encodings()
            
            self._encodings_computed = True
            logger.info("Encoding computation completed successfully")
            
        except Exception as e:
            logger.error(f"Encoding computation failed: {str(e)}")
            raise
        finally:
            self._set_encoding_computation_mode(False)
    
    def _collect_statistics(self, callback: Callable, callback_args: Any):
        """收集统计信息"""
        original_mode = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                if callback_args is not None:
                    callback(self.model, callback_args)
                else:
                    callback(self.model)
        finally:
            self.model.train(original_mode)
    
    def _compute_all_encodings(self):
        """计算所有编码"""
        for wrapper_name, wrapper in self.quantization_wrappers.items():
            try:
                wrapper.compute_encodings()
            except Exception as e:
                logger.error(f"Failed to compute encodings for {wrapper_name}: {str(e)}")
                raise
    
    def _set_encoding_computation_mode(self, enabled: bool):
        """设置编码计算模式"""
        self._encoding_computation_mode = enabled
        
        for wrapper in self.quantization_wrappers.values():
            wrapper.set_encoding_computation_mode(enabled)
    
    def _reset_all_encoding_stats(self):
        """重置所有编码统计"""
        for wrapper in self.quantization_wrappers.values():
            wrapper.reset_encoding_stats()
    
    def export(self, path: str, filename_prefix: str) -> str:
        """导出模型"""
        if not self._encodings_computed:
            raise RuntimeError("Encodings must be computed before export")
        
        # 导出量化模型
        model_path = f"{path}/{filename_prefix}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # 导出编码信息
        encodings = self._collect_encodings()
        encoding_path = f"{path}/{filename_prefix}_encodings.json"
        
        import json
        with open(encoding_path, 'w') as f:
            json.dump(encodings, f, indent=2, default=str)
        
        logger.info(f"Model exported to {model_path}")
        logger.info(f"Encodings exported to {encoding_path}")
        
        return model_path
    
    def _collect_encodings(self) -> Dict[str, Any]:
        """收集编码信息"""
        encodings = {}
        
        for wrapper_name, wrapper in self.quantization_wrappers.items():
            wrapper_encodings = {}
            
            # 输入量化器编码
            for i, quantizer in enumerate(wrapper.input_quantizers):
                encoding = quantizer.get_encoding()
                if encoding:
                    wrapper_encodings[f'input_{i}'] = {
                        'min': encoding.min,
                        'max': encoding.max,
                        'scale': encoding.scale,
                        'offset': encoding.offset,
                        'bitwidth': encoding.bitwidth,
                        'symmetric': encoding.symmetric
                    }
            
            # 输出量化器编码
            for i, quantizer in enumerate(wrapper.output_quantizers):
                encoding = quantizer.get_encoding()
                if encoding:
                    wrapper_encodings[f'output_{i}'] = {
                        'min': encoding.min,
                        'max': encoding.max,
                        'scale': encoding.scale,
                        'offset': encoding.offset,
                        'bitwidth': encoding.bitwidth,
                        'symmetric': encoding.symmetric
                    }
            
            # 参数量化器编码
            for param_name, quantizer in wrapper.param_quantizers.items():
                encoding = quantizer.get_encoding()
                if encoding:
                    wrapper_encodings[f'param_{param_name}'] = {
                        'min': encoding.min,
                        'max': encoding.max,
                        'scale': encoding.scale,
                        'offset': encoding.offset,
                        'bitwidth': encoding.bitwidth,
                        'symmetric': encoding.symmetric
                    }
            
            if wrapper_encodings:
                encodings[wrapper_name] = wrapper_encodings
        
        return encodings
    
    def __call__(self, *args, **kwargs):
        """使模型可调用"""
        return self.model(*args, **kwargs)
```

## 6. 第四阶段：测试和验证 (第11-12周)

### 6.1 步骤8：编写单元测试

#### 6.1.1 测试TensorQuantizer
```python
# aimet/tests/test_torch/test_tensor_quantizer.py
import unittest
import torch
from aimet.torch.tensor_quantizer import TensorQuantizer
from aimet.common.defs import QuantScheme

class TestTensorQuantizer(unittest.TestCase):
    
    def setUp(self):
        """测试准备"""
        self.quantizer = TensorQuantizer(
            bitwidth=8,
            quant_scheme=QuantScheme.post_training_tf,
            use_symmetric_encodings=True
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.quantizer.bitwidth, 8)
        self.assertEqual(self.quantizer.quant_scheme, QuantScheme.post_training_tf)
        self.assertTrue(self.quantizer.use_symmetric_encodings)
        self.assertTrue(self.quantizer.enabled)
        self.assertFalse(self.quantizer.encoding_computed)
    
    def test_stats_collection(self):
        """测试统计收集"""
        # 设置统计收集模式
        self.quantizer.set_stats_collection_mode(True)
        
        # 输入数据
        input_tensor = torch.randn(10, 10)
        output = self.quantizer(input_tensor)
        
        # 验证输出等于输入（统计模式下）
        torch.testing.assert_close(output, input_tensor)
        
        # 验证统计信息已更新
        self.assertTrue(self.quantizer.encoding_analyzer.stats_updated)
    
    def test_encoding_computation(self):
        """测试编码计算"""
        # 收集统计信息
        self.quantizer.set_stats_collection_mode(True)
        input_tensor = torch.randn(100, 100)
        self.quantizer(input_tensor)
        
        # 计算编码
        self.quantizer.compute_encoding()
        
        # 验证编码已计算
        self.assertTrue(self.quantizer.encoding_computed)
        
        # 验证编码参数
        encoding = self.quantizer.get_encoding()
        self.assertIsNotNone(encoding)
        self.assertGreater(encoding.scale, 0)
        self.assertEqual(encoding.bitwidth, 8)
    
    def test_quantization(self):
        """测试量化功能"""
        # 准备数据和编码
        input_tensor = torch.randn(50, 50)
        self.quantizer.set_stats_collection_mode(True)
        self.quantizer(input_tensor)
        self.quantizer.compute_encoding()
        
        # 执行量化
        self.quantizer.set_stats_collection_mode(False)
        quantized_output = self.quantizer(input_tensor)
        
        # 验证输出形状
        self.assertEqual(quantized_output.shape, input_tensor.shape)
        
        # 验证量化误差在合理范围内
        error = torch.abs(input_tensor - quantized_output)
        max_error = error.max().item()
        self.assertLess(max_error, 1.0)  # 根据具体情况调整阈值
    
    def test_encoding_persistence(self):
        """测试编码持久化"""
        # 创建编码
        from aimet.common.defs import QuantizationEncoding
        encoding = QuantizationEncoding(
            min=-1.0, max=1.0, scale=0.1, offset=0, bitwidth=8, symmetric=True
        )
        
        # 设置编码
        self.quantizer.set_encoding(encoding)
        
        # 验证编码
        retrieved_encoding = self.quantizer.get_encoding()
        self.assertAlmostEqual(retrieved_encoding.scale, 0.1)
        self.assertEqual(retrieved_encoding.bitwidth, 8)
        self.assertTrue(retrieved_encoding.symmetric)

if __name__ == '__main__':
    unittest.main()
```

#### 6.1.2 测试QuantizationSimModel
```python
# aimet/tests/test_torch/test_quantsim.py
import unittest
import torch
import torch.nn as nn
from aimet.torch.quantsim import QuantizationSimModel
from aimet.common.defs import QuantScheme

class SimpleModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TestQuantizationSimModel(unittest.TestCase):
    
    def setUp(self):
        """测试准备"""
        self.model = SimpleModel()
        self.dummy_input = torch.randn(1, 3, 32, 32)
        self.sim = QuantizationSimModel(
            model=self.model,
            dummy_input=self.dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_output_bw=8,
            default_param_bw=8
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.sim.connected_graph)
        self.assertGreater(len(self.sim.quantization_wrappers), 0)
        
        # 验证包装器创建
        expected_wrappers = ['conv1', 'conv2', 'fc']  # 预期的量化模块
        for wrapper_name in expected_wrappers:
            self.assertIn(wrapper_name, self.sim.quantization_wrappers)
    
    def test_forward_pass(self):
        """测试前向传播"""
        output = self.sim(self.dummy_input)
        self.assertEqual(output.shape, (1, 10))
    
    def test_encoding_computation(self):
        """测试编码计算"""
        def forward_pass_callback(model, args=None):
            """前向传播回调"""
            with torch.no_grad():
                model(self.dummy_input)
        
        # 计算编码
        self.sim.compute_encodings(forward_pass_callback)
        
        # 验证编码已计算
        self.assertTrue(self.sim._encodings_computed)
        
        # 验证每个包装器的编码
        for wrapper in self.sim.quantization_wrappers.values():
            for quantizer in wrapper.input_quantizers:
                if quantizer.encoding_analyzer.stats_updated:
                    self.assertTrue(quantizer.encoding_computed)
    
    def test_export(self):
        """测试模型导出"""
        import tempfile
        import os
        
        # 计算编码
        def forward_pass_callback(model, args=None):
            model(self.dummy_input)
        
        self.sim.compute_encodings(forward_pass_callback)
        
        # 导出模型
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = self.sim.export(temp_dir, "test_model")
            
            # 验证文件存在
            self.assertTrue(os.path.exists(model_path))
            encoding_path = f"{temp_dir}/test_model_encodings.json"
            self.assertTrue(os.path.exists(encoding_path))

if __name__ == '__main__':
    unittest.main()
```

### 6.2 步骤9：集成测试和示例

#### 6.2.1 端到端测试
```python
# aimet/tests/test_integration/test_end_to_end.py
import unittest
import torch
import torch.nn as nn
import torchvision.models as models
from aimet.torch.quantsim import QuantizationSimModel
from aimet.common.defs import QuantScheme

class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""
    
    def test_resnet_quantization(self):
        """测试ResNet量化"""
        # 创建预训练模型
        model = models.resnet18(pretrained=False)
        model.eval()
        
        # 创建虚拟输入
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 创建量化仿真模型
        sim = QuantizationSimModel(
            model=model,
            dummy_input=dummy_input,
            quant_scheme=QuantScheme.post_training_tf,
            default_output_bw=8,
            default_param_bw=8
        )
        
        # 准备校准数据
        calibration_data = [torch.randn(4, 3, 224, 224) for _ in range(10)]
        
        def calibration_callback(model, args=None):
            """校准回调函数"""
            for batch in calibration_data:
                with torch.no_grad():
                    model(batch)
        
        # 计算编码
        sim.compute_encodings(calibration_callback)
        
        # 测试推理
        with torch.no_grad():
            original_output = model(dummy_input)
            quantized_output = sim(dummy_input)
        
        # 验证输出形状一致
        self.assertEqual(original_output.shape, quantized_output.shape)
        
        # 验证输出数值在合理范围内
        diff = torch.abs(original_output - quantized_output)
        max_diff = diff.max().item()
        self.assertLess(max_diff, 10.0)  # 根据具体情况调整
        
        print(f"Max difference: {max_diff:.4f}")
        print(f"Mean difference: {diff.mean().item():.4f}")
```

#### 6.2.2 使用示例
```python
# examples/basic_quantization_example.py
"""基础量化示例"""

import torch
import torch.nn as nn
from aimet.torch.quantsim import QuantizationSimModel
from aimet.common.defs import QuantScheme

class SimpleModel(nn.Module):
    """简单示例模型"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    """主函数"""
    print("AIMET Basic Quantization Example")
    print("=" * 40)
    
    # 1. 创建模型
    model = SimpleModel()
    model.eval()
    print("✓ Model created")
    
    # 2. 准备虚拟输入
    dummy_input = torch.randn(1, 3, 32, 32)
    print("✓ Dummy input prepared")
    
    # 3. 创建量化仿真模型
    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf,
        default_output_bw=8,
        default_param_bw=8
    )
    print(f"✓ Quantization sim model created with {len(sim.quantization_wrappers)} wrappers")
    
    # 4. 准备校准数据
    calibration_data = [torch.randn(2, 3, 32, 32) for _ in range(5)]
    print("✓ Calibration data prepared")
    
    # 5. 定义校准回调函数
    def calibration_callback(model, args=None):
        """校准数据前向传播"""
        for batch in calibration_data:
            with torch.no_grad():
                model(batch)
    
    # 6. 计算量化编码
    print("Computing quantization encodings...")
    sim.compute_encodings(calibration_callback)
    print("✓ Encodings computed successfully")
    
    # 7. 测试量化效果
    test_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        # 原始模型输出
        original_output = model(test_input)
        
        # 量化模型输出
        quantized_output = sim(test_input)
    
    # 8. 分析结果
    diff = torch.abs(original_output - quantized_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("\nQuantization Results:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Original output range: [{original_output.min().item():.6f}, {original_output.max().item():.6f}]")
    print(f"  Quantized output range: [{quantized_output.min().item():.6f}, {quantized_output.max().item():.6f}]")
    
    # 9. 导出模型
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = sim.export(temp_dir, "quantized_model")
        print(f"✓ Model exported to {model_path}")
    
    print("\n✓ Example completed successfully!")

if __name__ == "__main__":
    main()
```

## 7. 第五阶段：文档和部署 (第13-14周)

### 7.1 步骤10：创建文档

#### 7.1.1 API文档
```python
# docs/generate_docs.py
"""生成API文档脚本"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Sphinx配置
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

# 项目信息
project = 'AIMET Implementation'
copyright = '2024, AIMET Team'
author = 'AIMET Team'

# 主题配置
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# 自动文档配置
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
```

### 7.2 步骤11：性能优化

#### 7.2.1 CUDA加速（可选）
```python
# aimet/torch/quantization_cuda.py (可选)
"""CUDA加速量化操作"""

import torch

def quantize_dequantize_cuda(input_tensor, scale, offset, qmin, qmax):
    """CUDA加速的量化-反量化操作"""
    if not torch.cuda.is_available() or not input_tensor.is_cuda:
        # 回退到CPU实现
        quantized = torch.clamp(
            torch.round(input_tensor / scale) + offset,
            qmin, qmax
        )
        return (quantized - offset) * scale
    
    # CUDA实现（需要编写CUDA内核）
    # 这里使用PyTorch的内置操作作为示例
    with torch.cuda.device(input_tensor.device):
        quantized = torch.clamp(
            torch.round(input_tensor / scale) + offset,
            qmin, qmax
        )
        return (quantized - offset) * scale
```

## 8. 验证和测试策略

### 8.1 测试金字塔

```
    ┌─────────────────┐
    │   集成测试       │  ← 端到端测试，真实模型
    ├─────────────────┤
    │   组件测试       │  ← 模块间交互测试
    ├─────────────────┤
    │   单元测试       │  ← 单个类/函数测试
    └─────────────────┘
```

### 8.2 测试运行脚本
```bash
#!/bin/bash
# run_tests.sh

echo "Running AIMET Implementation Tests"
echo "================================="

# 运行单元测试
echo "Running unit tests..."
python -m pytest aimet/tests/test_common/ -v
python -m pytest aimet/tests/test_torch/ -v

# 运行集成测试
echo "Running integration tests..."
python -m pytest aimet/tests/test_integration/ -v

# 运行示例
echo "Running examples..."
python examples/basic_quantization_example.py

# 生成覆盖率报告
echo "Generating coverage report..."
python -m pytest --cov=aimet --cov-report=html

echo "All tests completed!"
```

## 9. 部署和发布

### 9.1 打包脚本
```python
# scripts/build_package.py
"""构建发布包脚本"""

import subprocess
import sys
import os

def build_package():
    """构建Python包"""
    print("Building AIMET package...")
    
    # 清理旧的构建文件
    subprocess.run([sys.executable, "setup.py", "clean", "--all"])
    
    # 构建源码包
    subprocess.run([sys.executable, "setup.py", "sdist"])
    
    # 构建wheel包
    subprocess.run([sys.executable, "setup.py", "bdist_wheel"])
    
    print("Package built successfully!")
    print("Distribution files are in the 'dist/' directory")

if __name__ == "__main__":
    build_package()
```

## 10. 总结和后续步骤

### 10.1 实现检查清单

- [ ] 基础框架搭建完成
- [ ] 连接图模块实现
- [ ] 编码分析器实现
- [ ] 张量量化器实现
- [ ] 量化包装器实现
- [ ] 量化仿真模型实现
- [ ] 单元测试编写
- [ ] 集成测试编写
- [ ] 文档编写
- [ ] 示例代码编写
- [ ] 性能优化
- [ ] 打包和发布

### 10.2 后续扩展方向

1. **高级量化算法**
   - AdaRound实现
   - TF-Enhanced优化
   - 百分位量化

2. **框架扩展**
   - ONNX支持
   - TensorFlow支持

3. **性能优化**
   - CUDA加速
   - 多线程支持
   - 内存优化

4. **工具链完善**
   - 可视化工具
   - 自动调优
   - 模型分析

通过按照这个详细的实现指南，新手开发者可以逐步构建出一个功能完整的AIMET量化工具包。每个步骤都提供了具体的代码实现和测试方法，确保开发过程的可控性和质量保证。