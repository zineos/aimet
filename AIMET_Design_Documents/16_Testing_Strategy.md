# AIMET 测试策略文档

## 1. 测试概述

### 1.1 测试目标
本文档定义了AIMET系统的全面测试策略，确保系统的正确性、性能和可靠性。测试策略涵盖了从单元测试到端到端集成测试的各个层面。

### 1.2 测试原则
- **全面性**: 覆盖所有核心功能和边界条件
- **自动化**: 所有测试可自动执行和验证
- **可重复性**: 测试结果一致且可重现
- **快速反馈**: 快速发现和定位问题
- **质量保证**: 确保代码质量和功能正确性

## 2. 测试金字塔

### 2.1 测试层次结构
```
        ┌─────────────────┐
        │   端到端测试     │  ← 完整工作流程测试
        │   (E2E Tests)   │
        ├─────────────────┤
        │   集成测试       │  ← 模块间交互测试  
        │ (Integration)   │
        ├─────────────────┤
        │   组件测试       │  ← 单个组件功能测试
        │ (Component)     │
        ├─────────────────┤
        │   单元测试       │  ← 函数/类级别测试
        │  (Unit Tests)   │
        └─────────────────┘
```

### 2.2 测试分布比例
- **单元测试**: 70% - 快速、隔离、覆盖基础逻辑
- **组件测试**: 20% - 验证模块功能完整性
- **集成测试**: 8% - 验证模块间协作
- **端到端测试**: 2% - 验证完整用户场景

## 3. 单元测试 (Unit Tests)

### 3.1 测试范围
单元测试覆盖所有核心类和函数的基础功能。

#### 3.1.1 TensorQuantizer单元测试
```python
# tests/unit/test_tensor_quantizer.py
import unittest
import torch
import numpy as np
from aimet.torch.tensor_quantizer import TensorQuantizer
from aimet.common.defs import QuantScheme, QuantizationEncoding

class TestTensorQuantizer(unittest.TestCase):
    """TensorQuantizer单元测试"""
    
    def setUp(self):
        """测试准备"""
        self.quantizer = TensorQuantizer(
            bitwidth=8,
            quant_scheme=QuantScheme.post_training_tf,
            use_symmetric_encodings=True
        )
        self.test_tensor = torch.randn(10, 10)
    
    def test_initialization(self):
        """测试初始化参数"""
        self.assertEqual(self.quantizer.bitwidth, 8)
        self.assertTrue(self.quantizer.use_symmetric_encodings)
        self.assertTrue(self.quantizer.enabled)
        self.assertFalse(self.quantizer.encoding_computed)
    
    def test_stats_collection_mode(self):
        """测试统计收集模式"""
        # 启用统计收集
        self.quantizer.set_stats_collection_mode(True)
        self.assertTrue(self.quantizer.stats_collection_mode)
        
        # 前向传播应该收集统计信息
        output = self.quantizer(self.test_tensor)
        torch.testing.assert_close(output, self.test_tensor)
        self.assertTrue(self.quantizer.encoding_analyzer.stats_updated)
        
        # 禁用统计收集
        self.quantizer.set_stats_collection_mode(False)
        self.assertFalse(self.quantizer.stats_collection_mode)
    
    def test_encoding_computation(self):
        """测试编码计算"""
        # 先收集统计信息
        self.quantizer.set_stats_collection_mode(True)
        self.quantizer(self.test_tensor)
        
        # 计算编码
        self.quantizer.compute_encoding()
        
        # 验证编码已计算
        self.assertTrue(self.quantizer.encoding_computed)
        encoding = self.quantizer.get_encoding()
        self.assertIsNotNone(encoding)
        self.assertGreater(encoding.scale, 0)
        self.assertEqual(encoding.bitwidth, 8)
    
    def test_quantization_accuracy(self):
        """测试量化精度"""
        # 准备已知分布的数据
        test_data = torch.linspace(-2.0, 2.0, 1000)
        
        # 收集统计信息并计算编码
        self.quantizer.set_stats_collection_mode(True)
        self.quantizer(test_data)
        self.quantizer.compute_encoding()
        
        # 执行量化
        self.quantizer.set_stats_collection_mode(False)
        quantized_data = self.quantizer(test_data)
        
        # 验证量化误差在合理范围内
        error = torch.abs(test_data - quantized_data)
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        # 对于8位量化，最大误差应该小于量化步长的一半
        encoding = self.quantizer.get_encoding()
        expected_max_error = encoding.scale / 2
        
        self.assertLess(max_error, expected_max_error * 1.1)  # 允许10%的误差
        self.assertLess(mean_error, expected_max_error * 0.5)
    
    def test_encoding_persistence(self):
        """测试编码持久化"""
        # 创建自定义编码
        custom_encoding = QuantizationEncoding(
            min=-1.0, max=1.0, scale=0.1, offset=0, bitwidth=8, symmetric=True
        )
        
        # 设置编码
        self.quantizer.set_encoding(custom_encoding)
        
        # 验证编码设置成功
        retrieved_encoding = self.quantizer.get_encoding()
        self.assertAlmostEqual(retrieved_encoding.scale, 0.1, places=6)
        self.assertEqual(retrieved_encoding.bitwidth, 8)
        self.assertTrue(retrieved_encoding.symmetric)
    
    def test_disabled_quantizer(self):
        """测试禁用的量化器"""
        self.quantizer.enabled = False
        output = self.quantizer(self.test_tensor)
        torch.testing.assert_close(output, self.test_tensor)
    
    def test_edge_cases(self):
        """测试边界条件"""
        # 测试空张量
        empty_tensor = torch.empty(0)
        with self.assertRaises(ValueError):
            self.quantizer.encoding_analyzer.update_stats(empty_tensor)
        
        # 测试包含NaN的张量
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        # 应该能处理但发出警告
        self.quantizer.encoding_analyzer.update_stats(nan_tensor)
        
        # 测试包含Inf的张量
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        self.quantizer.encoding_analyzer.update_stats(inf_tensor)
```

#### 3.1.2 EncodingAnalyzer单元测试
```python
# tests/unit/test_encoding_analyzer.py
import unittest
import torch
import numpy as np
from aimet.common.encoding_analyzer import (
    MinMaxEncodingAnalyzer, 
    TfEnhancedEncodingAnalyzer,
    PercentileEncodingAnalyzer
)

class TestEncodingAnalyzers(unittest.TestCase):
    """编码分析器单元测试"""
    
    def test_minmax_analyzer(self):
        """测试MinMax分析器"""
        analyzer = MinMaxEncodingAnalyzer()
        
        # 测试数据：正态分布
        data1 = torch.randn(1000) * 2 + 1  # 均值1，标准差2
        data2 = torch.randn(1000) * 3 - 2  # 均值-2，标准差3
        
        # 更新统计信息
        analyzer.update_stats(data1)
        analyzer.update_stats(data2)
        
        # 验证统计信息
        self.assertTrue(analyzer.stats_updated)
        self.assertLess(analyzer.min_val, -5)  # 应该捕获到负值
        self.assertGreater(analyzer.max_val, 5)  # 应该捕获到正值
        
        # 计算对称编码
        symmetric_encoding = analyzer.compute_encoding(8, use_symmetric=True)
        self.assertTrue(symmetric_encoding.symmetric)
        self.assertAlmostEqual(symmetric_encoding.min, -symmetric_encoding.max, places=5)
        
        # 计算非对称编码
        asymmetric_encoding = analyzer.compute_encoding(8, use_symmetric=False)
        self.assertFalse(asymmetric_encoding.symmetric)
        self.assertNotEqual(asymmetric_encoding.min, -asymmetric_encoding.max)
    
    def test_tf_enhanced_analyzer(self):
        """测试TF增强分析器"""
        analyzer = TfEnhancedEncodingAnalyzer(num_bins=1024)
        
        # 使用正态分布数据
        normal_data = torch.randn(10000)
        analyzer.update_stats(normal_data)
        
        # 计算编码
        encoding = analyzer.compute_encoding(8, use_symmetric=False)
        
        # 验证编码质量
        self.assertIsNotNone(encoding)
        self.assertGreater(encoding.scale, 0)
        
        # TF增强应该比MinMax有更好的覆盖率
        coverage_ratio = self._compute_coverage_ratio(normal_data, encoding)
        self.assertGreater(coverage_ratio, 0.99)  # 期望99%以上的数据在量化范围内
    
    def test_percentile_analyzer(self):
        """测试百分位分析器"""
        analyzer = PercentileEncodingAnalyzer(percentile=99.9)
        
        # 创建包含异常值的数据
        normal_data = torch.randn(1000)
        outliers = torch.tensor([100.0, -100.0, 200.0, -200.0])
        data_with_outliers = torch.cat([normal_data, outliers])
        
        analyzer.update_stats(data_with_outliers)
        encoding = analyzer.compute_encoding(8, use_symmetric=True)
        
        # 百分位编码应该忽略异常值
        self.assertLess(abs(encoding.max), 50)  # 最大值应该远小于异常值
        self.assertGreater(abs(encoding.max), 2)  # 但应该大于正常数据范围
    
    def _compute_coverage_ratio(self, data, encoding):
        """计算数据覆盖率"""
        in_range = ((data >= encoding.min) & (data <= encoding.max)).float()
        return in_range.mean().item()
```

### 3.2 测试工具和Mock对象

#### 3.2.1 测试工具类
```python
# tests/utils/test_helpers.py
import torch
import numpy as np
from typing import Tuple, List

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_normal_data(shape: Tuple[int, ...], 
                           mean: float = 0.0, 
                           std: float = 1.0) -> torch.Tensor:
        """生成正态分布数据"""
        return torch.normal(mean, std, shape)
    
    @staticmethod
    def generate_uniform_data(shape: Tuple[int, ...], 
                            low: float = -1.0, 
                            high: float = 1.0) -> torch.Tensor:
        """生成均匀分布数据"""
        return torch.rand(shape) * (high - low) + low
    
    @staticmethod
    def generate_outlier_data(base_data: torch.Tensor, 
                            outlier_ratio: float = 0.01,
                            outlier_magnitude: float = 10.0) -> torch.Tensor:
        """在基础数据中添加异常值"""
        data = base_data.clone()
        num_outliers = int(data.numel() * outlier_ratio)
        outlier_indices = torch.randperm(data.numel())[:num_outliers]
        
        outlier_values = torch.randn(num_outliers) * outlier_magnitude
        data.view(-1)[outlier_indices] = outlier_values
        
        return data

class MockModel(torch.nn.Module):
    """用于测试的模拟模型"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc = torch.nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class QuantizationTestCase(unittest.TestCase):
    """量化测试基类"""
    
    def setUp(self):
        """测试准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MockModel().to(self.device)
        self.dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
    
    def assert_quantization_error_within_bounds(self, 
                                               original: torch.Tensor,
                                               quantized: torch.Tensor,
                                               max_error_ratio: float = 0.1):
        """断言量化误差在合理范围内"""
        error = torch.abs(original - quantized)
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        original_range = original.max().item() - original.min().item()
        max_allowed_error = original_range * max_error_ratio
        
        self.assertLess(max_error, max_allowed_error, 
                       f"Max quantization error {max_error} exceeds threshold {max_allowed_error}")
        
        mean_allowed_error = max_allowed_error * 0.1
        self.assertLess(mean_error, mean_allowed_error,
                       f"Mean quantization error {mean_error} exceeds threshold {mean_allowed_error}")
```

## 4. 组件测试 (Component Tests)

### 4.1 QuantizationWrapper组件测试
```python
# tests/component/test_quantization_wrapper.py
import unittest
import torch
from aimet.torch.quantization_wrapper import QuantizationWrapper
from aimet.common.defs import QuantScheme
from tests.utils.test_helpers import QuantizationTestCase

class TestQuantizationWrapper(QuantizationTestCase):
    """量化包装器组件测试"""
    
    def test_wrapper_creation(self):
        """测试包装器创建"""
        original_module = torch.nn.Conv2d(3, 16, 3)
        wrapper = QuantizationWrapper(
            original_module=original_module,
            module_name='test_conv',
            quant_scheme=QuantScheme.post_training_tf
        )
        
        # 验证包装器结构
        self.assertEqual(wrapper.module_name, 'test_conv')
        self.assertIs(wrapper.original_module, original_module)
        self.assertGreater(len(wrapper.input_quantizers), 0)
        self.assertGreater(len(wrapper.output_quantizers), 0)
        self.assertGreater(len(wrapper.param_quantizers), 0)
    
    def test_forward_pass_without_quantization(self):
        """测试未量化时的前向传播"""
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        wrapper = QuantizationWrapper(conv, 'test_conv')
        
        input_tensor = torch.randn(1, 3, 32, 32)
        
        # 原始输出
        original_output = conv(input_tensor)
        
        # 包装器输出（未量化）
        wrapper_output = wrapper(input_tensor)
        
        # 由于未计算编码，输出应该相同
        torch.testing.assert_close(wrapper_output, original_output, atol=1e-6, rtol=1e-6)
    
    def test_encoding_computation_workflow(self):
        """测试编码计算工作流程"""
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        wrapper = QuantizationWrapper(conv, 'test_conv')
        
        # 1. 设置统计收集模式
        wrapper.set_encoding_computation_mode(True)
        
        # 2. 前向传播收集统计信息
        for _ in range(10):
            input_tensor = torch.randn(2, 3, 32, 32)
            wrapper(input_tensor)
        
        # 3. 计算编码
        wrapper.compute_encodings()
        
        # 4. 验证编码已计算
        for quantizer in wrapper.input_quantizers:
            if quantizer.encoding_analyzer.stats_updated:
                self.assertTrue(quantizer.encoding_computed)
        
        for quantizer in wrapper.output_quantizers:
            if quantizer.encoding_analyzer.stats_updated:
                self.assertTrue(quantizer.encoding_computed)
    
    def test_quantized_forward_pass(self):
        """测试量化前向传播"""
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        wrapper = QuantizationWrapper(conv, 'test_conv')
        
        # 准备和计算编码
        wrapper.set_encoding_computation_mode(True)
        calibration_data = [torch.randn(2, 3, 32, 32) for _ in range(20)]
        for batch in calibration_data:
            wrapper(batch)
        wrapper.compute_encodings()
        
        # 切换到推理模式
        wrapper.set_encoding_computation_mode(False)
        
        # 测试量化推理
        test_input = torch.randn(1, 3, 32, 32)
        original_output = conv(test_input)
        quantized_output = wrapper(test_input)
        
        # 验证输出形状一致
        self.assertEqual(quantized_output.shape, original_output.shape)
        
        # 验证量化误差在合理范围内
        self.assert_quantization_error_within_bounds(original_output, quantized_output)
```

### 4.2 ConnectedGraph组件测试
```python
# tests/component/test_connected_graph.py
import unittest
import torch
from aimet.common.connected_graph.connectedgraph import ConnectedGraph
from tests.utils.test_helpers import MockModel

class TestConnectedGraph(unittest.TestCase):
    """连接图组件测试"""
    
    def setUp(self):
        self.model = MockModel()
        self.dummy_input = torch.randn(1, 3, 32, 32)
        self.graph = ConnectedGraph(self.model, self.dummy_input)
    
    def test_graph_construction(self):
        """测试图构建"""
        ops = self.graph.get_all_ops()
        
        # 验证操作节点创建
        expected_ops = ['conv1', 'conv2', 'fc']
        for op_name in expected_ops:
            self.assertIn(op_name, ops)
            op = ops[op_name]
            self.assertEqual(op.name, op_name)
            self.assertIsNotNone(op.module)
    
    def test_topological_ordering(self):
        """测试拓扑排序"""
        ordered_ops = self.graph.get_ordered_ops()
        
        # 验证操作顺序
        op_names = [op.name for op in ordered_ops]
        
        # conv1应该在conv2之前
        conv1_idx = op_names.index('conv1')
        conv2_idx = op_names.index('conv2')
        self.assertLess(conv1_idx, conv2_idx)
        
        # conv2应该在fc之前
        fc_idx = op_names.index('fc')
        self.assertLess(conv2_idx, fc_idx)
    
    def test_data_flow_analysis(self):
        """测试数据流分析"""
        ops = self.graph.get_all_ops()
        
        # 验证输入输出连接
        conv1_op = ops['conv1']
        conv2_op = ops['conv2']
        
        # conv2应该是conv1的输出消费者之一
        conv1_consumers = [consumer.name for product in conv1_op.outputs 
                          for consumer in product.consumers]
        self.assertIn('conv2', conv1_consumers)
```

## 5. 集成测试 (Integration Tests)

### 5.1 量化仿真模型集成测试
```python
# tests/integration/test_quantsim_integration.py
import unittest
import torch
import torchvision.models as models
from aimet.torch.quantsim import QuantizationSimModel
from aimet.common.defs import QuantScheme
from tests.utils.test_helpers import TestDataGenerator

class TestQuantSimIntegration(unittest.TestCase):
    """量化仿真模型集成测试"""
    
    def test_resnet_quantization_workflow(self):
        """测试ResNet完整量化流程"""
        # 创建模型
        model = models.resnet18(pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 创建量化仿真模型
        sim = QuantizationSimModel(
            model=model,
            dummy_input=dummy_input,
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            default_output_bw=8,
            default_param_bw=8
        )
        
        # 验证初始化
        self.assertGreater(len(sim.quantization_wrappers), 10)  # ResNet18有很多层
        
        # 准备校准数据
        calibration_data = [torch.randn(4, 3, 224, 224) for _ in range(50)]
        
        def calibration_callback(model, args=None):
            for batch in calibration_data:
                with torch.no_grad():
                    model(batch)
        
        # 计算编码
        sim.compute_encodings(calibration_callback)
        
        # 验证编码计算完成
        self.assertTrue(sim._encodings_computed)
        
        # 测试推理
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = model(test_input)
            quantized_output = sim(test_input)
        
        # 验证输出
        self.assertEqual(original_output.shape, quantized_output.shape)
        
        # 计算精度损失
        diff = torch.abs(original_output - quantized_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 对于ResNet18，量化误差应该在合理范围内
        self.assertLess(max_diff, 5.0)
        self.assertLess(mean_diff, 1.0)
        
        print(f"ResNet18 Quantization - Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
    
    def test_different_quantization_schemes(self):
        """测试不同量化方案的对比"""
        model = models.mobilenet_v2(pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        schemes = [
            QuantScheme.post_training_tf,
            QuantScheme.post_training_tf_enhanced,
        ]
        
        results = {}
        
        for scheme in schemes:
            sim = QuantizationSimModel(
                model=model,
                dummy_input=dummy_input,
                quant_scheme=scheme
            )
            
            # 校准
            calibration_data = [torch.randn(2, 3, 224, 224) for _ in range(20)]
            
            def calibration_callback(model, args=None):
                for batch in calibration_data:
                    with torch.no_grad():
                        model(batch)
            
            sim.compute_encodings(calibration_callback)
            
            # 测试
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                original_output = model(test_input)
                quantized_output = sim(test_input)
            
            diff = torch.abs(original_output - quantized_output)
            results[scheme.name] = {
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item()
            }
        
        # 打印结果对比
        for scheme_name, metrics in results.items():
            print(f"{scheme_name}: Max={metrics['max_diff']:.4f}, Mean={metrics['mean_diff']:.4f}")
```

### 5.2 模块间协作测试
```python
# tests/integration/test_module_interaction.py
import unittest
import torch
from aimet.torch.quantsim import QuantizationSimModel
from aimet.torch.tensor_quantizer import TensorQuantizer
from aimet.common.encoding_analyzer import MinMaxEncodingAnalyzer
from tests.utils.test_helpers import MockModel

class TestModuleInteraction(unittest.TestCase):
    """模块间交互测试"""
    
    def test_quantizer_analyzer_interaction(self):
        """测试量化器与分析器的交互"""
        quantizer = TensorQuantizer(bitwidth=8)
        analyzer = quantizer.encoding_analyzer
        
        # 验证分析器类型
        self.assertIsInstance(analyzer, MinMaxEncodingAnalyzer)
        
        # 测试统计收集和编码计算的协作
        quantizer.set_stats_collection_mode(True)
        
        # 多次前向传播
        for _ in range(10):
            data = torch.randn(100)
            quantizer(data)
        
        # 验证统计信息传递
        self.assertTrue(analyzer.stats_updated)
        
        # 计算编码
        quantizer.compute_encoding()
        
        # 验证编码参数传递
        self.assertTrue(quantizer.encoding_computed)
        encoding = quantizer.get_encoding()
        
        # 验证编码参数合理性
        self.assertGreater(encoding.scale, 0)
        self.assertLessEqual(abs(encoding.offset), 2**7)  # 8位量化的偏移范围
    
    def test_sim_wrapper_quantizer_chain(self):
        """测试仿真模型->包装器->量化器的调用链"""
        model = MockModel()
        dummy_input = torch.randn(1, 3, 32, 32)
        
        sim = QuantizationSimModel(model, dummy_input)
        
        # 验证调用链建立
        wrapper_names = list(sim.quantization_wrappers.keys())
        self.assertGreater(len(wrapper_names), 0)
        
        # 选择一个包装器测试
        wrapper = sim.quantization_wrappers[wrapper_names[0]]
        
        # 验证量化器创建
        self.assertGreater(len(wrapper.input_quantizers), 0)
        self.assertGreater(len(wrapper.output_quantizers), 0)
        
        # 测试统计收集模式传播
        sim._set_encoding_computation_mode(True)
        
        # 验证模式传播到量化器
        for quantizer in wrapper.input_quantizers:
            self.assertTrue(quantizer.stats_collection_mode)
        
        for quantizer in wrapper.output_quantizers:
            self.assertTrue(quantizer.stats_collection_mode)
```

## 6. 端到端测试 (E2E Tests)

### 6.1 完整工作流程测试
```python
# tests/e2e/test_complete_workflow.py
import unittest
import torch
import torchvision.models as models
import tempfile
import os
import json
from aimet.torch.quantsim import QuantizationSimModel
from aimet.common.defs import QuantScheme

class TestCompleteWorkflow(unittest.TestCase):
    """端到端完整工作流程测试"""
    
    def test_full_quantization_pipeline(self):
        """测试完整量化流水线"""
        print("Testing complete quantization pipeline...")
        
        # 1. 模型准备
        model = models.resnet18(pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        print("✓ Model prepared")
        
        # 2. 创建量化仿真模型
        sim = QuantizationSimModel(
            model=model,
            dummy_input=dummy_input,
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            default_output_bw=8,
            default_param_bw=8
        )
        
        print(f"✓ QuantSim created with {len(sim.quantization_wrappers)} wrappers")
        
        # 3. 准备校准数据集
        num_calibration_batches = 100
        batch_size = 8
        calibration_data = []
        
        for _ in range(num_calibration_batches):
            batch = torch.randn(batch_size, 3, 224, 224)
            calibration_data.append(batch)
        
        print(f"✓ Calibration data prepared: {num_calibration_batches} batches")
        
        # 4. 定义校准回调
        def calibration_callback(model, args=None):
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(calibration_data):
                    model(batch)
                    if (i + 1) % 20 == 0:
                        print(f"  Processed {i + 1}/{len(calibration_data)} batches")
        
        # 5. 计算编码
        print("Computing quantization encodings...")
        sim.compute_encodings(calibration_callback)
        print("✓ Encodings computed")
        
        # 6. 验证量化效果
        test_batches = [torch.randn(4, 3, 224, 224) for _ in range(10)]
        total_max_diff = 0
        total_mean_diff = 0
        
        print("Evaluating quantization quality...")
        with torch.no_grad():
            for i, test_batch in enumerate(test_batches):
                original_output = model(test_batch)
                quantized_output = sim(test_batch)
                
                diff = torch.abs(original_output - quantized_output)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                total_max_diff = max(total_max_diff, max_diff)
                total_mean_diff += mean_diff
        
        avg_mean_diff = total_mean_diff / len(test_batches)
        
        print(f"✓ Quantization quality - Max diff: {total_max_diff:.4f}, Avg mean diff: {avg_mean_diff:.4f}")
        
        # 7. 导出模型
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path, encoding_path = sim.export(temp_dir, 'resnet18_quantized', dummy_input)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(encoding_path))
            
            # 验证编码文件内容
            with open(encoding_path, 'r') as f:
                encodings = json.load(f)
            
            self.assertGreater(len(encodings), 0)
            print(f"✓ Model exported to {temp_dir}")
        
        # 8. 性能基准验证
        self.assertLess(total_max_diff, 10.0, "Max quantization error too large")
        self.assertLess(avg_mean_diff, 2.0, "Average quantization error too large")
        
        print("✓ Complete workflow test passed!")
    
    def test_model_size_comparison(self):
        """测试模型大小对比"""
        model = models.mobilenet_v2(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 原始模型大小
        original_size = sum(p.numel() * 4 for p in model.parameters())  # 假设float32
        
        # 量化模型
        sim = QuantizationSimModel(model, dummy_input)
        
        # 简单校准
        calibration_data = [torch.randn(2, 3, 224, 224) for _ in range(10)]
        
        def calibration_callback(model, args=None):
            for batch in calibration_data:
                with torch.no_grad():
                    model(batch)
        
        sim.compute_encodings(calibration_callback)
        
        # 计算量化后的理论大小（8位参数）
        quantized_size = sum(p.numel() for p in model.parameters())  # 假设int8
        
        compression_ratio = original_size / quantized_size
        
        print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # 验证压缩效果
        self.assertGreater(compression_ratio, 3.5)  # 期望至少3.5x压缩
```

## 7. 性能测试

### 7.1 基准测试
```python
# tests/performance/test_benchmarks.py
import unittest
import time
import torch
import torchvision.models as models
from aimet.torch.quantsim import QuantizationSimModel
from aimet.torch.tensor_quantizer import TensorQuantizer

class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试"""
    
    def test_quantization_overhead(self):
        """测试量化开销"""
        model = models.resnet18(pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 原始模型推理时间
        with torch.no_grad():
            # 预热
            for _ in range(10):
                model(dummy_input)
            
            # 测量
            start_time = time.time()
            for _ in range(100):
                model(dummy_input)
            original_time = time.time() - start_time
        
        # 量化模型推理时间
        sim = QuantizationSimModel(model, dummy_input)
        
        # 简单校准
        sim.compute_encodings(lambda m: m(dummy_input))
        
        with torch.no_grad():
            # 预热
            for _ in range(10):
                sim(dummy_input)
            
            # 测量
            start_time = time.time()
            for _ in range(100):
                sim(dummy_input)
            quantized_time = time.time() - start_time
        
        overhead = (quantized_time - original_time) / original_time * 100
        
        print(f"Original inference time: {original_time:.4f}s")
        print(f"Quantized inference time: {quantized_time:.4f}s")
        print(f"Overhead: {overhead:.2f}%")
        
        # 量化开销应该在合理范围内
        self.assertLess(overhead, 50)  # 不超过50%开销
    
    def test_encoding_computation_time(self):
        """测试编码计算时间"""
        quantizer = TensorQuantizer(bitwidth=8)
        
        # 测试不同数据量的编码计算时间
        data_sizes = [1000, 10000, 100000, 1000000]
        
        for size in data_sizes:
            data = torch.randn(size)
            
            quantizer.set_stats_collection_mode(True)
            quantizer.reset_encoding_stats()
            
            # 统计收集时间
            start_time = time.time()
            quantizer(data)
            stats_time = time.time() - start_time
            
            # 编码计算时间
            start_time = time.time()
            quantizer.compute_encoding()
            encoding_time = time.time() - start_time
            
            print(f"Data size {size}: Stats={stats_time:.4f}s, Encoding={encoding_time:.4f}s")
            
            # 编码计算时间应该是合理的
            self.assertLess(stats_time, 1.0)  # 统计收集不超过1秒
            self.assertLess(encoding_time, 0.1)  # 编码计算不超过0.1秒
```

## 8. 测试自动化和CI/CD

### 8.1 测试运行脚本
```bash
#!/bin/bash
# scripts/run_tests.sh

set -e  # 遇到错误立即退出

echo "AIMET Test Suite"
echo "================"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建测试报告目录
mkdir -p test_reports

echo "Running unit tests..."
python -m pytest tests/unit/ \
    --verbose \
    --cov=aimet \
    --cov-report=html:test_reports/coverage_unit \
    --cov-report=xml:test_reports/coverage_unit.xml \
    --junit-xml=test_reports/unit_tests.xml

echo "Running component tests..."
python -m pytest tests/component/ \
    --verbose \
    --junit-xml=test_reports/component_tests.xml

echo "Running integration tests..."
python -m pytest tests/integration/ \
    --verbose \
    --junit-xml=test_reports/integration_tests.xml

echo "Running E2E tests..."
python -m pytest tests/e2e/ \
    --verbose \
    --junit-xml=test_reports/e2e_tests.xml

echo "Running performance tests..."
python -m pytest tests/performance/ \
    --verbose \
    --junit-xml=test_reports/performance_tests.xml

# 生成综合覆盖率报告
python -m pytest tests/ \
    --cov=aimet \
    --cov-report=html:test_reports/coverage_all \
    --cov-report=xml:test_reports/coverage_all.xml \
    --quiet

echo "All tests completed successfully!"
echo "Test reports available in test_reports/ directory"
```

### 8.2 CI/CD配置
```yaml
# .github/workflows/test.yml
name: AIMET Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 aimet/ --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check aimet/
    
    - name: Run unit tests
      run: |
        python -m pytest tests/unit/ --cov=aimet --cov-report=xml
    
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## 9. 测试数据管理

### 9.1 测试数据集
```python
# tests/data/test_datasets.py
import torch
import os
from typing import List, Tuple

class TestDatasets:
    """测试数据集管理"""
    
    @staticmethod
    def get_imagenet_sample() -> Tuple[torch.Tensor, torch.Tensor]:
        """获取ImageNet样本数据"""
        # 模拟ImageNet数据
        images = torch.randn(10, 3, 224, 224)
        labels = torch.randint(0, 1000, (10,))
        return images, labels
    
    @staticmethod
    def get_calibration_data(model_type: str, num_batches: int = 100) -> List[torch.Tensor]:
        """获取校准数据"""
        if model_type == 'resnet':
            return [torch.randn(8, 3, 224, 224) for _ in range(num_batches)]
        elif model_type == 'mobilenet':
            return [torch.randn(16, 3, 224, 224) for _ in range(num_batches)]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def save_test_data(data: torch.Tensor, filename: str) -> None:
        """保存测试数据"""
        os.makedirs('tests/data/saved', exist_ok=True)
        torch.save(data, f'tests/data/saved/{filename}')
    
    @staticmethod
    def load_test_data(filename: str) -> torch.Tensor:
        """加载测试数据"""
        return torch.load(f'tests/data/saved/{filename}')
```

## 10. 测试质量保证

### 10.1 测试覆盖率要求
- **单元测试覆盖率**: >= 90%
- **分支覆盖率**: >= 85%
- **函数覆盖率**: >= 95%

### 10.2 测试质量检查
```python
# scripts/check_test_quality.py
"""测试质量检查脚本"""

import ast
import os
from typing import List, Dict

class TestQualityChecker:
    """测试质量检查器"""
    
    def __init__(self, test_dir: str):
        self.test_dir = test_dir
        self.issues = []
    
    def check_test_naming(self) -> None:
        """检查测试命名规范"""
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith('.py') and file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    self._check_file_naming(file_path)
    
    def check_test_structure(self) -> None:
        """检查测试结构"""
        # 检查是否有setUp和tearDown方法
        # 检查断言的使用
        # 检查测试独立性
        pass
    
    def generate_report(self) -> Dict:
        """生成质量报告"""
        return {
            'total_issues': len(self.issues),
            'issues': self.issues
        }
```

这个测试策略文档为AIMET系统提供了全面的测试框架，确保代码质量和功能正确性，支持持续集成和持续部署。