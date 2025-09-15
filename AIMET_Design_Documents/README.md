# AIMET 设计文档集合

本文件夹包含了AIMET（AI Model Efficiency Toolkit）的完整设计文档，每个模块都有独立的设计文档。

## 🌟 快速开始

**推荐**: 首先访问 **[📱 设计文档门户 (index.html)](index.html)** 获得最佳的浏览体验！

门户页面提供：
- 🎨 精美的可视化界面
- 🧭 个性化的学习路径指导
- 📊 文档统计信息
- 🔗 便捷的导航链接

## 文档结构

### 🎨 精美可视化图表集合

#### 📊 静态图表 (推荐 - 兼容性好)
- **[images/](images/)** - 📁 **高质量SVG图表集合**
  - 🏗️ [五层架构图](images/overall_architecture.svg)
  - 🔗 [模块依赖关系图](images/module_dependency.svg)
  - 🎯 [QuantizationSimModel类图](images/quantsim_class_diagram.svg)
  - 🔧 [TensorQuantizer类层次图](images/tensor_quantizer_hierarchy.svg)
  - 📈 [EncodingAnalyzer类层次图](images/encoding_analyzer_hierarchy.svg)
  - 🌐 [ConnectedGraph组件图](images/connected_graph_components.svg)
  - 🔄 [量化工作流程图](images/quantization_workflow.svg)
  - 🌊 [数据流图](images/data_flow_diagram.svg)
  - ⏰ [模块交互时序图](images/module_interaction_sequence.svg)

#### 🚀 交互式图表 (需要浏览器支持)
- **[architecture_diagrams.html](architecture_diagrams.html)** - 🌟 **交互式架构图表集合**
  - 💡 悬停提示和交互功能
  - 🎨 D3.js制作的动态图表
  - 📱 响应式设计
  
- **[3d_architecture.html](3d_architecture.html)** - 🚀 **3D立体架构可视化**
  - 🎮 Three.js制作的3D立体架构
  - 🔄 可旋转、缩放、分解视图
  - ✨ 实时动画和光影效果
  
- **[workflow_diagrams.html](workflow_diagrams.html)** - 🔄 **工作流程可视化**
  - 📋 完整量化工作流程图
  - ⚙️ 编码计算详细流程
  - ⏰ 模块交互时序图

- **[data_flow_visualization.html](data_flow_visualization.html)** - 🌊 **数据流动态可视化**
  - ▶️ 动态流动动画效果
  - 🎯 关键路径高亮显示
  - 📈 实时统计信息展示

### 核心架构文档
- [00_Overall_Architecture.md](00_Overall_Architecture.md) - 系统整体架构设计
- [01_Module_Interaction.md](01_Module_Interaction.md) - 模块间交互机制

### 核心模块设计文档
- [02_QuantizationSimModel.md](02_QuantizationSimModel.md) - 量化仿真模型设计
- [03_TensorQuantizer.md](03_TensorQuantizer.md) - 张量量化器设计
- [04_EncodingAnalyzer.md](04_EncodingAnalyzer.md) - 编码分析器设计
- [05_ConnectedGraph.md](05_ConnectedGraph.md) - 连接图模块设计
- [06_QuantSimConfig.md](06_QuantSimConfig.md) - 量化配置模块设计
- [07_AutoQuant.md](07_AutoQuant.md) - 自动量化模块设计
- [08_ModelCompression.md](08_ModelCompression.md) - 模型压缩模块设计
- [09_MixedPrecision.md](09_MixedPrecision.md) - 混合精度模块设计
- [10_Visualization.md](10_Visualization.md) - 可视化模块设计
- [11_ModelExport.md](11_ModelExport.md) - 模型导出模块设计
- [12_CacheManager.md](12_CacheManager.md) - 缓存管理模块设计
- [13_Utilities.md](13_Utilities.md) - 工具集模块设计

### 实现指南
- [14_Implementation_Guide.md](14_Implementation_Guide.md) - 详细实现步骤指南
- [15_API_Reference.md](15_API_Reference.md) - API接口参考文档
- [16_Testing_Strategy.md](16_Testing_Strategy.md) - 测试策略文档

## 阅读顺序建议

### 对于架构师和技术负责人
1. 先阅读整体架构设计（00_Overall_Architecture.md）
2. 了解模块间交互机制（01_Module_Interaction.md）
3. 重点关注核心模块设计（02-13）

### 对于开发工程师
1. 从实现指南开始（14_Implementation_Guide.md）
2. 根据负责的模块阅读对应的设计文档
3. 参考API接口文档进行开发（15_API_Reference.md）

### 对于测试工程师
1. 阅读整体架构了解系统结构
2. 重点关注测试策略文档（16_Testing_Strategy.md）
3. 了解各模块的接口设计

## 文档维护

- 每个模块文档包含：设计目标、架构设计、接口定义、实现细节、测试要求
- 文档采用Markdown格式，便于版本控制和协作
- 定期更新文档以保持与代码实现的同步

## 版本信息

- 文档版本：v1.0
- 基于AIMET开源版本分析创建
- 最后更新：2024年