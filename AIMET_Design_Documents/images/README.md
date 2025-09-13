# AIMET 设计图表资源

本文件夹包含了AIMET设计文档中使用的所有架构图表和可视化资源。

## 📊 图表列表

### 🏗️ 架构设计图表

#### 1. 整体架构图
- **文件**: [overall_architecture.svg](overall_architecture.svg)
- **描述**: AIMET的五层架构设计图，展示了从应用接口层到基础设施层的完整架构
- **特点**: 
  - 清晰的层次结构
  - 渐变色彩设计
  - 组件分布展示
  - 数据流向标识

#### 2. 模块依赖关系图
- **文件**: [module_dependency.svg](module_dependency.svg)
- **描述**: 12个核心模块之间的依赖关系图
- **特点**:
  - 主控制器突出显示
  - 颜色编码区分模块类型
  - 依赖关系清晰标注
  - 组合关系可视化

### 🔧 类设计图表

#### 3. QuantizationSimModel类关系图
- **文件**: [quantsim_class_diagram.svg](quantsim_class_diagram.svg)
- **描述**: QuantizationSimModel及其相关类的UML类图
- **特点**:
  - 标准UML类图格式
  - 继承关系和组合关系
  - 主要方法列表
  - 类型分色标识

#### 4. TensorQuantizer类层次图
- **文件**: [tensor_quantizer_hierarchy.svg](tensor_quantizer_hierarchy.svg)
- **描述**: TensorQuantizer的完整类层次结构
- **特点**:
  - 抽象类特殊标识
  - 继承链清晰展示
  - 具体实现类分类
  - 组合关系标注

#### 5. EncodingAnalyzer类层次图
- **文件**: [encoding_analyzer_hierarchy.svg](encoding_analyzer_hierarchy.svg)
- **描述**: EncodingAnalyzer及其子类的层次结构和组件关系
- **特点**:
  - 策略模式体现
  - 算法分类说明
  - 工具组件关系
  - 详细功能注释

#### 6. ConnectedGraph组件图
- **文件**: [connected_graph_components.svg](connected_graph_components.svg)
- **描述**: ConnectedGraph模块的内部组件关系
- **特点**:
  - 组件职责分工
  - 协作关系展示
  - 不同类型组件标识
  - 数据流和控制流

### 🔄 流程图表

#### 7. 量化工作流程图
- **文件**: [quantization_workflow.svg](quantization_workflow.svg)
- **描述**: 完整的AIMET量化工作流程
- **特点**:
  - 端到端流程展示
  - 决策点标识
  - 分支流程处理
  - 阶段划分清晰
  - 详细步骤说明

#### 8. 数据流图
- **文件**: [data_flow_diagram.svg](data_flow_diagram.svg)
- **描述**: 量化过程中的数据流向和处理
- **特点**:
  - 数据流向可视化
  - 不同类型数据区分
  - 处理节点标识
  - 存储和缓存展示

#### 9. 模块交互时序图
- **文件**: [module_interaction_sequence.svg](module_interaction_sequence.svg)
- **描述**: AIMET模块间的详细交互时序
- **特点**:
  - 完整的交互流程
  - 阶段划分清晰
  - 消息传递顺序
  - 激活框标识
  - 详细的注释说明

## 🎨 设计规范

### 颜色方案
- **🔴 红色渐变**: 主要类、输入数据、开始/结束节点
- **🔵 蓝色渐变**: 核心组件、处理过程、框架适配层
- **🟢 绿色渐变**: 具体实现、输出结果、算法实现层
- **🟠 橙色渐变**: 工具类、存储节点、核心引擎层
- **🟣 紫色渐变**: 抽象类、控制流、基础设施层

### 图形元素
- **矩形**: 标准类和组件
- **圆形**: 输入/输出数据节点
- **菱形**: 决策判断节点
- **椭圆**: 开始/结束节点
- **圆柱**: 存储和缓存节点
- **六边形**: 控制组件

### 连接线规范
- **实线箭头**: 组合关系、主数据流
- **虚线箭头**: 继承关系、控制流
- **粗线**: 主要连接
- **细线**: 辅助连接
- **曲线**: 复杂路径连接

## 📱 使用说明

### 在Markdown中引用
```markdown
![图表标题](./images/图片文件名.svg)
```

### 在HTML中引用
```html
<img src="./images/图片文件名.svg" alt="图表标题" width="800">
```

### 图片特点
- **矢量格式**: SVG格式确保任意缩放不失真
- **高质量**: 专业的视觉设计和配色
- **信息丰富**: 包含详细的标注和说明
- **标准化**: 统一的设计规范和风格

## 🔧 技术实现

### SVG特性
- **可缩放**: 矢量图形支持任意缩放
- **小文件**: 相比位图文件更小
- **可编辑**: 可以直接编辑SVG代码
- **兼容性**: 所有现代浏览器支持

### 设计工具
- **手工编码**: 直接编写SVG代码
- **D3.js概念**: 借鉴D3.js的可视化设计理念
- **UML标准**: 遵循UML图表绘制标准
- **工程图规范**: 符合软件工程图表规范

这些图表资源为AIMET设计文档提供了专业、美观、信息丰富的可视化支持。