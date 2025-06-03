# Spark分布式深度学习框架开发文档

## 1. 项目概述

本项目旨在使用Kotlin和Spark实现一个简易的分布式深度学习框架，包含自动微分、分布式训练等核心功能，最终在MNIST数据集上进行验证

### 1.1 功能特性

- [x] 支持基础Tensor操作和自动微分
- [x] 实现Linear神经网络层
- [x] 实现SGD和Adam优化器
- [x] 支持Batch
- [x] 支持Spark分布式数据并行训练
- [ ] 支持模型并行
- [x] MNIST手写数字识别完整实现
- [ ] (可选)SafeTensor模型导出格式
- [ ] (可选)与PyTorch性能对比

### 1.2 技术栈

语言: Kotlin

计算引擎: Apache Spark 3.x

构建工具: Maven

测试框架: JUnit 5

## 2. 架构设计

### 2.1 整体架构

```
[Spark Cluster]
├── [Driver] - 协调节点
└── [Executor] - 工作节点
    ├── Tensor Core
    ├── Autograd Engine
    ├── Distributed Communication
    └── Parallel Training
```

### 2.2 模块设计

核心模块

- Tensor: 封装张量数据结构，提供常用操作（加减乘除、广播、reshape 等），支持前向和反向传播
- Autograd: 实现自动微分计算图机制，支持链式求导
- NN: 提供基础神经网络组件（如 Linear 层）与前向传播逻辑
- Optim: 提供优化器（如 SGD、Adam）
- Functions:  提供常用函数，如 ReLU、Softmax 等

分布式模块

- distributed.spark 模块：实现数据并行

应用模块

- data: 加载数据集
- models: 定义网络结构（如两层 MLP），支持模型保存与加载。
- utils: 通用工具类

## 快速开始
1. 编译代码
```bash
./gradlew shadowJar
```

2. 使用docker-compose快捷部署，MNIST数据集模型训练
```bash
docekr-compose -f docker-compose.yml -p sparkdlframework up -d --build
```
模型结构为784\*128\*10
具体训练代码见`src/main/scala/xyz/ifilk/models/mnist`

**注意**：submit容器需要等待init容器提交完成后才可以正常运行