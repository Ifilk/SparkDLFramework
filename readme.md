# Spark分布式深度学习框架开发文档

## 1. 项目概述

本项目旨在使用Kotlin和Spark实现一个简易的分布式深度学习框架，包含自动微分、分布式训练和模型并行等核心功能，最终在MNIST数据集上进行验证。

### 1.1 功能特性

- [x] 支持基础Tensor操作和自动微分
- [ ] 实现Dense神经网络层
- [ ] 模拟NCCL集合通信接口
- [ ] 提供类似torch.distributed的分布式训练能力
- [ ] 支持数据并行和模型并行策略
- [ ] MNIST手写数字识别完整实现
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

- Tensor: 封装张量数据结构，提供常用操作（加减乘除、广播、reshape 等），支持前向和反向传播。
- Autograd: 实现自动微分计算图机制，支持链式求导和动态图结构。
- NN: 提供基础神经网络组件（如 Dense 层、激活函数、损失函数等）与前向传播逻辑。
- Optim: 提供优化器（如 SGD、Adam），支持梯度更新、学习率衰减等。

分布式模块

- Distributed.nccl: 模拟 NCCL 接口，包括 all-reduce、broadcast、gather 等基本集合通信操作。
- Distributed.torch: 模拟 PyTorch 分布式接口（如 `init_process_group`、`DistributedDataParallel` 等）。
- Parallel: 实现数据并行（Data Parallelism）和模型并行（Model Parallelism）策略调度。

应用模块

- data: 加载与预处理 MNIST 数据集，支持分布式数据加载与划分。
- models: 定义网络结构（如两层 MLP），支持模型保存与加载。
- utils: 通用工具类，包括日志记录、参数配置、计时器等。

## 3. 模块实现细节

### 3.1 Tensor 模块

**功能说明**

实现张量数据结构（多维数组）以及基础操作（如加减乘除、矩阵乘法、reshape 等），并集成自动微分所需的梯度存储与操作记录。

**核心类**

- Tensor
    - 属性：
        - double[] data：张量存储的一维数据
        - int[] shape：张量的维度信息
        - Tensor grad：梯度信息
        - Function backwardFn：反向传播函数
    - 方法：
        - add(Tensor other)、mul(Tensor other) 等基础操作
        - reshape(int...)
        - backward()：启动反向传播
- TensorOp
  定义操作接口及其对应的前向与反向传播逻辑。

### 3.2 Autograd 模块

**功能说明**

构建自动求导机制，记录计算图并支持链式反向传播。

**核心类**

- AutogradEngine
    - 功能：
        - 构建动态计算图
        - 管理梯度传播
        - 执行拓扑排序以确保反向传播顺序
- Function（抽象类）
    - 表示一个可微分的操作，子类如 AddFunction, MatMulFunction
    - 提供：
        - forward() 前向传播
        - backward(gradOutput) 反向传播

## 4. 自动微分设计

### 4.1 自动微分原理

本框架采用动态图(Dynamic Computational Graph)构建策略，与 PyTorch 类似。每当执行一个 Tensor
运算，框架会在后台动态记录一个反向传播所需的计算图，直到用户调用 backward() 方法时，按计算图拓扑顺序回溯并执行梯度传播。

### 4.2 计算图节点结构

每个参与计算的 Tensor 包含如下附加属性：

| 属性名            | 类型         | 说明               |
|----------------|------------|------------------|
| `grad`         | `Tensor`   | 当前节点的梯度          |
| `creator`      | `Function` | 该 Tensor 的创建函数   |
| `requiresGrad` | `boolean`  | 是否需要自动求导         |
| `isLeaf`       | `boolean`  | 是否是叶子节点（通常是模型参数） |

### 4.3  计算图构建

当执行一个 Tensor 运算时，框架会记录当前 Tensor 的创建函数，并设置 `requiresGrad` 属性为 true。
当调用 backward() 方法时，框架会从输出节点开始，沿着计算图进行反向传播，并计算每个节点的梯度。

### 4.4  梯度传播

梯度传播算法如下：

1. 创建一个空的梯度缓存列表 `gradBuffer`。
2. 遍历计算图，从输出节点开始，沿着计算图进行反向传播。
3. 对于每个节点，如果该节点需要自动求导（即 `requiresGrad` 属性为 true），则计算该节点的梯度，并添加到 `gradBuffer` 中。
4. 遍历 `gradBuffer`，将每个节点的梯度累加到该节点的 `grad` 属性中。
5. 重置 `gradBuffer`，将所有节点的 `grad` 属性设置为 null。
6. 返回根节点的梯度。

## 5. 通信协议模拟细节（Distributed Communication Simulation）

为实现分布式深度学习，框架提供模拟版 NCCL 通信协议，支持在 Spark 的多 Executor 之间进行张量同步、广播等操作。

### 5.1 Spark 通信基础

Spark 的通信协议是 RPC（Remote Procedure Call）协议，即在分布式计算中，每个节点都拥有一个 RPC 服务，用于接收其他节点的请求，并返回结果。
由于 Spark Executor 无法直接通信（不像原生分布式系统），我们采用以下替代方式：

- 所有通信通过 Spark 的 Driver 协调。
- 每个 Executor 通过 RPC + Spark广播变量 模拟数据同步。

### 5.2 通信原语接口

```java
interface Comm {
    Tensor allReduce(Tensor input, String op);  // 如 sum, mean

    void broadcast(Tensor tensor, int rootRank);

    Tensor gather(Tensor input, int rootRank);

    Tensor reduceScatter(List<Tensor> inputs, String op);
}
```

框架提供模拟版 NCCL 模块，实现上述通信原语接口。

### 5.3 模拟 AllReduce 实现原理

AllReduce 是一种分布式通信协议，用于将多个节点上的张量进行聚合，并返回聚合后的结果。在分布式深度学习中，AllReduce
通常用于聚合多个节点上的梯度，并更新模型参数。
模拟 AllReduce(sum) 步骤：

1. 所有 Executor 将自己的梯度通过 Spark 的 `mapPartitions` 上报给 Driver。
2. Driver 收集所有梯度，并执行 `sum` 聚合。
3. Driver 将聚合结果广播回所有 Executor。
4. 各 Executor 将对应的参数更新。

```java
// pseudo code
List<Tensor> grads = rdd.mapPartitions(localGrad -> sendToDriver(localGrad)).collect();
Tensor reduced = sum(grads);
Broadcast<Tensor> bReduced = sparkContext.broadcast(reduced);
```

### 5.4 兼容 torch.distributed 接口模拟

框架提供兼容 torch.distributed 接口的模拟版 NCCL 模块，实现上述通信原语接口。

- `initProcessGroup(String backend, int worldSize)`：初始化通信环境
- `getRank()`：当前进程 rank
- `getWorldSize()`：进程总数
- `barrier()`：同步阻塞
  这些接口由 `DistributedContext` 类封装，内部通过 Spark TaskContext 或 executor id 进行模拟。

## 6. 测试、性能评估与 PyTorch 对比实验设计

测试框架主要通过单元测试和集成测试完成，单元测试主要测试 Tensor 模块、Autograd 模块和 Distributed Communication Simulation
模块，集成测试主要测试整个框架。

### 6.1 测试方案设计

#### 6.1.1 单元测试（Unit Test）

| 模块         | 测试重点                  |
|------------|-----------------------|
| `Tensor`   | 张量构造、运算、shape 检查、广播机制 |
| `Autograd` | 自动微分链条正确性，梯度值验证       |
| `NN`       | 层结构前向输出正确性，参数更新验证     |
| `Optim`    | 参数步进逻辑正确性，特殊超参数测试     |
| `Comm`     | 多进程模拟通信结果一致性          |

#### 6.1.2 集成测试（Integration Test）

构建小规模模型（如 2 层 MLP），在 MNIST 上跑通完整训练流程，验证：

- [ ] 梯度同步正确性（通信模拟）
- [ ] 并行策略下各 executor 的数据是否正确分配
- [ ] 模型是否成功收敛

#### 6.1.3 回归测试（Regression Test）

- 确保对已有功能更改不会破坏已有模型训练流程。
- 自动比对 loss 值是否呈下降趋势。

### 6.2 性能评估指标

#### 6.2.1 模型训练指标

| 指标         | 说明                 |
|------------|--------------------|
| Accuracy   | 在 MNIST 测试集上的分类准确率 |
| Loss       | 训练损失下降曲线           |
| Epoch Time | 单轮训练耗时（秒）          |
| Total Time | 总训练时长              |

#### 6.2.2 系统资源指标

| 指标      | 工具建议                  | 描述                      |
|---------|-----------------------|-------------------------|
| CPU 使用率 | `Spark UI` 或 `jvmtop` | 单 Executor 执行线程开销       |
| 内存占用    | Spark metrics         | Tensor 缓存与中间变量占用        |
| 网络 IO   | 自定义统计器                | 模拟通信传输量（如 allReduce 大小） |

### 6.3 PyTorch 对比实验设计

#### 6.3.1 对比目的

- 验证本框架自动微分与优化器正确性
- 分析本框架在 Spark 上运行的性能瓶颈
- 比较 Java 实现与 PyTorch（C++/Python）在小模型上的执行效率

#### 6.3.2 实验设置

| 项目    | 本框架                         | PyTorch                       |
|-------|-----------------------------|-------------------------------|
| 语言    | Java 21                     | Python 3.10                   |
| 分布式机制 | Spark Executor + NCCL 模拟    | torch.distributed + NCCL/Gloo |
| 模型结构  | 784 → 128 → 64 → 10 (MLP)   | 相同结构                          |
| 数据集   | MNIST                       | MNIST                         |
| 优化器   | SGD (lr=0.01, momentum=0.9) | 同上                            |
| 并行策略  | 数据并行（4 executor）            | DDP + 4 GPU                   |

#### 6.3.3 对比指标

| 指标             | 单位       | 说明                       |
|----------------|----------|--------------------------|
| 收敛速度（accuracy） | epoch    | 达到 95% 准确率所需 epoch 数     |
| 单轮耗时           | 秒/epoch  | 单轮训练时间                   |
| 总耗时            | 秒        | 到收敛的总训练时间                |
| 通信开销           | MB/epoch | 每轮平均 allReduce 数据量（模拟测算） |
| 实现复杂度          | LOC      | 关键模块实现代码量（估计）            |
