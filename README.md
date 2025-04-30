该项目是构建一个用于 MNIST 手写数字识别的神经网络模型，采用从零实现的简单 CNN 架构，并通过训练优化模型性能。

## 项目内容

- 使用 `Conv2D`, `ReLU`, `MaxPool2D`, `Linear`, `Softmax + CrossEntropyLoss` 实现了一个简洁 CNN；
- 在 `test_train.py` 中构建训练流程，包含前向传播、反向传播、优化器更新；
- 在 `test_model.py` 中加载训练模型并评估测试集精度；
- 生成 `best_model.pkl` 模型文件，保存精度最优的模型；
- 提供模型可视化（卷积核可视化）功能。

---

## 已完成的问题要求（Project 1.2）

1. ✔️ 更改网络结构（隐藏单元个数）；
2. ✔️ 修改训练过程中的步长与动量（Momentum 优化）；
3. ✔️ 使用 L2 正则化与 Early Stopping；
4. ✔️ 实现 Cross Entropy Loss + Softmax 输出层；
5. ✔️ 自定义实现 Conv2D，构建 CNN 架构；
6. ✔️ 使用数据增强（翻转、旋转等）；
7. ✔️ 模型卷积核可视化（训练后绘制）。

---

## 🗂️ 目录结构说明

```
.
├── best_models/              # 保存的最佳模型（best_model.pkl）
├── dataset/MNIST/            # MNIST 数据集（压缩格式）
├── draw_tools/               # 可视化辅助工具
├── mynn/                     # 手动实现的神经网络组件
│   ├── op.py                 # Conv2D, ReLU, Pooling 等实现
│   └── optimizer.py          # SGD, Momentum 实现
├── test_train.py             # 主训练脚本
├── test_model.py             # 模型测试与评估
├── weight_visualization.py   # 卷积核可视化脚本
└── README.md                 # 本文件
