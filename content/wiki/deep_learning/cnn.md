+++
date = '2025-10-20T21:33:42+08:00'
draft = false
title = '卷积神经网络'
comments = true
weight = 2
+++

# 卷积神经网络

卷积神经网络（Convolutional Neural Networks, CNN）是深度学习中专门用于处理图像数据的神经网络架构，在计算机视觉领域取得了巨大成功。

## 🖼️ 图像处理基础

### 图像表示
图像可以表示为三维张量：
- **高度** (Height)
- **宽度** (Width)
- **通道数** (Channels): RGB图像为3，灰度图像为1

### 卷积操作
卷积是CNN的核心操作，通过在图像上滑动滤波器来提取特征：

**数学定义**：
$$(I * K)_{x,y} = \sum_{i}\sum_{j} I_{x+i, y+j} K_{i,j}$$

## 🔍 卷积层

### 卷积核
卷积核（Kernel/Filter）是用于特征提取的小矩阵：

**边缘检测**：
```
[-1, -1, -1]
[-1,  8, -1]
[-1, -1, -1]
```

**模糊滤波**：
```
[1, 1, 1]
[1, 1, 1]
[1, 1, 1]
```
（除以9进行归一化）

### 卷积参数

**步长 (Stride)**：卷积核滑动的步长
**填充 (Padding)**：图像边缘填充
**输出尺寸**：$\frac{W - F + 2P}{S} + 1$

## 🏊 池化层

### 最大池化 (Max Pooling)
保留区域内的最大值：
```
输入：
[[1, 3],
 [2, 4]]

最大池化：
[[4]]
```

### 平均池化 (Average Pooling)
计算区域内平均值：
```
输入：
[[1, 3],
 [2, 4]]

平均池化：
[[2.5]]
```

## 🏗️ 经典CNN架构

### LeNet-5 (1998)

** Yann LeCun的开创性工作 **

**架构**：
```
输入(32×32) → Conv(6@28×28) → Pool(6@14×14) → Conv(16@10×10) → Pool(16@5×5) → FC(120) → FC(84) → FC(10)
```

### AlexNet (2012)

**ImageNet 2012冠军**

**创新点**：
- 使用ReLU激活函数
- 引入Dropout防止过拟合
- 使用GPU加速训练
- 数据增强技术

**架构**：
```
输入(227×227×3) → Conv+ReLU+Pool → Conv+ReLU+Pool → Conv+ReLU → Conv+ReLU → Conv+ReLU+Pool → FC+Dropout → FC+Dropout → FC(1000)
```

### VGGNet (2014)

**简洁而有效的架构**

**特点**：
- 使用3×3小卷积核
- 堆叠多个卷积层
- 验证了网络深度的重要性

**VGG16架构**：
```
输入(224×224×3) → 2×Conv3×3 → Pool → 2×Conv3×3 → Pool → 3×Conv3×3 → Pool → 3×Conv3×3 → Pool → 3×Conv3×3 → Pool → FC(4096) → FC(4096) → FC(1000)
```

### ResNet (2015)

**残差网络，解决梯度消失问题**

**残差块**：
$$y = F(x) + x$$

**创新点**：
- 引入残差连接
- 允许训练更深的网络
- ResNet-152: 152层

## 🎯 目标检测

### R-CNN系列

**R-CNN (2014)**：
1. 使用选择性搜索生成候选框
2. 对每个候选框提取CNN特征
3. 使用SVM进行分类

**Fast R-CNN (2015)**：
- 共享卷积特征
- 引入RoI池化层
- 多任务损失函数

**Faster R-CNN (2015)**：
- 引入RPN（Region Proposal Network）
- 端到端训练
- 实时目标检测

### YOLO系列

**YOLO (You Only Look Once)**：
- 单阶段检测器
- 直接预测边界框和类别
- 速度快，适合实时应用

**YOLOv3**：
- 多尺度预测
- 更好的小目标检测
- 平衡速度和精度

## 🔄 反向传播

### 卷积层梯度

**权重梯度**：
$$\frac{\partial L}{\partial K} = \frac{\partial L}{\partial Y} * X^T$$

**输入梯度**：
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * K^T$$

### 池化层梯度

**最大池化**：
- 最大值位置的梯度为1
- 其他位置的梯度为0

**平均池化**：
- 所有位置的梯度相等

## 🚀 编程实现

### PyTorch实现

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # 展平
        x = x.view(-1, 128 * 4 * 4)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```

### 手动实现卷积

```python
def conv2d(input_tensor, kernel, stride=1, padding=0):
    """手动实现2D卷积"""
    # 输入尺寸
    batch_size, in_channels, in_height, in_width = input_tensor.shape

    # 卷积核尺寸
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape

    # 输出尺寸计算
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    # 输出张量
    output = np.zeros((batch_size, out_channels, out_height, out_width))

    # 填充输入
    if padding > 0:
        padded_input = np.pad(input_tensor,
                            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                            mode='constant')
    else:
        padded_input = input_tensor

    # 卷积操作
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    # 提取输入块
                    input_block = padded_input[
                        b,
                        :,
                        oh*stride : oh*stride + kernel_height,
                        ow*stride : ow*stride + kernel_width
                    ]

                    # 计算卷积
                    output[b, oc, oh, ow] = np.sum(input_block * kernel[oc])

    return output
```

## 📊 性能优化

### GPU加速
```python
# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型移动到GPU
model = CNN().to(device)
inputs = inputs.to(device)
```

### 批量归一化
```python
# 在卷积层后添加BN层
self.bn1 = nn.BatchNorm2d(32)
self.bn2 = nn.BatchNorm2d(64)
self.bn3 = nn.BatchNorm2d(128)
```

### 残差连接
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        return x
```

## 🎨 计算机视觉应用

### 图像分类
- **ImageNet**：1000类图像分类
- **CIFAR-10/100**：10/100类小图像分类
- **MNIST**：手写数字识别

### 目标检测
- **边界框回归**：预测目标位置
- **分类**：判断目标类别
- **置信度**：预测检测的准确性

### 语义分割
- **FCN**：全卷积网络
- **U-Net**：医学图像分割
- **DeepLab**：多尺度特征融合

### 图像生成
- **GAN**：生成对抗网络
- **VAE**：变分自编码器
- **Style Transfer**：风格迁移

## 📈 评估指标

### 分类任务
- **Top-1准确率**：预测类别与真实类别完全匹配
- **Top-5准确率**：预测的5个类别中包含真实类别

### 目标检测
- **mAP (mean Average Precision)**：平均精度均值
- **IoU (Intersection over Union)**：预测框与真实框的重叠度

### 语义分割
- **Pixel Accuracy**：像素级准确率
- **IoU (Intersection over Union)**：类别级IoU
- **mIoU**：平均IoU

## 🔧 实用技巧

### 数据增强
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 迁移学习
```python
# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)

# 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 替换分类器
model.fc = nn.Linear(2048, num_classes)
```

### 模型可视化
```python
# 可视化卷积核
def visualize_kernels(model):
    kernels = model.conv1.weight.data.cpu().numpy()
    # 绘制卷积核图像
    plt.figure(figsize=(10, 10))
    for i in range(32):
        plt.subplot(6, 6, i+1)
        plt.imshow(kernels[i, 0], cmap='gray')
        plt.axis('off')
    plt.show()
```

## 📚 学习资源

### 吴恩达课程
- [第四周：卷积神经网络](https://www.coursera.org/learn/convolutional-neural-networks)

### 经典论文
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) - AlexNet
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - VGG
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet

### 在线资源
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [PyTorch Vision Tutorials](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

---
*最近更新: {{ .Lastmod.Format "2006-01-02" }}*