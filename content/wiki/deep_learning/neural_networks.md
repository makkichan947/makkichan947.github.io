+++
date = '2025-10-19T20:10:09+08:00'
draft = false
title = '神经网络基础'
comments = true
weight = 1
+++

# 神经网络基础

神经网络（Neural Networks）是深度学习的基础，模拟了人脑神经元的工作方式。本章详细介绍神经网络的基本概念、结构和训练方法。

## 🧠 神经元模型

### 生物神经元
人脑中的神经元通过突触接收和处理信号，神经网络中的人工神经元模仿了这一过程。

### 人工神经元
人工神经元（Artificial Neuron）是神经网络的基本单元：

**数学模型**：
$$z = \sum_{i=1}^{n} w_i x_i + b$$
$$a = g(z)$$

其中：
- $x_i$：输入特征
- $w_i$：权重参数
- $b$：偏置项
- $g(\cdot)$：激活函数
- $a$：神经元输出

### 神经网络结构

**单层神经网络**：
```
输入层 → 输出层
```

**多层神经网络**：
```
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
```

## 🔄 前向传播

### 计算过程

**第一层**：
$$z^{(1)} = W^{(1)} x + b^{(1)}$$
$$a^{(1)} = g^{(1)}(z^{(1)})$$

**第l层**：
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = g^{(l)}(z^{(l)})$$

### 向量表示

使用矩阵运算提高效率：
$$Z = W \cdot A + b$$
$$A = g(Z)$$

## 📉 反向传播

### 损失函数

**均方误差 (MSE)**：
$$J = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**交叉熵损失**：
$$J = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)]$$

### 梯度计算

**输出层梯度**：
$$\frac{\partial J}{\partial z^{(L)}} = a^{(L)} - y$$

**隐藏层梯度**：
$$\frac{\partial J}{\partial z^{(l)}} = \frac{\partial J}{\partial z^{(l+1)}} \cdot (W^{(l+1)})^T \odot g'(z^{(l)})$$

## ⚡ 激活函数

### Sigmoid函数
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**导数**：
$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

### ReLU函数
$$ReLU(z) = \max(0, z)$$

**导数**：
$$ReLU'(z) = \begin{cases}
1 & z > 0 \\
0 & z \leq 0
\end{cases}$$

### Tanh函数
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**导数**：
$$\tanh'(z) = 1 - \tanh^2(z)$$

## 🎯 梯度下降

### 批量梯度下降 (BGD)

**参数更新**：
$$W = W - \alpha \frac{\partial J}{\partial W}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

### 随机梯度下降 (SGD)

**参数更新**：
$$W = W - \alpha \frac{\partial J^{(i)}}{\partial W}$$
$$b = b - \alpha \frac{\partial J^{(i)}}{\partial b}$$

### 小批量梯度下降 (Mini-batch SGD)

结合BGD和SGD的优点：
- 每次使用一小批样本计算梯度
- 收敛速度快且更稳定

## 🏗️ 网络架构

### 前馈神经网络 (FNN)

**特点**：
- 信息单向流动
- 没有循环连接
- 适合分类和回归任务

### 深度神经网络 (DNN)

**深度**：通常指具有多个隐藏层的网络
**优势**：
- 学习更复杂的特征表示
- 解决更复杂的实际问题

## 📊 过拟合与正则化

### 过拟合问题

**表现**：
- 训练误差低，测试误差高
- 模型在训练数据上表现很好
- 在新数据上表现很差

### L2正则化

**损失函数**：
$$J_{regularized} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{(l)}\|_F^2$$

**梯度**：
$$\frac{\partial J_{regularized}}{\partial W} = \frac{\partial J}{\partial W} + \frac{\lambda}{m} W$$

### Dropout

**训练时**：随机丢弃部分神经元
**测试时**：使用所有神经元但权重缩放

## 🚀 编程实现

### PyTorch实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 模型初始化
model = NeuralNetwork(784, 128, 10)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(num_epochs):
    for data, target in dataloader:
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 手动实现反向传播

```python
def forward_propagation(X, parameters):
    """前向传播"""
    W1, b1, W2, b2 = parameters

    # 第一层
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # 第二层
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def backward_propagation(parameters, cache, X, Y):
    """反向传播"""
    W1, b1, W2, b2 = parameters
    Z1, A1, Z2, A2 = cache.values()

    m = X.shape[1]

    # 输出层梯度
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # 隐藏层梯度
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients
```

## 🎨 激活函数可视化

### Sigmoid函数
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
```

### ReLU函数
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
```

## 📈 训练技巧

### 权重初始化

**Xavier初始化**：
```python
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))
```

### 学习率衰减

**指数衰减**：
```python
learning_rate = initial_lr * decay_rate ** epoch
```

### 梯度检查

验证反向传播的正确性：
```python
def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    for param_name in parameters:
        param = parameters[param_name]
        grad = gradients["d" + param_name]

        # 计算数值梯度
        numerical_grad = np.zeros_like(param)
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                param_plus = param.copy()
                param_minus = param.copy()
                param_plus[i,j] += epsilon
                param_minus[i,j] -= epsilon

                loss_plus = compute_loss(param_plus)
                loss_minus = compute_loss(param_minus)
                numerical_grad[i,j] = (loss_plus - loss_minus) / (2 * epsilon)

        # 比较梯度
        diff = np.linalg.norm(grad - numerical_grad) / np.linalg.norm(grad + numerical_grad)
        print(f"{param_name} gradient check: {diff}")
```

## 🔧 超参数调优

### 网格搜索

```python
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [64, 128, 256]
batch_sizes = [32, 64, 128]

best_accuracy = 0
best_params = {}

for lr in learning_rates:
    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            # 训练模型
            accuracy = train_model(lr, hidden_size, batch_size)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size}
```

### 随机搜索

```python
def random_search(num_trials=100):
    best_accuracy = 0
    best_params = {}

    for _ in range(num_trials):
        # 随机采样超参数
        lr = 10 ** np.random.uniform(-5, -1)
        hidden_size = 2 ** np.random.randint(5, 9)
        batch_size = 2 ** np.random.randint(4, 8)

        # 训练并评估
        accuracy = train_model(lr, hidden_size, batch_size)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size}

    return best_params
```

## 📊 评估指标

### 分类任务

**准确率 (Accuracy)**：
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**精确率 (Precision)**：
$$Precision = \frac{TP}{TP + FP}$$

**召回率 (Recall)**：
$$Recall = \frac{TP}{TP + FN}$$

**F1分数**：
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 回归任务

**均方误差 (MSE)**：
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**平均绝对误差 (MAE)**：
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

## 🎯 应用实例

### 手写数字识别

```python
# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 模型定义
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练模型
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 📚 学习资源

### 吴恩达课程
- [第一周：神经网络和深度学习基础](https://www.coursera.org/learn/neural-networks-deep-learning)

### 经典论文
- [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://www.cs.cmu.edu/~./epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) - McCulloch & Pitts (1943)
- [Learning Internal Representations by Error Propagation](https://www.iro.umontreal.ca/~pift6266/A06/refs/backprop.pdf) - Rumelhart et al. (1986)

### 在线资源
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/)