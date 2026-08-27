+++
date = '2025-10-20T21:38:39+08:00'
draft = false
title = '循环神经网络'
comments = true
weight = 3
+++

# 循环神经网络

循环神经网络（Recurrent Neural Networks, RNN）是专门用于处理序列数据的神经网络架构，能够记忆历史信息并用于当前预测。

## ⏰ 序列数据处理

### 序列数据特点
- **时间依赖**：当前状态依赖于之前的状态
- **变长输入**：序列长度可能不同
- **上下文信息**：需要理解前后文关系

### 常见序列数据
- **文本序列**：句子、文章、代码
- **时间序列**：股票价格、天气数据
- **音频序列**：语音信号、音乐
- **视频序列**：动作序列、行为识别

## 🔄 RNN基本结构

### 简单RNN单元

**隐藏状态更新**：
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

**输出计算**：
$$y_t = W_{hy} h_t + c$$

### 循环机制
- **状态传递**：$h_t$ 依赖于 $h_{t-1}$
- **参数共享**：所有时间步使用相同的权重
- **记忆能力**：通过状态传递保留历史信息

## 🚨 梯度消失与爆炸

### 梯度消失问题
- **原因**：链式求导导致梯度指数级衰减
- **影响**：长期依赖无法学习
- **解决方案**：LSTM、GRU等改进架构

### 梯度爆炸问题
- **原因**：梯度指数级增长
- **影响**：训练不稳定，参数更新过大
- **解决方案**：梯度裁剪、权重正则化

## 🧠 长短期记忆网络 (LSTM)

### LSTM单元结构

**遗忘门**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**细胞状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### LSTM优势
- **长期记忆**：细胞状态传递历史信息
- **选择性遗忘**：遗忘门控制信息保留
- **梯度稳定**：避免梯度消失/爆炸

## 🔄 门控循环单元 (GRU)

### GRU简化结构

**更新门**：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**重置门**：
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**候选隐藏状态**：
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$

**隐藏状态更新**：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### GRU vs LSTM
- **参数更少**：GRU比LSTM参数量少25%
- **训练速度**：GRU通常训练更快
- **性能相当**：在许多任务上性能相似

## 📝 自然语言处理应用

### 文本生成
```python
def generate_text(model, start_text, length=100):
    generated = start_text

    for _ in range(length):
        # 编码输入
        x = encode_text(generated[-seq_length:])

        # 预测下一个字符
        pred = model.predict(x)
        next_char = decode_prediction(pred)

        # 添加到生成文本
        generated += next_char

    return generated
```

### 机器翻译
- **编码器**：将源语言编码为向量
- **解码器**：生成目标语言序列
- **注意力机制**：关注源语言的相关部分

### 情感分析
- **输入**：文本序列
- **输出**：情感极性（正面/负面）
- **应用**：产品评论、社交媒体分析

## 📈 时间序列预测

### 股票价格预测
```python
# 数据预处理
def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])

    return np.array(sequences), np.array(targets)

# 模型构建
model = Sequential([
    LSTM(50, input_shape=(seq_length, 1)),
    Dense(1)
])
```

### 天气预测
- **输入特征**：温度、湿度、气压等历史数据
- **预测目标**：未来几天的天气状况
- **挑战**：处理多变量时间序列

## 🎵 音频处理应用

### 语音识别
- **声谱图**：将音频转换为图像
- **CTC损失**：处理序列长度不匹配
- **语言模型**：提高识别准确率

### 音乐生成
- **MIDI序列**：音符序列生成
- **风格迁移**：学习音乐风格
- **和声生成**：自动作曲系统

## 🏗️ 高级RNN架构

### 双向RNN (BiRNN)
- **前向RNN**：正向处理序列
- **后向RNN**：反向处理序列
- **拼接输出**：结合前后文信息

### 多层RNN
- **堆叠结构**：多个RNN层堆叠
- **特征层次**：学习不同抽象层次的特征
- **梯度问题**：更严重的梯度消失

### 注意力增强RNN
- **注意力机制**：关注序列中的重要部分
- **长距离依赖**：直接连接远距离信息
- **可解释性**：可视化注意力权重

## 🚀 编程实现

### PyTorch LSTM实现

```python
import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM输出
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])

        return output
```

### Keras GRU实现

```python
from keras.models import Sequential
from keras.layers import GRU, Dense

model = Sequential([
    GRU(64, input_shape=(seq_length, num_features)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
```

## 📊 评估指标

### 序列生成任务
- **困惑度 (Perplexity)**：衡量语言模型的预测能力
- **BLEU分数**：机器翻译质量评估
- **ROUGE分数**：文本摘要质量评估

### 时间序列预测
- **均方误差 (MSE)**：预测值与真实值的平均误差
- **平均绝对误差 (MAE)**：绝对误差的平均值
- **平均绝对百分比误差 (MAPE)**：相对误差的平均值

## 🔧 实用技巧

### 序列填充
```python
from keras.preprocessing.sequence import pad_sequences

# 填充序列到相同长度
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
```

### 掩码机制
```python
# 处理变长序列
mask = tf.sequence_mask(sequence_lengths, maxlen=max_length)
```

### 注意力可视化
```python
def plot_attention(attention_weights, input_tokens, output_tokens):
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights, cmap='viridis')
    plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
    plt.yticks(range(len(output_tokens)), output_tokens)
    plt.colorbar()
    plt.show()
```

## 🎯 应用实例

### 文本生成
```python
# 训练文本生成模型
model = Sequential([
    LSTM(128, input_shape=(seq_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

# 生成文本
def generate_text(seed_text, num_words=50):
    for _ in range(num_words):
        # 编码种子文本
        encoded = encode_sequence(seed_text[-seq_length:])

        # 预测下一个词
        prediction = model.predict(encoded)
        next_word = decode_prediction(prediction)

        # 添加到文本
        seed_text += ' ' + next_word

    return seed_text
```

### 情感分析
```python
# 加载预训练词向量
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=False
)

# 构建模型
model = Sequential([
    embedding_layer,
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## 📚 学习资源

### 吴恩达课程
- [第五周：序列模型](https://www.coursera.org/learn/sequence-models)

### 经典论文
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber (1997)
- [Learning to forget: Continual prediction with LSTM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/hochreiter97_lstm.pdf) - Gers et al. (2000)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) - Cho et al. (2014)

### 在线资源
- [Colah's Blog: Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Illustrated Guide to RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb803935)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers)

---
*最近更新: {{ .Lastmod.Format "2006-01-02" }}*