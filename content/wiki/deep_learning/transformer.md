+++
date = '2025-10-25T00:10:19+08:00'
draft = false
title = 'Transformer架构'
comments = true
weight = 4
+++

# Transformer架构

Transformer是2017年由Google提出的革命性架构，完全基于注意力机制，彻底改变了深度学习领域。本章详细介绍Transformer的核心概念和实现细节。

## 🎯 注意力机制

### 基本注意力
注意力机制允许模型在处理序列时动态地聚焦于相关部分：

**Scaled Dot-Product Attention**：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

其中：
- $Q$：查询矩阵
- $K$：键矩阵
- $V$：值矩阵
- $d_k$：键向量的维度

### 多头注意力
多头注意力机制并行计算多个注意力：

**多头计算**：
$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$
$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

## 🏗️ Transformer架构

### 编码器 (Encoder)

**编码器层结构**：
1. **多头自注意力**：$MultiHeadAttention$
2. **前馈网络**：$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$
3. **残差连接和层归一化**

### 解码器 (Decoder)

**解码器层结构**：
1. **掩码多头自注意力**：防止看到未来信息
2. **多头注意力**：关注编码器输出
3. **前馈网络**：与编码器相同
4. **残差连接和层归一化**

## 📝 位置编码

### 问题
Transformer没有循环或卷积结构，无法感知序列位置。

### 解决方案
**正弦位置编码**：
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**相对位置编码**：考虑相对位置关系

## 🎭 自注意力机制

### 自注意力计算

**查询、键、值**：
- $Q = XW^Q$
- $K = XW^K$
- $V = XW^V$

**注意力分数**：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

### 掩码机制

**填充掩码**：忽略填充位置
**序列掩码**：防止解码器看到未来信息

## 🚀 编程实现

### PyTorch实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

### 完整Transformer编码器

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        # 编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
```

## 🎨 注意力可视化

### 注意力权重
```python
def plot_attention(attention_weights, tokens):
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights, cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.show()
```

### 多头注意力
```python
def plot_multihead_attention(attention_weights, tokens, num_heads):
    fig, axes = plt.subplots(1, num_heads, figsize=(15, 5))

    for i in range(num_heads):
        axes[i].imshow(attention_weights[i], cmap='viridis')
        axes[i].set_xticks(range(len(tokens)))
        axes[i].set_xticklabels(tokens, rotation=45)
        axes[i].set_title(f'Head {i+1}')

    plt.tight_layout()
    plt.show()
```

## 📊 性能优化

### 模型并行
- **张量并行**：在多个GPU上分割模型参数
- **流水线并行**：不同GPU处理不同层
- **数据并行**：每个GPU处理不同批次

### 内存优化
- **梯度检查点**：减少激活值存储
- **混合精度训练**：使用float16减少内存
- **模型分片**：按需加载模型参数

## 🎯 应用领域

### 自然语言处理
- **机器翻译**：Google Translate, DeepL
- **文本生成**：GPT系列模型
- **文本摘要**：自动摘要生成
- **问答系统**：智能问答机器人

### 计算机视觉
- **图像描述**：为图像生成文字描述
- **视觉问答**：基于图像的问答
- **图像生成**：DALL-E, Stable Diffusion

### 语音处理
- **语音识别**：端到端语音识别
- **语音合成**：TTS系统
- **语音翻译**：实时语音翻译

### 多模态学习
- **图像-文本**：CLIP模型
- **视频理解**：视频问答系统
- **跨模态生成**：文本生成图像

## 🔧 实用技巧

### 学习率调度
```python
# Warmup + 余弦退火
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 标签平滑
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx=0, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
```

### Beam Search
```python
def beam_search(model, src, beam_size=5, max_len=50):
    model.eval()

    # 编码源序列
    with torch.no_grad():
        encoder_output = model.encode(src)

    # 初始化
    candidates = [(0, [model.bos_idx])]
    finished = []

    for step in range(max_len):
        new_candidates = []

        for score, sequence in candidates:
            if sequence[-1] == model.eos_idx:
                finished.append((score, sequence))
                continue

            # 解码
            tgt = torch.LongTensor(sequence).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model.decode(encoder_output, tgt)

            # 取top-k预测
            probs = torch.softmax(output[:, -1], dim=-1)
            top_probs, top_indices = torch.topk(probs, beam_size)

            for i in range(beam_size):
                new_score = score + torch.log(top_probs[0][i])
                new_sequence = sequence + [top_indices[0][i].item()]
                new_candidates.append((new_score, new_sequence))

        # 保留top-k候选
        candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]

    # 返回最佳序列
    best_sequence = max(finished + candidates, key=lambda x: x[0])[1]
    return best_sequence
```

## 📚 学习资源

### 吴恩达课程
- [第五周：序列模型](https://www.coursera.org/learn/sequence-models)

### 经典论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al. (2018)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3

### 在线资源
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer代码实现](https://github.com/huggingface/transformers)
- [注意力机制可视化](https://transformer-viz.com/)

---
*最近更新: {{ .Lastmod.Format "2006-01-02" }}*