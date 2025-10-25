+++
date = '2025-10-24T21:58:51+08:00'
draft = false
title = 'TensorFlow高级应用'
comments = true
weight = 3
+++

# TensorFlow高级应用

本章介绍TensorFlow在实际项目中的高级应用，包括计算机视觉、自然语言处理、强化学习、模型部署等领域的完整解决方案和最佳实践。

## 🎨 计算机视觉应用

### 目标检测 - YOLO实现
```python
import tensorflow as tf
import numpy as np
import cv2

def create_yolo_model(num_classes, input_shape=(416, 416, 3)):
    """创建YOLO模型"""
    inputs = tf.keras.Input(shape=input_shape)

    # 特征提取网络
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    # 检测头
    x = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)

    # 输出层：边界框、置信度和类别预测
    output = tf.keras.layers.Conv2D(
        num_classes + 5, 1, strides=1, padding='same', activation='linear'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

# YOLO损失函数
class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, grid_size=13, **kwargs):
        super(YOLOLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.grid_size = grid_size

    def call(self, y_true, y_pred):
        # 实现YOLO损失函数
        # 包括边界框回归损失、置信度损失和分类损失
        return total_loss

# 创建和训练YOLO模型
model = create_yolo_model(num_classes=20)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=YOLOLoss(num_classes=20),
    metrics=['accuracy']
)
```

### 图像分割 - U-Net实现
```python
def create_unet_model(input_shape=(256, 256, 3), num_classes=1):
    """创建U-Net模型"""
    inputs = tf.keras.Input(shape=input_shape)

    # 编码器
    c1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = tf.keras.layers.MaxPooling2D(pool_size=2)(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = tf.keras.layers.MaxPooling2D(pool_size=2)(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = tf.keras.layers.MaxPooling2D(pool_size=2)(c3)

    # 瓶颈层
    c4 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(c4)

    # 解码器
    u5 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c3])
    c5 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(u5)
    c5 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(u6)
    c6 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    c7 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(u7)
    c7 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c7)

    # 输出层
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(c7)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Dice损失函数
def dice_loss(y_true, y_pred):
    smooth = 1e-15
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

# 创建和训练U-Net
model = create_unet_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
)
```

## 📝 自然语言处理应用

### Transformer模型实现
```python
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len):
        super(TransformerModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_len, d_model)

        self.encoder_layers = [
            self.encoder_layer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]
        self.decoder_layers = [
            self.decoder_layer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def positional_encoding(self, max_len, d_model):
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates

        angle_rads = tf.where(i % 2 == 0, tf.sin(angle_rads), tf.cos(angle_rads))
        return angle_rads[tf.newaxis, ...]

    def encoder_layer(self, d_model, num_heads, d_ff):
        inputs = tf.keras.Input(shape=(None, d_model))

        # 多头注意力
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads, d_model)(inputs, inputs)
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # 前馈网络
        ffn_output = tf.keras.layers.Dense(d_ff, activation='relu')(out1)
        ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return tf.keras.Model(inputs=inputs, outputs=out2)

    def decoder_layer(self, d_model, num_heads, d_ff):
        inputs = tf.keras.Input(shape=(None, d_model))
        enc_output = tf.keras.Input(shape=(None, d_model))

        # 掩码多头注意力
        attn1 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)(inputs, inputs)
        attn1 = tf.keras.layers.Dropout(0.1)(attn1)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn1)

        # 编码器-解码器注意力
        attn2 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)(out1, enc_output)
        attn2 = tf.keras.layers.Dropout(0.1)(attn2)
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)

        # 前馈网络
        ffn_output = tf.keras.layers.Dense(d_ff, activation='relu')(out2)
        ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)

        return tf.keras.Model(inputs=[inputs, enc_output], outputs=out3)

    def call(self, inputs, targets=None, training=False):
        # 词嵌入 + 位置编码
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        # 编码器
        enc_output = x
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        # 解码器
        if targets is not None:
            y = self.embedding(targets)
            y *= tf.math.sqrt(tf.cast(tf.shape(y)[-1], tf.float32))
            y += self.pos_encoding[:, :tf.shape(y)[1], :]

            for layer in self.decoder_layers:
                y = layer([y, enc_output])

            outputs = self.final_layer(y)
        else:
            outputs = None

        return outputs, enc_output

# 创建Transformer模型
model = TransformerModel(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_len=100
)
```

### BERT预训练模型
```python
class BERTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len):
        super(BERTModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_len, d_model)
        self.segment_embedding = tf.keras.layers.Embedding(2, d_model)

        self.transformer_layers = [
            self.transformer_layer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ]

        self.mlm_head = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.nsp_head = tf.keras.layers.Dense(2, activation='softmax')

    def transformer_layer(self, d_model, num_heads, d_ff):
        inputs = tf.keras.Input(shape=(None, d_model))

        # 多头注意力
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads, d_model)(inputs, inputs)
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

        # 前馈网络
        ffn_output = tf.keras.layers.Dense(d_ff, activation='relu')(out1)
        ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return tf.keras.Model(inputs=inputs, outputs=out2)

    def call(self, inputs, segment_ids=None, masked_positions=None):
        # 词嵌入 + 位置编码 + 段编码
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        if segment_ids is not None:
            x += self.segment_embedding(segment_ids)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)

        # 掩码语言模型预测
        if masked_positions is not None:
            masked_outputs = tf.gather(x, masked_positions, axis=1, batch_dims=1)
            mlm_logits = self.mlm_head(masked_outputs)
        else:
            mlm_logits = None

        # 下一句预测
        cls_token = x[:, 0, :]  # [CLS] token
        nsp_logits = self.nsp_head(cls_token)

        return mlm_logits, nsp_logits

# 创建BERT模型
bert_model = BERTModel(
    vocab_size=30000,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    max_len=512
)
```

## 🎮 强化学习应用

### DQN实现
```python
import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """构建DQN网络"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机探索

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # 利用

    def replay(self, batch_size):
        """经验回放"""
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 使用DQN代理
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# 训练DQN
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            agent.update_target_model()
            print(f"Episode: {e}/{episodes}, Score: {time}")
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
```

### PPO实现
```python
class PPOAgent:
    def __init__(self, state_size, action_size, clip_ratio=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_ratio = clip_ratio

        # Actor网络
        self.actor = self._build_actor()
        # Critic网络
        self.critic = self._build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def _build_actor(self):
        """构建策略网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_critic(self):
        """构建价值网络"""
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_action(self, state):
        """获取动作"""
        state = np.reshape(state, [1, self.state_size])
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]

    def compute_advantages(self, rewards, values, next_values, dones):
        """计算优势函数"""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + 0.99 * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + 0.99 * 0.95 * next_non_terminal * last_gae_lam

        return advantages

    def train(self, states, actions, old_probs, advantages, returns):
        """PPO训练"""
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            values = self.critic(states)

            # 计算策略损失
            new_probs = tf.gather(probs, actions, axis=1, batch_dims=1)
            ratio = new_probs / old_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # 计算价值损失
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # 总损失
            loss = policy_loss + 0.5 * value_loss

        # 更新Actor
        actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # 更新Critic
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return loss
```

## 🚀 模型部署

### TensorFlow Serving
```python
# 保存模型用于Serving
import tensorflow as tf

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 训练模型（示例）
# model.fit(...)

# 保存模型
model.save('model/1')  # TensorFlow Serving格式

# 启动TensorFlow Serving
"""
tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=my_model \
    --model_base_path=/path/to/model
"""

# 使用REST API进行预测
import requests
import json

data = json.dumps({
    "signature_name": "serving_default",
    "inputs": {
        "dense_input": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    }
})

headers = {"content-type": "application/json"}
response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
print(response.json())
```

### TensorFlow Lite转换和部署
```python
# 转换为TensorFlow Lite
def convert_to_tflite(model, quantization='float32'):
    """转换为TensorFlow Lite格式"""

    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 提供代表性数据集用于量化
        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # 转换模型
    tflite_model = converter.convert()

    # 保存模型
    with open(f'model_{quantization}.tflite', 'wb') as f:
        f.write(tflite_model)

    return tflite_model

# 转换不同精度版本
model = create_model()  # 假设的模型创建函数

# 转换为不同精度
convert_to_tflite(model, 'float32')
convert_to_tflite(model, 'float16')
convert_to_tflite(model, 'int8')

# 评估模型大小和性能
import os

for precision in ['float32', 'float16', 'int8']:
    model_path = f'model_{precision}.tflite'
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024  # MB
        print(f"{precision}模型大小: {size:.2f} MB")
```

### TensorFlow.js转换
```python
# 转换为TensorFlow.js格式
import tensorflowjs as tfjs

# 保存为TensorFlow.js格式
tfjs.converters.save_keras_model(model, 'tfjs_model/')

# 或者转换为分层格式
tfjs.converters.save_keras_model(model, 'tfjs_model/', quantization_dtype=tfjs.quantization_config.INT8)

# HTML中使用模型
"""
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
</head>
<body>
    <script>
        async function loadModel() {
            // 加载模型
            const model = await tf.loadLayersModel('tfjs_model/model.json');

            // 准备输入数据
            const input = tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);

            // 进行预测
            const prediction = model.predict(input);
            prediction.print();
        }

        loadModel();
    </script>
</body>
</html>
"""
```

## 📊 生产环境最佳实践

### 模型监控
```python
class ModelMonitor:
    def __init__(self, model, metrics=['accuracy', 'latency', 'throughput']):
        self.model = model
        self.metrics = metrics
        self.predictions = []
        self.true_labels = []
        self.latencies = []

    def predict_with_monitoring(self, inputs, true_labels=None):
        """带监控的预测"""
        import time

        start_time = time.time()

        # 进行预测
        predictions = self.model.predict(inputs)

        end_time = time.time()
        latency = end_time - start_time

        # 记录指标
        self.latencies.append(latency)
        self.predictions.extend(predictions)

        if true_labels is not None:
            self.true_labels.extend(true_labels)

        return predictions

    def generate_report(self):
        """生成监控报告"""
        report = {}

        if self.latencies:
            report['avg_latency'] = np.mean(self.latencies)
            report['p95_latency'] = np.percentile(self.latencies, 95)
            report['throughput'] = len(self.latencies) / sum(self.latencies)

        if self.true_labels and self.predictions:
            predictions = np.array(self.predictions)
            true_labels = np.array(self.true_labels)

            if len(predictions.shape) > 1:
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(true_labels, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
                true_classes = true_labels

            report['accuracy'] = np.mean(pred_classes == true_classes)

        return report

# 使用模型监控
monitor = ModelMonitor(model)

# 模拟生产环境预测
for i in range(100):
    test_input = np.random.rand(1, 10)
    true_label = np.random.randint(0, 2, 1)

    prediction = monitor.predict_with_monitoring(test_input, true_label)

# 生成报告
report = monitor.generate_report()
print("模型监控报告:", report)
```

### A/B测试框架
```python
class ABTestFramework:
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split

        self.model_a_metrics = []
        self.model_b_metrics = []

    def predict(self, inputs, true_labels=None):
        """A/B测试预测"""
        results = []

        for i, (input_data, true_label) in enumerate(zip(inputs, true_labels or [])):
            # 随机分配流量
            if np.random.random() < self.traffic_split:
                model = self.model_a
                model_name = 'A'
            else:
                model = self.model_b
                model_name = 'B'

            # 进行预测
            prediction = model.predict(np.expand_dims(input_data, 0))[0]

            results.append({
                'model': model_name,
                'prediction': prediction,
                'true_label': true_label
            })

        return results

    def evaluate_models(self, results):
        """评估模型性能"""
        model_a_results = [r for r in results if r['model'] == 'A']
        model_b_results = [r for r in results if r['model'] == 'B']

        def calculate_metrics(results):
            if not results:
                return {}

            predictions = np.array([r['prediction'] for r in results])
            true_labels = np.array([r['true_label'] for r in results])

            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(true_labels, axis=1)

            accuracy = np.mean(pred_classes == true_classes)

            return {'accuracy': accuracy, 'sample_size': len(results)}

        metrics_a = calculate_metrics(model_a_results)
        metrics_b = calculate_metrics(model_b_results)

        return {'model_a': metrics_a, 'model_b': metrics_b}

# 使用A/B测试框架
ab_test = ABTestFramework(model_v1, model_v2, traffic_split=0.5)

# 模拟A/B测试
test_inputs = [np.random.rand(10) for _ in range(1000)]
test_labels = [np.random.randint(0, 2, 10) for _ in range(1000)]

results = ab_test.predict(test_inputs, test_labels)
metrics = ab_test.evaluate_models(results)

print("A/B测试结果:", metrics)
```

## 📚 学习资源

### 官方文档
- [TensorFlow Serving文档](https://www.tensorflow.org/tfx/guide/serving)
- [TensorFlow Lite文档](https://www.tensorflow.org/lite)
- [TensorFlow.js文档](https://www.tensorflow.org/js)

### 吴恩达课程
- 深度学习课程中关于实际应用的部分

### 经典项目
- [TensorFlow Models](https://github.com/tensorflow/models) - 官方模型库
- [TensorFlow Hub](https://tfhub.dev/) - 预训练模型仓库
- [Kaggle竞赛](https://www.kaggle.com/competitions) - 实践项目

## 🔧 部署最佳实践

### 容器化部署
```dockerfile
# Dockerfile
FROM tensorflow/serving:latest

COPY model/ /models/my_model
ENV MODEL_NAME=my_model

# 启动命令
CMD ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=my_model", "--model_base_path=/models/my_model"]
```

### 模型版本管理
```python
# 模型版本管理
import tensorflow as tf

class ModelVersionManager:
    def __init__(self, model_base_path):
        self.model_base_path = model_base_path
        self.versions = {}

    def save_model_version(self, model, version, metadata=None):
        """保存模型版本"""
        version_path = f"{self.model_base_path}/{version}"

        # 保存模型
        model.save(version_path)

        # 保存元数据
        if metadata:
            with open(f"{version_path}/metadata.json", 'w') as f:
                json.dump(metadata, f)

        self.versions[version] = {
            'path': version_path,
            'metadata': metadata,
            'timestamp': time.time()
        }

    def load_model_version(self, version):
        """加载模型版本"""
        if version not in self.versions:
            raise ValueError(f"版本 {version} 不存在")

        version_path = self.versions[version]['path']
        return tf.keras.models.load_model(version_path)

    def list_versions(self):
        """列出版本"""
        return list(self.versions.keys())

    def rollback(self, version):
        """回滚到指定版本"""
        model = self.load_model_version(version)
        self.save_model_version(model, 'current', {'rollback_from': version})
        return model

# 使用版本管理器
version_manager = ModelVersionManager('./models')

# 保存不同版本
for i, model in enumerate([model_v1, model_v2, model_v3]):
    version_manager.save_model_version(
        model,
        version=f"v{i+1}",
        metadata={'description': f'模型版本{i+1}', 'accuracy': 0.95 + i*0.01}
    )
```

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*