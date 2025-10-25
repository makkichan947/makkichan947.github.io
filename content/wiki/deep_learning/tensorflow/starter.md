+++
date = '2025-10-24T21:58:35+08:00'
draft = false
title = 'TensorFlow基础教程'
comments = true
weight = 1
+++

# TensorFlow基础教程

本教程将从零开始介绍TensorFlow的基础知识，包括环境搭建、基本概念、简单模型构建和训练。通过本教程，你将能够使用TensorFlow构建和训练基本的机器学习模型。

## 🛠️ 环境搭建

### 安装TensorFlow
```bash
# CPU版本
pip install tensorflow

# GPU版本（需要CUDA支持）
pip install tensorflow[and-cuda]

# 验证安装
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU' if tf.test.is_gpu_available() else 'CPU'))"
```

### 推荐开发环境
```bash
# 创建虚拟环境
conda create -n tf_env python=3.9
conda activate tf_env

# 安装TensorFlow和相关依赖
pip install tensorflow numpy matplotlib jupyter pandas scikit-learn

# 启动Jupyter Notebook
jupyter notebook
```

## 🎯 基础概念

### Hello TensorFlow
```python
import tensorflow as tf

# 打印TensorFlow版本
print(f"TensorFlow版本: {tf.__version__}")

# 创建常量张量
hello = tf.constant("Hello, TensorFlow!")
print(f"常量: {hello.numpy()}")

# 基本运算
a = tf.constant(5)
b = tf.constant(3)
print(f"5 + 3 = {tf.add(a, b).numpy()}")
print(f"5 * 3 = {tf.multiply(a, b).numpy()}")
```

### 张量操作
```python
import tensorflow as tf
import numpy as np

# 创建不同类型的张量
scalar = tf.constant(42)  # 标量
vector = tf.constant([1, 2, 3, 4, 5])  # 向量
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])  # 矩阵
tensor_3d = tf.constant(np.random.rand(2, 3, 4))  # 三维张量

print(f"标量形状: {scalar.shape}")
print(f"向量形状: {vector.shape}")
print(f"矩阵形状: {matrix.shape}")
print(f"三维张量形状: {tensor_3d.shape}")

# 张量运算
x = tf.constant([1, 2, 3, 4, 5])
y = tf.constant([6, 7, 8, 9, 10])

# 元素级运算
print(f"x + y = {tf.add(x, y).numpy()}")
print(f"x * y = {tf.multiply(x, y).numpy()}")

# 广播机制
matrix = tf.constant([[1, 2], [3, 4]])
vector = tf.constant([5, 6])
print(f"矩阵 + 向量广播结果:\n{tf.add(matrix, vector).numpy()}")
```

## 🏗️ 第一个神经网络

### 线性回归模型
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# 编译模型
model.compile(
    optimizer='sgd',  # 随机梯度下降
    loss='mse',       # 均方误差
    metrics=['mae']   # 平均绝对误差
)

# 模型摘要
model.summary()

# 训练模型
history = model.fit(X, y, epochs=100, verbose=0)

# 预测
y_pred = model.predict(X)

# 可视化结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='真实数据')
plt.plot(X, y_pred, color='red', label='预测结果')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归结果')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')

plt.tight_layout()
plt.show()

print(f"模型权重: {model.weights[0].numpy().flatten()}")
print(f"模型偏置: {model.weights[1].numpy()}")
```

### 分类模型 - MNIST手写数字识别
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 784) / 255.0  # 展平并归一化
x_test = x_test.reshape(-1, 784) / 255.0
y_train = to_categorical(y_train, 10)  # one-hot编码
y_test = to_categorical(y_test, 10)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),  # Dropout防止过拟合
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 输出层
])

# 编译模型
model.compile(
    optimizer='adam',  # Adam优化器
    loss='categorical_crossentropy',  # 交叉熵损失
    metrics=['accuracy']  # 准确率
)

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,  # 验证集
    verbose=1
)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集准确率: {test_acc:.4f}")

# 预测
predictions = model.predict(x_test[:5])
predicted_labels = tf.argmax(predictions, axis=1).numpy()
true_labels = tf.argmax(y_test[:5], axis=1).numpy()

print("预测结果:", predicted_labels)
print("真实标签:", true_labels)
```

## 📊 数据处理

### 使用tf.data API
```python
import tensorflow as tf
import numpy as np

# 创建数据集
def create_dataset():
    # 生成随机数据
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    # 创建tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # 数据预处理
    def preprocess(x, y):
        # 归一化
        x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
        return x, y

    # 应用预处理
    dataset = dataset.map(preprocess)

    # 打乱和批处理
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(32)

    # 重复和预取
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# 使用数据集训练
dataset = create_dataset()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练
model.fit(dataset, epochs=5, steps_per_epoch=10)
```

### 图像数据增强
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 创建图像数据生成器
datagen = ImageDataGenerator(
    rotation_range=20,      # 随机旋转角度
    width_shift_range=0.1,  # 水平平移
    height_shift_range=0.1, # 垂直平移
    shear_range=0.1,        # 剪切变换
    zoom_range=0.1,         # 缩放
    horizontal_flip=True,   # 水平翻转
    fill_mode='nearest'     # 填充模式
)

# 应用数据增强
# datagen.fit(x_train)  # 如果需要的话
# enhanced_data = datagen.flow(x_train, y_train, batch_size=32)
```

## 🎨 自定义模型和层

### 自定义层
```python
import tensorflow as tf

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # 创建权重
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # 前向传播
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        # 序列化配置
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config

# 使用自定义层
model = tf.keras.Sequential([
    CustomDense(128, activation='relu', input_shape=(784,)),
    CustomDense(64, activation='relu'),
    CustomDense(10, activation='softmax')
])
```

### 自定义模型
```python
class CustomModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.dense3(x)

# 使用自定义模型
model = CustomModel(num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## 🔧 模型保存和加载

### 保存和加载模型
```python
import tensorflow as tf

# 构建和训练模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型（这里使用随机数据作为示例）
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

model.fit(X_train, y_train, epochs=5, verbose=0)

# 保存完整模型
model.save('my_model.h5')  # HDF5格式
model.save('my_model')     # TensorFlow SavedModel格式

# 加载模型
loaded_model_h5 = tf.keras.models.load_model('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model')

# 比较预测结果
test_data = np.random.rand(5, 10)
original_pred = model.predict(test_data)
loaded_pred = loaded_model.predict(test_data)

print(f"预测结果一致: {np.allclose(original_pred, loaded_pred)}")
```

### 保存和加载权重
```python
# 保存模型权重
model.save_weights('model_weights.h5')

# 创建相同架构的新模型
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 加载权重
new_model.load_weights('model_weights.h5')

# 验证权重是否相同
original_weights = model.get_weights()
loaded_weights = new_model.get_weights()

for i, (orig, loaded) in enumerate(zip(original_weights, loaded_weights)):
    print(f"权重{i}一致: {np.allclose(orig, loaded)}")
```

## 📈 模型评估和可视化

### 评估指标
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 预测测试集
y_pred = model.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_true = tf.argmax(y_test, axis=1).numpy()

# 分类报告
print("分类报告:")
print(classification_report(y_true, y_pred_classes))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()
```

### 训练历史可视化
```python
import matplotlib.pyplot as plt

# 训练历史
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

# 绘制训练曲线
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练和验证损失')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('训练和验证准确率')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🚀 实际项目：房价预测

### 完整项目代码
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成模拟房价数据
np.random.seed(42)
n_samples = 1000

# 特征：面积、房间数、年龄、距离市中心距离
areas = np.random.normal(100, 20, n_samples)  # 平方米
rooms = np.random.randint(1, 6, n_samples)     # 房间数
ages = np.random.randint(0, 50, n_samples)     # 房屋年龄
distances = np.random.normal(10, 3, n_samples) # 距离市中心（公里）

# 目标：房价（万元）
prices = 50 + 0.8 * areas + 10 * rooms - 0.5 * ages - 2 * distances + np.random.normal(0, 10, n_samples)

# 创建DataFrame
data = pd.DataFrame({
    'area': areas,
    'rooms': rooms,
    'age': ages,
    'distance': distances,
    'price': prices
})

# 数据预处理
features = ['area', 'rooms', 'age', 'distance']
X = data[features].values
y = data['price'].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # 回归问题，输出层无激活函数
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

# 训练模型
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=1
)

# 评估模型
test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集MAE: {test_mae:.2f}万元")
print(f"测试集MSE: {test_mse:.2f}万元²")

# 预测
predictions = model.predict(X_test[:5])
print("\n预测结果:")
for i in range(5):
    print(f"真实价格: {y_test[i]:.2f}万元, 预测价格: {predictions[i][0]:.2f}万元")

# 保存模型
model.save('house_price_model.h5')
print("\n模型已保存为 'house_price_model.h5'")
```

## 🎯 最佳实践

### 代码组织
```python
# 推荐的项目结构
house_price_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
├── config/
│   └── config.yaml
└── requirements.txt
```

### 调试技巧
```python
# 启用TensorFlow调试
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少警告信息

# 检查模型结构
model.summary()

# 检查张量形状
print(f"输入形状: {x_train.shape}")
print(f"输出形状: {model.predict(x_train[:1]).shape}")

# 使用tf.debugging
x = tf.constant([1, 2, 3])
tf.debugging.assert_shapes([
    (x, ('N',)),  # 断言x的形状
])
```

## 📚 学习资源

### 官方教程
- [TensorFlow官方教程](https://www.tensorflow.org/tutorials)
- [Keras文档](https://keras.io/getting_started/)
- [TensorFlow 2.0快速入门](https://www.tensorflow.org/tutorials/quickstart/beginner)

### 吴恩达课程
- 深度学习课程中关于TensorFlow的部分

### 实践项目
- [TensorFlow Examples](https://github.com/tensorflow/examples)
- [Kaggle竞赛](https://www.kaggle.com/competitions)
- [Google Colab](https://colab.research.google.com/)

## 🔧 常见问题

### GPU内存不足
```python
# 限制GPU内存使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### 模型不收敛
```python
# 调整学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 使用学习率调度
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 数据预处理
```python
# 正确的数据预处理流程
def preprocess_data(X, y):
    # 1. 处理缺失值
    # 2. 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return X_train, X_test, y_train, y_test, scaler
```

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*