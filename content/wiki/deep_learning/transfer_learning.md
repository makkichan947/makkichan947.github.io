+++
date = '2025-10-20T10:15:00+08:00'
draft = false
title = '迁移学习'
comments = true
weight = 8
+++

# 迁移学习

迁移学习（Transfer Learning）是深度学习中的重要技术，通过将在一个任务上学到的知识应用到另一个相关任务中，显著减少训练时间并提高模型性能，特别适用于数据量有限的场景。

## 🎯 迁移学习基础

### 迁移学习概念

**定义**：
迁移学习是指将从一个任务（源任务）中学到的知识应用到另一个相关任务（目标任务）中的学习过程。

**核心思想**：
- **特征复用**：底层特征（边缘、纹理）在不同任务间是通用的
- **知识迁移**：高层语义特征可以从源任务迁移到目标任务
- **微调**：在目标任务上微调预训练模型的参数

### 迁移学习优势

**传统机器学习 vs 迁移学习**：
```python
# 传统方法：从零开始训练
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=100)  # 需要大量数据和时间

# 迁移学习：使用预训练模型
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # 冻结预训练权重

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10)  # 快速收敛
```

## 🏗️ 迁移学习策略

### 特征提取 (Feature Extraction)

**方法**：使用预训练模型作为特征提取器，训练新的分类器。

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def create_feature_extractor(base_model_name='resnet50', num_classes=10):
    """创建特征提取模型"""

    # 加载预训练模型（不包含顶部分类层）
    if base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'vgg16':
        base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # 冻结预训练层
    base_model.trainable = False

    # 添加新的分类层
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# 创建特征提取模型
model = create_feature_extractor('resnet50', num_classes=10)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
```

### 微调 (Fine-tuning)

**方法**：解冻部分预训练层，在目标任务上进行微调。

```python
def create_finetune_model(base_model_name='resnet50', num_classes=10, unfreeze_layers=10):
    """创建微调模型"""

    # 加载预训练模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 冻结所有层
    base_model.trainable = False

    # 构建模型
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译并训练特征提取器
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    # 解冻顶层
    base_model.trainable = True

    # 只训练最后几层
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # 使用较小的学习率进行微调
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# 创建微调模型
finetune_model = create_finetune_model('resnet50', num_classes=10, unfreeze_layers=20)
finetune_model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
```

### 领域自适应 (Domain Adaptation)

**方法**：减少源领域和目标领域之间的分布差异。

```python
class DomainAdaptationModel(tf.keras.Model):
    def __init__(self, feature_extractor, task_classifier, domain_classifier):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.domain_classifier = domain_classifier

    def call(self, inputs, lambda_adapt=1.0):
        # 特征提取
        features = self.feature_extractor(inputs)

        # 任务分类
        task_outputs = self.task_classifier(features)

        # 领域分类（梯度反转）
        domain_outputs = self.domain_classifier(features)

        return task_outputs, domain_outputs, features

def create_domain_adaptation_model(input_shape, num_classes):
    """创建领域自适应模型"""

    # 特征提取器
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    # 任务分类器
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 领域分类器
    domain_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return DomainAdaptationModel(feature_extractor, task_classifier, domain_classifier)

# 自定义损失函数
def domain_adaptation_loss(y_true_task, y_pred_task, y_true_domain, y_pred_domain, lambda_adapt=1.0):
    """领域自适应损失"""

    # 任务损失
    task_loss = tf.keras.losses.categorical_crossentropy(y_true_task, y_pred_task)

    # 领域损失（反转标签）
    domain_labels = tf.ones_like(y_true_domain) - y_true_domain  # 反转标签
    domain_loss = tf.keras.losses.binary_crossentropy(domain_labels, y_pred_domain)

    # 总损失
    total_loss = tf.reduce_mean(task_loss) + lambda_adapt * tf.reduce_mean(domain_loss)

    return total_loss
```

## 🎨 预训练模型应用

### ImageNet预训练模型

```python
# 常用ImageNet预训练模型
models_dict = {
    'resnet50': tf.keras.applications.ResNet50,
    'resnet101': tf.keras.applications.ResNet101,
    'resnet152': tf.keras.applications.ResNet152,
    'vgg16': tf.keras.applications.VGG16,
    'vgg19': tf.keras.applications.VGG19,
    'inceptionv3': tf.keras.applications.InceptionV3,
    'xception': tf.keras.applications.Xception,
    'mobilenet': tf.keras.applications.MobileNet,
    'mobilenetv2': tf.keras.applications.MobileNetV2,
    'densenet121': tf.keras.applications.DenseNet121,
    'densenet169': tf.keras.applications.DenseNet169,
    'densenet201': tf.keras.applications.DenseNet201,
    'nasnetmobile': tf.keras.applications.NASNetMobile,
    'nasnetlarge': tf.keras.applications.NASNetLarge,
    'efficientnetb0': tf.keras.applications.EfficientNetB0,
    'efficientnetb1': tf.keras.applications.EfficientNetB1,
    'efficientnetb7': tf.keras.applications.EfficientNetB7
}

def load_pretrained_model(model_name, input_shape=(224, 224, 3), include_top=False):
    """加载预训练模型"""

    if model_name not in models_dict:
        raise ValueError(f"不支持的模型: {model_name}")

    model_class = models_dict[model_name]

    # 加载预训练模型
    base_model = model_class(
        weights='imagenet',
        include_top=include_top,
        input_shape=input_shape
    )

    return base_model

# 使用不同预训练模型
for model_name in ['resnet50', 'vgg16', 'inceptionv3']:
    print(f"\n=== {model_name.upper()} ===")
    base_model = load_pretrained_model(model_name)
    print(f"输入形状: {base_model.input_shape}")
    print(f"输出形状: {base_model.output_shape}")
    print(f"参数数量: {base_model.count_params()}")
```

### 模型性能对比

```python
def compare_models(x_train, y_train, x_test, y_test, model_names=['resnet50', 'vgg16', 'mobilenet']):
    """对比不同预训练模型的性能"""

    results = {}

    for model_name in model_names:
        print(f"\n训练 {model_name}...")

        # 加载预训练模型
        base_model = load_pretrained_model(model_name)
        base_model.trainable = False

        # 构建完整模型
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        history = model.fit(
            x_train, y_train,
            epochs=10,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # 评估模型
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results[model_name] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'parameters': model.count_params(),
            'history': history.history
        }

        print(f"{model_name} 测试准确率: {test_acc:.4f}")

    return results

# 可视化对比结果
def plot_comparison(results):
    """可视化模型对比结果"""

    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    params = [results[name]['parameters'] for name in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 准确率对比
    ax1.bar(model_names, accuracies)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # 参数数量对比
    ax2.bar(model_names, params)
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Model Size Comparison')
    ax2.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')

    for i, v in enumerate(params):
        ax2.text(i, v + 1000, f'{v/1e6:.1f}M', ha='center')

    plt.tight_layout()
    plt.show()

# 运行对比
results = compare_models(x_train, y_train, x_test, y_test)
plot_comparison(results)
```

## 📝 自然语言处理中的迁移学习

### BERT迁移学习

```python
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

def create_bert_model(num_classes=2, max_len=128):
    """创建基于BERT的迁移学习模型"""

    # 加载BERT预处理模型
    bert_preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )

    # 加载BERT编码器
    bert_encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=True
    )

    # 构建模型
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # 使用BERT的CLS token输出
    cls_output = outputs['pooled_output']

    # 添加分类层
    dropout = tf.keras.layers.Dropout(0.1)(cls_output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=text_input, outputs=output)

    return model

# 创建BERT模型
bert_model = create_bert_model(num_classes=2)

# 编译模型
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
history = bert_model.fit(
    x_train_text, y_train,
    validation_data=(x_val_text, y_val),
    epochs=3,
    batch_size=16
)
```

### GPT模型微调

```python
class GPTFineTuner:
    def __init__(self, model_name='gpt2', num_classes=None):
        self.model_name = model_name
        self.num_classes = num_classes

        # 加载预训练GPT模型
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.model = self._load_gpt_model()

    def _load_gpt_model(self):
        """加载GPT模型"""
        if self.model_name == 'gpt2':
            # 使用Hugging Face的transformers库
            from transformers import TFGPT2Model, GPT2Tokenizer

            model = TFGPT2Model.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            return model
        else:
            # 自定义GPT模型
            return self._create_custom_gpt()

    def _create_custom_gpt(self):
        """创建自定义GPT模型"""
        # 简化的GPT架构
        inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

        # 词嵌入
        embedding = tf.keras.layers.Embedding(30000, 768)(inputs)

        # 位置编码
        position_embedding = tf.keras.layers.Embedding(1024, 768)(tf.range(1024))
        x = embedding + position_embedding

        # Transformer块
        for _ in range(12):
            # 多头注意力
            attn_output = tf.keras.layers.MultiHeadAttention(12, 768)(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)

            # 前馈网络
            ffn_output = tf.keras.layers.Dense(3072, activation='gelu')(x)
            ffn_output = tf.keras.layers.Dense(768)(ffn_output)
            x = tf.keras.layers.LayerNormalization()(x + ffn_output)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def fine_tune(self, texts, labels, epochs=3):
        """微调GPT模型"""

        # 编码文本
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

        # 构建微调模型
        inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
        gpt_outputs = self.model(inputs)

        # 取最后一个token的输出
        last_token_output = gpt_outputs[:, -1, :]

        # 分类层
        if self.num_classes:
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(last_token_output)
        else:
            outputs = tf.keras.layers.Dense(768, activation='linear')(last_token_output)

        fine_tuned_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 编译模型
        fine_tuned_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss='categorical_crossentropy' if self.num_classes else 'mse',
            metrics=['accuracy'] if self.num_classes else []
        )

        # 训练模型
        fine_tuned_model.fit(
            encoded_texts['input_ids'], labels,
            epochs=epochs,
            batch_size=8,
            validation_split=0.1
        )

        return fine_tuned_model
```

## 🎯 实际应用案例

### 医学图像分类

```python
def create_medical_image_model(base_model='resnet50', num_classes=2):
    """创建医学图像分类模型"""

    # 加载预训练模型
    if base_model == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

    # 冻结预训练层
    base_model.trainable = False

    # 构建模型
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def train_medical_model(x_train, y_train, x_val, y_val):
    """训练医学图像模型"""

    # 数据增强
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # 创建模型
    model = create_medical_image_model('densenet121', num_classes=2)

    # 第一阶段：训练分类器
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    print("第一阶段：训练分类器...")
    model.fit(
        train_datagen.flow(x_train, y_train, batch_size=32),
        epochs=20,
        validation_data=val_datagen.flow(x_val, y_val, batch_size=32),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )

    # 第二阶段：微调
    print("第二阶段：微调模型...")

    # 解冻部分层
    for layer in model.layers[0].layers[-20:]:
        layer.trainable = True

    # 使用较小的学习率
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    model.fit(
        train_datagen.flow(x_train, y_train, batch_size=16),
        epochs=30,
        validation_data=val_datagen.flow(x_val, y_val, batch_size=16),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('finetuned_model.h5', save_best_only=True)
        ]
    )

    return model

# 使用示例
model = train_medical_model(x_train_medical, y_train_medical, x_val_medical, y_val_medical)
```

### 文本分类

```python
def create_text_classification_model(model_type='bert', num_classes=5):
    """创建文本分类模型"""

    if model_type == 'bert':
        # 使用BERT
        bert_preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        )
        bert_encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=True
        )

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)
        cls_output = outputs['pooled_output']

    elif model_type == 'universal_sentence_encoder':
        # 使用Universal Sentence Encoder
        use_layer = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            trainable=False
        )

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        cls_output = use_layer(text_input)

    # 分类层
    dropout = tf.keras.layers.Dropout(0.1)(cls_output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=text_input, outputs=output)

    return model

def train_text_model(texts, labels, model_type='bert'):
    """训练文本分类模型"""

    # 编码标签
    label_encoder = tf.keras.utils.to_categorical if len(np.unique(labels)) > 2 else lambda x: x
    y_encoded = label_encoder(labels)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(
        texts, y_encoded, test_size=0.2, random_state=42
    )

    # 创建模型
    model = create_text_classification_model(model_type, num_classes=len(np.unique(labels)))

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5 if model_type == 'bert' else 0.001),
        loss='categorical_crossentropy' if len(np.unique(labels)) > 2 else 'binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5 if model_type == 'bert' else 20,
        batch_size=16 if model_type == 'bert' else 32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_text_model.h5', save_best_only=True)
        ]
    )

    return model, history

# 使用示例
model, history = train_text_model(texts, labels, model_type='bert')
```

## 📊 模型评估和优化

### 评估迁移学习模型

```python
def evaluate_transfer_model(model, x_test, y_test, class_names=None):
    """评估迁移学习模型"""

    # 预测
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 分类报告
    print("分类报告:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    # ROC曲线（二分类）
    if y_pred.shape[1] == 2:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return {
        'accuracy': np.mean(y_pred_classes == y_true),
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred_classes, output_dict=True)
    }

# 使用评估函数
results = evaluate_transfer_model(model, x_test, y_test, class_names=['class1', 'class2', 'class3'])
```

### 超参数优化

```python
def optimize_transfer_learning(x_train, y_train, x_val, y_val):
    """迁移学习超参数优化"""

    def objective(trial):
        # 超参数搜索空间
        base_model_name = trial.suggest_categorical('base_model', ['resnet50', 'vgg16', 'mobilenetv2'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        dense_units = trial.suggest_categorical('dense_units', [128, 256, 512])
        unfreeze_layers = trial.suggest_int('unfreeze_layers', 0, 50)

        # 创建模型
        base_model = load_pretrained_model(base_model_name)
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型
        model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=0)

        # 微调
        if unfreeze_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-unfreeze_layers]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=0)

        # 评估
        _, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return accuracy

    # 使用Optuna进行超参数优化
    import optuna

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f"最佳参数: {study.best_params}")
    print(f"最佳准确率: {study.best_value}")

    return study.best_params

# 运行超参数优化
best_params = optimize_transfer_learning(x_train, y_train, x_val, y_val)
```

## 📚 学习资源

### 官方文档
- [TensorFlow迁移学习指南](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Keras预训练模型](https://keras.io/applications/)
- [TensorFlow Hub](https://tfhub.dev/)

### 经典论文
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) - AlexNet
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - VGG
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet
- [Going deeper with convolutions](https://arxiv.org/abs/1409.4842) - GoogLeNet

### 吴恩达课程
- 深度学习课程中关于迁移学习的部分

## 🎯 最佳实践

### 数据准备
```python
def prepare_data_for_transfer_learning(images, labels, validation_split=0.2):
    """为迁移学习准备数据"""

    # 调整图像大小
    target_size = (224, 224)  # 大多数预训练模型的输入大小
    resized_images = tf.image.resize(images, target_size)

    # 图像预处理（ImageNet标准化）
    preprocessed_images = tf.keras.applications.resnet50.preprocess_input(resized_images)

    # 编码标签
    if len(np.unique(labels)) > 2:
        encoded_labels = tf.keras.utils.to_categorical(labels)
    else:
        encoded_labels = labels

    # 划分数据集
    num_val_samples = int(len(images) * validation_split)
    x_train = preprocessed_images[:-num_val_samples]
    y_train = encoded_labels[:-num_val_samples]
    x_val = preprocessed_images[-num_val_samples:]
    y_val = encoded_labels[-num_val_samples:]

    return x_train, y_train, x_val, y_val

# 使用数据准备函数
x_train, y_train, x_val, y_val = prepare_data_for_transfer_learning(images, labels)
```

### 模型选择指南
```python
def select_best_model(dataset_size, num_classes, time_budget):
    """根据数据集大小和时间预算选择最佳模型"""

    model_recommendations = {
        'small_dataset': {
            'models': ['mobilenetv2', 'efficientnetb0', 'resnet50'],
            'strategy': 'feature_extraction',
            'epochs': 20
        },
        'medium_dataset': {
            'models': ['resnet50', 'densenet121', 'efficientnetb3'],
            'strategy': 'fine_tuning',
            'epochs': 50
        },
        'large_dataset': {
            'models': ['resnet152', 'densenet201', 'efficientnetb7'],
            'strategy': 'full_training',
            'epochs': 100
        }
    }

    # 根据数据集大小选择推荐
    if dataset_size < 1000:
        recommendation = model_recommendations['small_dataset']
    elif dataset_size < 10000:
        recommendation = model_recommendations['medium_dataset']
    else:
        recommendation = model_recommendations['large_dataset']

    # 根据时间预算调整
    if time_budget < 60:  # 1小时
        recommendation['models'] = recommendation['models'][:1]
        recommendation['epochs'] = min(recommendation['epochs'], 10)

    return recommendation

# 使用模型选择指南
recommendation = select_best_model(dataset_size=5000, num_classes=10, time_budget=120)
print(f"推荐模型: {recommendation['models']}")
print(f"推荐策略: {recommendation['strategy']}")
print(f"推荐训练轮数: {recommendation['epochs']}")
```

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*