+++
date = '2026-05-19T23:49:18+08:00'
draft = false
title = 'Colab Unsloth SFT 快速训练！'
comments = true
tags = ['AI','计算机科学']
+++

# 序言
大部分人的GPU都没有训练AI的显存要求，而谷歌Colab完全可以解决这一点，谷歌Colab可以让你免费使用谷歌的服务器4个小时左右来进行训练；并且这是一个Jupyter Notebook环境。

这是它的地址： [https://colab.research.google.com/](https://colab.research.google.com/)

这是Unsloth的训练笔记本： [https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp)

# 准备工作
谷歌Colab可以直接通过谷歌账号登录，登录完毕后打开上述 Unsloth 笔记本，点击「Copy to Drive」保存到你自己的云盘，避免官方版本更新导致你的修改丢失。

## 修改代码
首先先不要急着运行运行时，先修改一下笔记本的代码，因为Unsloth 的官方笔记本是通用模板，你需要根据自己的需求修改两处核心内容：**模型** 和 **数据集**。

### 模型下载

首先，修改第二个代码框的这一部分：
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B", # 修改这里
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

将需要修改的部分修改为你要使用的模型，可以直接使用Hugging Face库名。

> btw：如果模型是 gated（需要申请权限），请在 Hugging Face 获取 Access Token 并填入 token 参数。

### 数据集准备

数据集的修改是这一步中最关键也最容易出错的地方。笔记本默认使用 `yahma/alpaca-cleaned`，但如果你要用自己的数据，就必须修改 `formatting_prompts_func`。

**示例：使用弱智吧数据集 (`LooksJuicy/ruozhiba`)**

这个数据集只有两个字段：`instruction` 和 `output`，没有单独的 `input` 字段。

原模板要求三个字段，所以我们需要调整映射关系——将 `instruction` 放入 `input` 位置，而 `instruction` 字段留空（也可以填入通用指令）。

修改后的代码如下：

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 必须加上，否则生成会无限继续

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # 将 instruction 放入 input，instruction 字段留空
        text = alpaca_prompt.format("", instruction, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("LooksJuicy/ruozhiba", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

> btw：如果你的数据集格式复杂（比如有 system、history 等多轮对话结构），强烈建议借助 ChatGPT 或 Kimi 等 AI 辅助写解析代码。大多数 SFT 数据集本质上都是把“输入”映射成“输出”，关键是按模型期望的 prompt template 拼装。

### 修改上传部分

修改笔记本中的 `Saving, loading finetuned models` 部分，它们提供了三种保存方式：

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| `merged_16bit` | 合并 LoRA 权重，导出为完整 16bit 模型 | 通用部署、后续训练 |
| `merged_4bit` | 合并后量化为 4bit | 低显存推理 |
| `lora` | 仅保存 LoRA 适配器权重 | 极速上传、与基座模型分开存储 |

如果你愿意可以使用Inference部分进行推理，现在可以直接跳到 Saving to float16 for VLLM 部分。

```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```
16bit那里可以保存为16bit模型，4bit则是4位模型，你也可以只存储lora适配器，但需要上传到单独的库，否则会发生冲突

将hf/model改为 你的用户名/模型名 的格式，例如：`safe049/llama-3-ruozhiba`

将token内的值设置为你的Hugging Face Token的值，请确保你的token有访问和写入权限

将你要导出的部分的False改为True即可上传

加入要上传16bit而不保存至colab服务器内，代码如下：
```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if True: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

```

你还可以保存为GGUF，对于低显存显卡，这是必要的

在GGUF / llama.cpp Conversion部分的修改与完整模型的上传类似，修改模型名与token，这里我建议上传16bit GGUF与q4_k_m GGUF，修改后代码如下：
```python
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if True: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if True: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )
```

现在所有代码修改都完毕了，可以开始训练

# 训练

一切准备就绪后，按顺序运行所有代码框：

1. **安装依赖**：自动完成 Unsloth 及其依赖的安装
2. **加载模型**：下载你指定的基座模型
3. **加载数据集**：下载并格式化你的数据集
4. **配置训练参数**：根据需要调整学习率、epoch、batch size 等
5. **执行训练**：`trainer_stats = trainer.train()`

> ⚠️ **实时监控 Loss**：训练开始后，留意 Loss 曲线。如果 Loss 长时间不再下降甚至反弹，说明模型可能已经过拟合，可以提前终止该单元格——训练状态仍然会被保存，不会丢失。

训练完成后，你可以先在 `Inference` 部分加载微调后的模型进行快速测试，确保效果符合预期。

最后，依次运行保存和上传部分的代码，把模型推送到 Hugging Face 仓库，然后我们就完事了

# 总结
Unsloth可以让你在小显存的GPU上快速训练AI模型，他还支持更多的训练方式，如DPO和DeepSeek的GRPO模式

Colab 的免费版对网络环境有一定要求（需要能访问 Google 服务），且会话有时限，建议在稳定的网络下进行训练。如果你有更复杂的训练任务，也可以考虑 Colab Pro 或 Pro+，获得更好的 GPU（如 V100、A100）和更长的运行时间