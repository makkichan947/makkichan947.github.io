+++
date = '2025-10-10T11:42:47+08:00'
draft = false
title = '使用Ollama本地部署大语言模型'
tags = ["AI", "工具", "教程"]
comments = true
+++

# 使用Ollama本地部署大语言模型

## 从云端到本地：为什么我选择Ollama

过去一年，我几乎每天都在跟各种大语言模型打交道。ChatGPT、Claude、Gemini……它们确实强大，但有几个问题始终让我不太舒服：**数据隐私**、**网络依赖**，以及**每月的订阅费用**。尤其是当我需要处理一些敏感代码或内部文档时，把内容上传到云端总让我心里打鼓。

于是我开始探索本地部署方案。试过`llama.cpp`，也试过`text-generation-webui`，它们都很强大，但配置起来多少有些繁琐——编译、依赖、环境变量，折腾一圈下来，我发现自己把时间花在了“让模型跑起来”而不是“用模型解决问题”上。

直到遇见了Ollama。**它把复杂的模型管理变得简单**。作为一个Arch Linux的忠实用户，这种“一条命令搞定”的体验让我瞬间有了归属感。

## 什么是Ollama？

Ollama是一个开源的、轻量级的大语言模型运行时。它封装了模型下载、推理、API服务等环节，让你可以在本地快速运行诸如Llama 2、Mistral、Code Llama等主流开源模型。

它的核心优势在于：

- **隐私安全**：所有数据留在本地，无需担心外泄
- **离线可用**：断网环境下依然正常工作
- **零成本**：无需API密钥，没有使用限额
- **开箱即用**：内置REST API，方便集成到各种应用中
- **模型复用**：支持从Hugging Face导入自定义模型

## 安装Ollama：我的Arch Way

我的主力发行版是Arch Linux，所以安装Ollama自然首选`pacman`：

```bash
# Arch Linux 官方社区仓库已收录
sudo pacman -S ollama
```

安装完成后，启动服务并设置开机自启：

```bash
# 启动ollama服务
systemctl --user start ollama

# 开机自动启动
systemctl --user enable ollama
```

如果你是Debian/Ubuntu用户，也可以使用官方脚本一键安装：

```bash
# 官方推荐方式（会添加apt仓库）
curl -fsSL https://ollama.com/install.sh | sh
```

或者使用Docker（适合隔离环境或快速试用）：

```bash
# 拉取并运行容器
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 进入容器执行命令
docker exec -it ollama ollama
```

无论哪种方式，Ollama默认会在`http://localhost:11434`开启API服务，随时等待你的调用。

## 模型选择：找到最适合你的那一款

Ollama官方模型库（[https://ollama.com/library](https://ollama.com/library)）提供了几十种模型。我根据自己的硬件和实际需求，筛选出以下几款常用模型：

### 通用对话：Llama 2 与 Mistral

```bash
# Llama 2 7B（平衡性能与资源）
ollama pull llama2:7b

# Mistral 7B（更小更快，表现接近Llama 2 13B）
ollama pull mistral
```

### 轻量级选手：Phi-2 与 TinyLlama

```bash
# 微软Phi-2（2.7B参数，适合低配设备）
ollama pull phi

# TinyLlama 1.1B（极致轻量，树莓派都能跑）
ollama pull tinyllama
```

**关于量化版本**：如果显存紧张，可以拉取量化后的模型，例如`llama2:7b-q4_0`，体积更小，速度更快，质量损失在可接受范围内。

## 基本使用：命令行与API

### 交互式对话

```bash
# 直接进入聊天模式
ollama run mistral

>>> 解释一下什么是“注意力机制”
```

### 单次生成

```bash
# 非交互式生成，适合脚本调用
ollama run codellama:7b "写一个Python函数，判断一个字符串是否为回文"
```

### REST API调用

Ollama内置了兼容OpenAI格式的API，可以用`curl`或任何HTTP客户端调用：

```bash
# 生成文本（流式响应）
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "用三句话介绍自己",
  "stream": false
}'

# 聊天补全（类似ChatGPT接口）
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}'
```

## 定制你的专属模型：Modelfile

Ollama最让我心动的地方在于**自定义模型**。通过编写`Modelfile`，你可以调整推理参数、设定系统提示词，甚至基于某个基础模型“嫁接”新的能力。

举个例子，我想创建一个专用于写Shell脚本的助手：

```bash
# 创建一个Modelfile
cat > ShellHelper <<EOF
FROM codellama:7b
PARAMETER temperature 0.3
PARAMETER top_p 0.9
SYSTEM 你是一个Shell脚本专家，只输出可运行的bash代码，不要添加额外解释。
EOF

# 构建自定义模型
ollama create shell-helper -f ShellHelper

# 运行
ollama run shell-helper "写一个循环重命名当前目录下所有.jpg文件的脚本"
```

这样一来，每次调用都能得到稳定、精简的代码输出，省去了反复调整提示词的麻烦。

## 性能优化：让模型跑得更快

### 显存管理

```bash
# 查看当前加载的模型
ollama ps

# 卸载不再使用的模型，释放显存
ollama stop model-name

# 删除本地模型文件（释放磁盘空间）
ollama rm model-name
```

### GPU加速

Ollama会自动检测CUDA或ROCm环境。如果有多张GPU，可以通过环境变量指定设备：

```bash
# 指定使用第一张卡
CUDA_VISIBLE_DEVICES=0 ollama run llama2
```

### 并发请求

Ollama默认支持并发处理，你可以通过调整`OLLAMA_NUM_PARALLEL`来增加并发数（视显存而定）：

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

## 实际应用：我是如何融入工作流的

### 1. Neovim 代码辅助

我在Neovim中配置了一个快捷键，选中代码后调用Ollama API进行解释或优化。搭配`codellama`模型，就像一个本地化的GitHub Copilot。

### 2. 命令行快速问答

写了一个`ask`脚本，封装`curl`请求，随时随地提问：

```bash
#!/bin/bash
ollama run mistral "$@"
```

现在遇到不熟悉的命令参数，直接`ask "tar命令的-xzf参数是什么意思"`，秒得答案。

### 3. 本地知识库检索（RAG）

配合向量数据库（如Chroma），将个人文档嵌入后，用Ollama做检索增强生成，实现私有化智能问答系统——数据永远不出本地。

## 与其他方案的对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Ollama** | 安装简单、模型丰富、API标准 | 定制性不如llama.cpp灵活 |
| **llama.cpp** | 极致轻量、支持CPU推理 | 需要编译，缺乏统一管理 |
| **text-generation-webui** | 图形界面、功能全面 | 依赖多，启动慢 |
| **vLLM** | 高吞吐，适合服务部署 | 配置复杂，侧重生产环境 |

对我这种既要效率又要掌控感的用户来说，Ollama恰好站在了“易用”和“可控”的平衡点上。

## 一些踩坑经验

- **模型下载缓慢**：可以设置代理，或者手动从Hugging Face下载后放到`~/.ollama/models`目录。
- **显存不足**：优先选择量化版本，或使用`--num-gpu`限制GPU层数。
- **中文支持**：有些模型中文能力较弱，推荐使用`qwen`或`Yi`系列（Ollama已支持）。
- **长时间运行**：建议定期重启服务，避免内存泄漏。

## 总结

Ollama让我彻底告别了“云端模型依赖症”。现在，我的主力机器上常驻着Mistral和Code Llama，它们随叫随到，回答稳定，隐私无忧。虽然偶尔会有“模型幻觉”，但比起每次都要上传文件的顾虑，这点小瑕疵完全可以接受。


## 相关资源

- [Ollama官网](https://ollama.ai)
- [GitHub仓库](https://github.com/jmorganca/ollama)
- [模型库](https://ollama.ai/library)
