+++
title = "echo"
date = '2026-08-08T14:03:04+08:00'
description = "输出文本"
weight = 30
comments = false
+++

## 名称

echo — 输出文本

## 概要

```text
echo [-e] [-E] [-n] 参数...
```

## 描述

将参数输出到标准输出。默认在末尾添加换行。

## 选项

- `-e` — 启用转义序列解释
- `-E` — 禁用转义序列解释（默认）
- `-n` — 不输出末尾换行

## 转义序列

- `\n` — 换行
- `\t` — 制表符
- `\\` — 反斜杠
- `\0NNN` — 八进制字符

## 示例

```text
echo hello world
```

输出 "hello world"

```text
echo -e "line1\nline2"
```

输出两行

## 参见

[grep(1)](grep/)、[cat(1)](cat/)
