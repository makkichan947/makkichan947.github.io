+++
title = "groupmod"
date = '2026-08-08T14:03:11+08:00'
description = "修改组"
weight = 20
comments = false
+++

## 名称

groupmod — 修改组

## 概要

```text
groupmod [选项] 组名
```

## 描述

修改 `/export/cfg/groups.db` 中的组信息。需要 prime 组成员权限。

## 选项

- `-a, --add 用户` — 添加用户到组
- `-d, --del 用户` — 从组中移除用户
- `-n, --new 新名` — 重命名组

## 参见

[usermod(8)](usermod/)、[passwd(8)](passwd/)、[quell(8)](quell/)
