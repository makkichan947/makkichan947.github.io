+++
title = "quell"
date = '2026-08-08T14:03:11+08:00'
description = "以 prime 组权限执行命令"
weight = 40
comments = false
+++

## 名称

quell — 以 prime 组权限执行命令

## 概要

```text
quell 命令 [参数...]
```

## 描述

以 root 权限（euid=0）执行指定命令。调用者必须在 prime 组（GID 1001）中。这是 M4KK1 中 sudo 的替代方案。

## 示例

```text
quell mount /dev/yafs0 /mnt
```

挂载文件系统

```text
quell usermod -a --group prime alice
```

添加 alice 到 prime 组

## 参见

[usermod(8)](usermod/)、[groupmod(8)](groupmod/)、[passwd(8)](passwd/)
