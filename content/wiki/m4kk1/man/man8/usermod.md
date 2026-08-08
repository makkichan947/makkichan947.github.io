+++
title = "usermod"
date = '2026-08-08T14:03:12+08:00'
description = "修改用户账户"
weight = 50
comments = false
+++

## 名称

usermod — 修改用户账户

## 概要

```text
usermod [选项] 用户名
```

## 描述

修改 `/export/cfg/passwd.db` 中的用户账户信息。需要 prime 组成员权限。

## 选项

- `-a, --add 组` — 添加用户到附加组
- `-d, --del 组` — 从组中移除用户
- `-s, --shell 路径` — 更改登录 Shell
- `-g, --group 组` — 设置主组
- `-L, --lock` — 锁定账户
- `-U, --unlock` — 解锁账户

## 参见

[groupmod(8)](groupmod/)、[passwd(8)](passwd/)、[quell(8)](quell/)
