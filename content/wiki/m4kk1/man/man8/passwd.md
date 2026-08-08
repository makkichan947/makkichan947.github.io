+++
title = "passwd"
date = '2026-08-08T14:03:11+08:00'
description = "修改用户密码"
weight = 30
comments = false
+++

## 名称

passwd — 修改用户密码

## 概要

```text
passwd [用户名]
```

## 描述

修改用户密码。无参数时修改当前用户密码。指定用户名时需要 prime 组成员权限。密码使用 SHA-256 + salt 哈希。

## 参见

[usermod(8)](usermod/)、[quell(8)](quell/)、[login.conf(5)](../man5/login.conf/)
