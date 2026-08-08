+++
title = "m4k_get_session_list"
date = '2026-08-08T14:03:07+08:00'
description = "获取会话列表"
weight = 10
comments = false
+++

## 名称

m4k_get_session_list — 获取会话列表

## 概要

```c
#include <m4k/syscall.h>
int m4k_get_session_list(void *buf, int max);
```

## 描述

获取当前所有活跃登录会话的列表。每个会话包含 TTY、UID、登录时间和用户名。

## 返回值

成功返回会话数量，失败返回负的 `M4K_E*` 错误码。

## 参见

[m4k_register_session(2)](m4k_register_session/)、[who(1)](../man1/who/)
