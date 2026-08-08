+++
title = "m4k_register_session"
date = '2026-08-08T14:03:07+08:00'
description = "注册登录会话"
weight = 20
comments = false
+++

## 名称

m4k_register_session — 注册登录会话

## 概要

```c
#include <m4k/syscall.h>
int m4k_register_session(const char *tty, uid_t uid, const char *username);
```

## 描述

在 `/sys/sessions/` 中注册一个新的登录会话，用于 `who` 和 `userlog` 命令。

## 返回值

成功返回 0，失败返回负的 `M4K_E*` 错误码。

## 参见

[m4k_get_session_list(2)](m4k_get_session_list/)、[who(1)](../man1/who/)
