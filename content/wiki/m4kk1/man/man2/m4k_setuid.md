+++
title = "m4k_setuid"
date = '2026-08-08T14:03:07+08:00'
description = "设置用户 ID"
weight = 40
comments = false
+++

## 名称

m4k_setuid — 设置用户 ID

## 概要

```c
#include <m4k/syscall.h>
int m4k_setuid(uid_t uid);
```

## 描述

设置调用进程的实际 UID 和有效 UID。

## 返回值

成功返回 0，失败返回负的 `M4K_E*` 错误码。

## 错误

- `M4K_EPERM` — 调用者无权限（非 root 且不是目标 UID）

## 参见

[m4k_getuid(2)](../man2/m4k_getuid/)、[m4k_setgid(2)](m4k_setgid/)
