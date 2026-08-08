+++
title = "m4k_setgid"
date = '2026-08-08T14:03:07+08:00'
description = "设置组 ID"
weight = 30
comments = false
+++

## 名称

m4k_setgid — 设置组 ID

## 概要

```c
#include <m4k/syscall.h>
int m4k_setgid(gid_t gid);
```

## 描述

设置调用进程的实际 GID 和有效 GID。

## 返回值

成功返回 0，失败返回负的 `M4K_E*` 错误码。

## 错误

- `M4K_EPERM` — 调用者无权限

## 参见

[m4k_setuid(2)](m4k_setuid/)、[m4k_getgid(2)](../man2/m4k_getgid/)
