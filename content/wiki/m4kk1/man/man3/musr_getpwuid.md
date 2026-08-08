+++
title = "musr_getpwuid"
date = '2026-08-08T14:03:08+08:00'
description = "按 UID 查找用户数据库条目"
weight = 30
comments = false
+++

## 名称

musr_getpwuid — 按 UID 查找用户数据库条目

## 概要

```c
#include <m4k/pwd.h>
int musr_getpwuid(uid_t uid, passwd_entry_t *out);
```

## 描述

在 `/export/cfg/passwd.db` 中查找匹配 UID 的条目，结果写入 `out`。

## 返回值

成功返回 0，未找到返回负的 `M4K_ENOENT`。

## 参见

[musr_getpwnam(3)](musr_getpwnam/)、[musr_getgrnam(3)](musr_getgrnam/)
