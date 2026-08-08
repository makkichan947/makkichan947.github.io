+++
title = "musr_getgrnam"
date = '2026-08-08T14:03:08+08:00'
description = "按组名查找组数据库条目"
weight = 10
comments = false
+++

## 名称

musr_getgrnam — 按组名查找组数据库条目

## 概要

```c
#include <m4k/grp.h>
int musr_getgrnam(const char *name, group_entry_t *out);
```

## 描述

在 `/export/cfg/groups.db` 中查找匹配组名的条目，结果写入 `out`。

## 返回值

成功返回 0，未找到返回负的 `M4K_ENOENT`。

## 参见

[musr_getpwnam(3)](musr_getpwnam/)、[musr_getpwuid(3)](musr_getpwuid/)
