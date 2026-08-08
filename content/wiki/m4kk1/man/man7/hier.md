+++
title = "hier"
date = '2026-08-08T14:03:10+08:00'
description = "M4KK1 文件系统层次结构"
weight = 10
comments = false
+++

## 名称

hier — M4KK1 文件系统层次结构

## 描述

M4KK1 标准文件系统层次:

- `/` — 根文件系统
- `/bin/` — 可执行文件 (login, m4sh)
- `/dev/` — 设备文件 (console, null, zero)
- `/export/cfg/` — 配置文件 (passwd.db, groups.db)
- `/export/home/` — 用户家目录
- `/sys/proc/` — 进程信息伪文件系统
- `/sys/sessions/` — 登录会话
- `/tmp/` — 临时文件

## 参见

[m4k_std(7)](m4k_std/)
