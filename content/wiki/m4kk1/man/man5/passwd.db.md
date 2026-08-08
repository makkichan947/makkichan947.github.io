+++
title = "passwd.db"
date = '2026-08-08T14:03:10+08:00'
description = "M4KK1 用户数据库"
weight = 30
comments = false
+++

## 名称

passwd.db — M4KK1 用户数据库

## 描述

`/etc/passwd.db`（实际位于 `/export/cfg/passwd.db`）包含每个用户的账户信息。每一行包含由冒号分隔的 7 个字段:

- **username** — 用户名
- **uid** — 用户 ID
- **gid** — 主组 ID
- **home** — 家目录路径
- **shell** — 登录 Shell
- **gecos** — 用户全名/描述
- **password_hash** — 密码哈希（`$SHA$salt$hash` 格式）

## 参见

[groups.db(5)](groups.db/)、[login.conf(5)](login.conf/)
