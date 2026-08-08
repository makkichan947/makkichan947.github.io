+++
title = "groups.db"
date = '2026-08-08T14:03:09+08:00'
description = "M4KK1 组数据库"
weight = 10
comments = false
+++

## 名称

groups.db — M4KK1 组数据库

## 描述

`/etc/groups.db`（实际位于 `/export/cfg/groups.db`）包含系统组信息。每一行包含由冒号分隔的字段:

- **groupname** — 组名
- **gid** — 组 ID
- **members** — 逗号分隔的成员列表

## 示例

```text
prime:1001:root,testuser
```

## 参见

[passwd.db(5)](passwd.db/)、[login.conf(5)](login.conf/)
