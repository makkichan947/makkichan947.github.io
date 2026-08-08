+++
title = "login.conf"
date = '2026-08-08T14:03:09+08:00'
description = "M4KK1 登录类别资源限制配置"
weight = 20
comments = false
+++

## 名称

login.conf — M4KK1 登录类别资源限制配置

## 描述

`/export/login.conf` 定义 BSD 风格的登录类别和资源限制。

## 文件格式

```text
<类名>:\
    :<键>=<值>:\
    :<键>=<值>:
```

## 资源键

- **cputime** — CPU 时间（秒）
- **datasize** — 数据段 + 堆大小
- **stacksize** — 栈大小
- **maxproc** — 最大进程数
- **openfiles** — 最大打开文件数

## 参见

[passwd.db(5)](passwd.db/)、[groups.db(5)](groups.db/)
