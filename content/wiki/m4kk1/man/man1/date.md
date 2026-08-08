+++
title = "date"
date = '2026-08-08T14:03:04+08:00'
description = "显示或设置系统日期和时间"
weight = 20
comments = false
+++

## 名称

date — 显示或设置系统日期和时间

## 概要

```text
date [选项]
```

## 描述

显示当前系统日期和时间，或设置系统时间。

## 选项

- `-f, --format FMT` — strftime 格式输出
- `-u, --utc` — 以 UTC 显示时间
- `-d, --date DT` — 解析并显示日期
- `-s, --set DT` — 设置系统时间
- `--uptime` — 显示系统运行时间
- `--hwclock` — 读取 RTC 时间
- `--timers` — 显示活跃内核定时器数

## 参见

[cal(1)](../man1/cal/)、[uname(1)](../man1/uname/)
