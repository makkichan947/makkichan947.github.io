+++
title = "M4KK1 操作系统"
date = "2025-10-16"
description = "基于 4P1 独立标准的 i386 操作系统"
weight = 2
comments = false
toc = true
+++

M4KK1 是一个从零开发的 **i386 操作系统**，采用 **4P1 独立标准**，不兼容 POSIX。项目使用 freestanding C 和 NASM 汇编编写，遵循宏内核设计。

## 快速开始

```bash
git clone https://github.com/makkichan947/M4KK1.git
cd M4KK1
bash tools/build/build_krn.sh
qemu-system-x86_64 -cdrom m4kk1-test.iso -serial stdio
```

## 关键概念

| 概念 | 说明 |
|------|------|
| 内核 | **Y4KU** 宏内核，i386 保护模式，GRUB multiboot 引导 |
| 用户空间 | **M4SH** 单进程 Shell（PID 1），40+ 内置命令，静态链接 |
| 文件系统 | **YAFS** — COW B+Tree RAM 磁盘（16 MB） |
| 标准 | **4P1** — 不兼容 POSIX，无 `errno`，无 MMU |
| 文档 | BSD 风格手册页 + i18n（zh_CN / en_US） |

## 文档导航

- **[系统手册](./handbook/)** — 完整使用手册（9 章）
- **[专题文章](./articles/)** — 设计细节深入解析
- **[手册页](./man/)** — BSD 格式命令参考（man1–man8）

## 项目状态

当前为 **v1.0.0-alpha1**，处于早期实验阶段。已完成内核基础、YAFS 文件系统、M4SH Shell 和 40+ 内置命令。

### 已知限制

- i386 32-bit 仅，无 SMP/APIC
- 单进程用户空间，全部 Ring 0
- 无 MMU 隔离、无网络栈、无图形界面
- YAFS 无日志/崩溃恢复
- 无 POSIX 兼容层

## 参与贡献

欢迎提交 Pull Request 或 Issue：<https://github.com/makkichan947/M4KK1>
