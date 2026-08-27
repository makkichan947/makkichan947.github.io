+++
title = "分区和文件系统"
date = "2025-10-28"
description = "LFS系统的磁盘分区和文件系统创建"
weight = 2
+++

# 分区和文件系统

LFS系统需要独立的磁盘分区来确保构建过程的隔离性和安全性。本章将详细介绍如何为LFS创建分区、格式化文件系统以及挂载分区。

## 💾 分区规划

### 分区方案

LFS推荐使用以下分区方案：

| 分区 | 挂载点 | 文件系统 | 大小 | 用途 |
|------|--------|----------|------|------|
| `/dev/sda1` | `/boot` | ext2/ext4 | 100MB | 引导分区 |
| `/dev/sda2` | `/` | ext4 | 10GB+ | 根分区 |
| `/dev/sda3` | `/home` | ext4 | 5GB+ | 用户主目录 |
| `/dev/sda4` | swap | swap | RAM*2 | 交换分区 |

### 分区大小建议

- **根分区**：至少8GB，推荐10GB以上
- **引导分区**：100-200MB（如果需要单独引导分区）
- **交换分区**：等于或大于物理内存大小
- **用户分区**：根据需要，5GB以上

## 🛠️ 创建分区

### 使用fdisk创建分区
```bash
# 查看当前磁盘
sudo fdisk -l

# 启动fdisk进行分区
sudo fdisk /dev/sda

# fdisk命令序列：
# n (新建分区)
# p (主分区)
# 1 (分区号)
# 默认 (起始扇区)
# +100M (分区大小)
# n, p, 2, 默认, +10G
# n, p, 3, 默认, +5G
# n, p, 4, 默认, 默认 (剩余空间)
# t, 4, 82 (设置交换分区类型)
# w (写入并退出)
```

### 使用parted创建分区（推荐）
```bash
# 使用parted创建分区
sudo parted /dev/sda

# parted命令：
# mklabel gpt  # 创建GPT分区表
# mkpart primary ext4 1MiB 101MiB  # /boot分区
# mkpart primary ext4 101MiB 10.1GiB  # 根分区
# mkpart primary ext4 10.1GiB 15.1GiB  # /home分区
# mkpart primary linux-swap 15.1GiB 100%  # 交换分区
# quit
```

### 验证分区
```bash
# 查看分区表
sudo fdisk -l /dev/sda

# 或者使用parted
sudo parted /dev/sda print
```

## 📁 格式化文件系统

### 格式化ext4文件系统
```bash
# 格式化根分区
sudo mkfs.ext4 /dev/sda2

# 格式化引导分区
sudo mkfs.ext4 /dev/sda1

# 格式化用户分区
sudo mkfs.ext4 /dev/sda3

# 格式化交换分区
sudo mkswap /dev/sda4
```

### 设置文件系统标签
```bash
# 设置分区标签（可选但推荐）
sudo e2label /dev/sda1 LFS_BOOT
sudo e2label /dev/sda2 LFS_ROOT
sudo e2label /dev/sda3 LFS_HOME
sudo swaplabel /dev/sda4 LFS_SWAP
```

### 检查文件系统
```bash
# 检查ext4文件系统
sudo tune2fs -l /dev/sda2

# 检查交换分区
sudo blkid /dev/sda4
```

## 🔗 挂载分区

### 创建挂载点
```bash
# 创建LFS根目录
sudo mkdir -pv $LFS

# 创建其他挂载点
sudo mkdir -pv $LFS/boot
sudo mkdir -pv $LFS/home
```

### 挂载分区
```bash
# 挂载根分区
sudo mount -v -t ext4 /dev/sda2 $LFS

# 挂载引导分区
sudo mount -v -t ext4 /dev/sda1 $LFS/boot

# 挂载用户分区
sudo mount -v -t ext4 /dev/sda3 $LFS/home

# 启用交换分区
sudo swapon /dev/sda4
```

### 验证挂载
```bash
# 检查挂载状态
mount | grep $LFS
df -h $LFS

# 检查交换空间
swapon -s
free -h
```

## 📋 创建fstab文件

### 生成fstab条目
```bash
# 获取UUID
blkid /dev/sda1  # 引导分区UUID
blkid /dev/sda2  # 根分区UUID
blkid /dev/sda3  # 用户分区UUID
blkid /dev/sda4  # 交换分区UUID
```

### 创建fstab文件
```bash
# 创建fstab文件
cat > $LFS/etc/fstab << "EOF"
# Begin /etc/fstab

# file system  mount-point  type     options             dump  fsck
#                                                              order

UUID=XXXX-XXXX-XXXX-XXXX   /            ext4    defaults            1     1
UUID=YYYY-YYYY-YYYY-YYYY   /boot        ext4    defaults            1     2
UUID=ZZZZ-ZZZZ-ZZZZ-ZZZZ   /home        ext4    defaults            1     2
UUID=WWWW-WWWW-WWWW-WWWW   swap         swap    pri=1               0     0
proc                                       /proc        proc     nosuid,noexec,nodev 0     0
sysfs                                      /sys         sysfs    nosuid,noexec,nodev 0     0
devpts                                     /dev/pts     devpts   gid=5,mode=620      0     0
tmpfs                                      /run         tmpfs    defaults            0     0
devtmpfs                                   /dev         devtmpfs  mode=0755,nosuid    0     0

# End /etc/fstab
EOF
```

### 更新fstab中的UUID
```bash
# 使用实际的UUID替换占位符
# 例如：
# UUID=550e8400-e29b-41d4-a716-446655440000   /            ext4    defaults            1     1
```

## 💾 高级分区方案

### LVM逻辑卷管理
```bash
# 创建物理卷
sudo pvcreate /dev/sda2

# 创建卷组
sudo vgcreate lfs_vg /dev/sda2

# 创建逻辑卷
sudo lvcreate -L 8G -n lfs_root lfs_vg
sudo lvcreate -L 2G -n lfs_home lfs_vg
sudo lvcreate -L 1G -n lfs_swap lfs_vg

# 格式化和挂载逻辑卷
sudo mkfs.ext4 /dev/lfs_vg/lfs_root
sudo mkfs.ext4 /dev/lfs_vg/lfs_home
sudo mkswap /dev/lfs_vg/lfs_swap

sudo mount /dev/lfs_vg/lfs_root $LFS
sudo mkdir $LFS/home
sudo mount /dev/lfs_vg/lfs_home $LFS/home
sudo swapon /dev/lfs_vg/lfs_swap
```

### Btrfs文件系统
```bash
# 创建Btrfs文件系统
sudo mkfs.btrfs /dev/sda2

# 挂载Btrfs
sudo mount -t btrfs /dev/sda2 $LFS

# 创建子卷
sudo btrfs subvolume create $LFS/@
sudo btrfs subvolume create $LFS/@home
sudo btrfs subvolume create $LFS/@snapshots

# 重新挂载子卷
sudo umount $LFS
sudo mount -t btrfs -o subvol=@ /dev/sda2 $LFS
sudo mkdir -p $LFS/home
sudo mount -t btrfs -o subvol=@home /dev/sda2 $LFS/home
```

## 🔄 备份和恢复

### 备份分区表
```bash
# 备份MBR分区表
sudo dd if=/dev/sda of=$LFS/backup/mbr_backup.img bs=512 count=1

# 备份GPT分区表
sudo sgdisk --backup=$LFS/backup/gpt_backup.bak /dev/sda
```

### 分区表恢复
```bash
# 恢复MBR分区表
sudo dd if=$LFS/backup/mbr_backup.img of=/dev/sda bs=512 count=1

# 恢复GPT分区表
sudo sgdisk --load-backup=$LFS/backup/gpt_backup.bak /dev/sda
```

## 🧪 测试和验证

### 文件系统完整性检查
```bash
# 检查ext4文件系统
sudo e2fsck -f /dev/sda2

# 检查Btrfs文件系统
sudo btrfs check /dev/sda2
```

### 性能测试
```bash
# 磁盘I/O性能测试
sudo hdparm -tT /dev/sda

# 文件系统性能测试
dd if=/dev/zero of=$LFS/test_file bs=1M count=100
rm $LFS/test_file
```

### 空间使用情况
```bash
# 查看分区使用情况
df -h $LFS

# 查看inode使用情况
df -i $LFS

# 查看大文件
du -sh $LFS/*
```

## 🚨 常见问题

### 分区无法挂载
```bash
# 检查分区是否存在
lsblk

# 检查文件系统类型
sudo blkid /dev/sda2

# 强制检查文件系统
sudo fsck -y /dev/sda2
```

### 空间不足
```bash
# 扩展分区（使用parted）
sudo parted /dev/sda
# resizepart 2 20G  # 扩展第二个分区到20G

# 扩展文件系统
sudo resize2fs /dev/sda2
```

### 引导问题
```bash
# 检查引导分区
ls $LFS/boot

# 验证GRUB配置
cat $LFS/boot/grub/grub.cfg
```

## 📚 相关资源

- [LFS官方文档 - 分区](http://www.linuxfromscratch.org/lfs/view/stable/chapter02/creatingpartition.html)
- [LFS官方文档 - 挂载](http://www.linuxfromscratch.org/lfs/view/stable/chapter02/mounting.html)
- [Arch Wiki - 分区](https://wiki.archlinux.org/title/Partitioning)
- [Linux文件系统层次标准](https://refspecs.linuxfoundation.org/FHS_3.0/fhs/index.html)

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*