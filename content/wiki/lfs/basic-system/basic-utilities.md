+++
title = "基本命令行工具"
date = "2025-10-28"
description = "安装LFS基本命令行工具"
weight = 4
+++

# 基本命令行工具

基本命令行工具是Linux系统日常使用的核心工具集，包括文本处理、文件操作、系统监控等功能。本章将详细介绍这些工具的编译和安装。

## 🎯 工具概述

### 工具分类

基本命令行工具主要包括：

- **文本处理工具**：grep, sed, awk, diffutils
- **压缩工具**：gzip, bzip2, xz
- **文件工具**：findutils, file
- **网络工具**：wget, curl
- **其他工具**：less, tar, patch

## 🔍 Grep工具

### 编译Grep
```bash
# 切换到lfs用户
su - lfs

# 创建构建目录
mkdir -pv $LFS/sources/basic_utils
cd $LFS/sources/basic_utils

# 解压Grep源码
tar -xf $LFS/sources/grep-3.8.tar.xz
cd grep-3.8

# 配置Grep
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 📝 Sed工具

### 编译Sed
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Sed源码
tar -xf $LFS/sources/sed-4.9.tar.xz
cd sed-4.9

# 配置Sed
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install

# 创建符号链接
ln -s ../bin/sed /usr/sbin/sed
```

## 📊 Awk工具

### 编译Gawk
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Gawk源码
tar -xf $LFS/sources/gawk-5.2.1.tar.xz
cd gawk-5.2.1

# 配置Gawk
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install

# 创建符号链接
ln -s gawk /usr/bin/awk
```

## 🔄 Diffutils工具

### 编译Diffutils
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Diffutils源码
tar -xf $LFS/sources/diffutils-3.9.tar.xz
cd diffutils-3.9

# 配置Diffutils
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 🔍 Findutils工具

### 编译Findutils
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Findutils源码
tar -xf $LFS/sources/findutils-4.9.0.tar.xz
cd findutils-4.9.0

# 配置Findutils
./configure --prefix=/usr \
            --localstatedir=/var/lib/locate

# 编译
make $LFS_MAKEFLAGS

# 安装
make install

# 创建符号链接
ln -s ../bin/find /usr/sbin/find
```

## 📄 File工具

### 编译File
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压File源码
tar -xf $LFS/sources/file-5.44.tar.gz
cd file-5.44

# 配置File
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 📦 Tar工具

### 编译Tar
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Tar源码
tar -xf $LFS/sources/tar-1.34.tar.xz
cd tar-1.34

# 配置Tar
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 🗜️ 压缩工具

### 编译Gzip
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Gzip源码
tar -xf $LFS/sources/gzip-1.12.tar.xz
cd gzip-1.12

# 配置Gzip
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

### 编译Patch
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Patch源码
tar -xf $LFS/sources/patch-2.7.6.tar.xz
cd patch-2.7.6

# 配置Patch
./configure --prefix=/usr

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 🌐 网络工具

### 编译Wget
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Wget源码
tar -xf $LFS/sources/wget-1.21.3.tar.gz
cd wget-1.21.3

# 配置Wget
./configure --prefix=/usr \
            --sysconfdir=/etc \
            --with-ssl=openssl

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

### 编译Curl
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Curl源码
tar -xf $LFS/sources/curl-7.87.0.tar.xz
cd curl-7.87.0

# 配置Curl
./configure --prefix=/usr \
            --disable-static \
            --with-openssl \
            --enable-threaded-resolver \
            --with-ca-path=/etc/ssl/certs

# 编译
make $LFS_MAKEFLAGS

# 安装
make install

# 清理
rm -rf /usr/share/man/man3
```

## 📖 Less工具

### 编译Less
```bash
# 返回源码目录
cd $LFS/sources/basic_utils

# 解压Less源码
tar -xf $LFS/sources/less-608.tar.gz
cd less-608

# 配置Less
./configure --prefix=/usr --sysconfdir=/etc

# 编译
make $LFS_MAKEFLAGS

# 安装
make install
```

## 📋 构建脚本

### 自动化构建脚本
```bash
# 创建基本工具构建脚本
cat > $LFS/build_basic_utils.sh << 'EOF'
#!/bin/bash
# LFS基本工具构建脚本

set -e

# 工具列表
utils=(
    "grep-3.8:grep:--prefix=/usr"
    "sed-4.9:sed:--prefix=/usr"
    "gawk-5.2.1:gawk:--prefix=/usr"
    "diffutils-3.9:diffutils:--prefix=/usr"
    "findutils-4.9.0:findutils:--prefix=/usr --localstatedir=/var/lib/locate"
    "file-5.44:file:--prefix=/usr"
    "tar-1.34:tar:--prefix=/usr"
    "gzip-1.12:gzip:--prefix=/usr"
    "patch-2.7.6:patch:--prefix=/usr"
    "wget-1.21.3:wget:--prefix=/usr --sysconfdir=/etc --with-ssl=openssl"
    "curl-7.87.0:curl:--prefix=/usr --disable-static --with-openssl --enable-threaded-resolver --with-ca-path=/etc/ssl/certs"
    "less-608:less:--prefix=/usr --sysconfdir=/etc"
)

total_utils=${#utils[@]}
completed=0

for util_info in "${utils[@]}"; do
    IFS=':' read -r package_name util_name configure_options <<< "$util_info"

    echo "=== 构建 $util_name ($((completed + 1))/$total_utils) ==="

    # 检查源码
    if [ ! -f "$LFS/sources/$package_name.tar.xz" ] && [ ! -f "$LFS/sources/$package_name.tar.gz" ]; then
        echo "错误: $package_name 源码不存在"
        exit 1
    fi

    cd $LFS/sources/basic_utils

    # 解压源码
    if [ -f "$LFS/sources/$package_name.tar.xz" ]; then
        tar -xf "$LFS/sources/$package_name.tar.xz"
    else
        tar -xf "$LFS/sources/$package_name.tar.gz"
    fi

    cd $package_name

    # 配置和构建
    ./configure $configure_options
    make $LFS_MAKEFLAGS
    make install

    # 特殊处理
    case $util_name in
        sed)
            ln -s ../bin/sed /usr/sbin/sed
            ;;
        gawk)
            ln -s gawk /usr/bin/awk
            ;;
        findutils)
            ln -s ../bin/find /usr/sbin/find
            ;;
        curl)
            rm -rf /usr/share/man/man3
            ;;
    esac

    # 验证安装
    echo "验证 $util_name 安装..."
    case $util_name in
        grep)
            [ -x /usr/bin/grep ] && echo "✓ grep 安装成功" || echo "✗ grep 安装失败"
            ;;
        sed)
            [ -x /usr/bin/sed ] && echo "✓ sed 安装成功" || echo "✗ sed 安装失败"
            ;;
        gawk)
            [ -x /usr/bin/gawk ] && echo "✓ gawk 安装成功" || echo "✗ gawk 安装失败"
            ;;
        diffutils)
            [ -x /usr/bin/diff ] && echo "✓ diffutils 安装成功" || echo "✗ diffutils 安装失败"
            ;;
        findutils)
            [ -x /usr/bin/find ] && echo "✓ findutils 安装成功" || echo "✗ findutils 安装失败"
            ;;
        file)
            [ -x /usr/bin/file ] && echo "✓ file 安装成功" || echo "✗ file 安装失败"
            ;;
        tar)
            [ -x /usr/bin/tar ] && echo "✓ tar 安装成功" || echo "✗ tar 安装失败"
            ;;
        gzip)
            [ -x /usr/bin/gzip ] && echo "✓ gzip 安装成功" || echo "✗ gzip 安装失败"
            ;;
        patch)
            [ -x /usr/bin/patch ] && echo "✓ patch 安装成功" || echo "✗ patch 安装失败"
            ;;
        wget)
            [ -x /usr/bin/wget ] && echo "✓ wget 安装成功" || echo "✗ wget 安装失败"
            ;;
        curl)
            [ -x /usr/bin/curl ] && echo "✓ curl 安装成功" || echo "✗ curl 安装失败"
            ;;
        less)
            [ -x /usr/bin/less ] && echo "✓ less 安装成功" || echo "✗ less 安装失败"
            ;;
    esac

    completed=$((completed + 1))
    echo "进度: $completed/$total_utils 完成"
    echo ""

    # 清理构建目录
    cd $LFS/sources/basic_utils
    rm -rf $package_name
done

echo "=== 所有基本工具构建完成 ==="
EOF

chmod +x $LFS/build_basic_utils.sh
```

## 🧪 功能验证

### 工具可用性测试
```bash
# 创建验证脚本
cat > $LFS/verify_basic_utils.sh << 'EOF'
#!/bin/bash
# 基本工具验证脚本

echo "=== LFS基本工具验证 ==="

# 定义要验证的工具
tools=(
    "/usr/bin/grep:grep"
    "/usr/bin/sed:sed"
    "/usr/bin/gawk:gawk"
    "/usr/bin/diff:diffutils"
    "/usr/bin/find:findutils"
    "/usr/bin/file:file"
    "/usr/bin/tar:tar"
    "/usr/bin/gzip:gzip"
    "/usr/bin/patch:patch"
    "/usr/bin/wget:wget"
    "/usr/bin/curl:curl"
    "/usr/bin/less:less"
)

passed=0
total=${#tools[@]}

for tool_info in "${tools[@]}"; do
    IFS=':' read -r tool_path tool_name <<< "$tool_info"

    echo -n "检查 $tool_name ($tool_path)... "

    if [ -x "$tool_path" ]; then
        echo "✓ 可用"

        # 基本功能测试
        case $tool_name in
            grep)
                echo "test" | grep "test" >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
            sed)
                echo "test" | sed 's/test/replace/' >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
            gawk)
                echo "1 2 3" | awk '{print $1}' >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
            diff)
                echo "test1" > file1.txt && echo "test2" > file2.txt
                diff file1.txt file2.txt >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                rm -f file1.txt file2.txt
                ;;
            find)
                find /usr -maxdepth 1 -name bin >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
            file)
                file /bin/sh >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
            tar)
                echo "test" > test.txt && tar -cf test.tar test.txt >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                rm -f test.txt test.tar
                ;;
            gzip)
                echo "test" > test.txt && gzip test.txt && gunzip test.txt.gz >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                rm -f test.txt test.txt.gz
                ;;
            patch)
                echo "基本功能检查跳过"  # patch需要特殊测试文件
                ;;
            wget)
                wget --version >/dev/null 2>&1 && echo "  版本检查: ✓" || echo "  版本检查: ✗"
                ;;
            curl)
                curl --version >/dev/null 2>&1 && echo "  版本检查: ✓" || echo "  版本检查: ✗"
                ;;
            less)
                echo "test" | less -F >/dev/null 2>&1 && echo "  基本功能: ✓" || echo "  基本功能: ✗"
                ;;
        esac

        passed=$((passed + 1))
    else
        echo "✗ 未找到"
    fi
done

echo ""
echo "=== 验证结果 ==="
echo "通过: $passed/$total"

if [ $passed -eq $total ]; then
    echo "✓ 所有基本工具都已正确安装"
    exit 0
else
    echo "✗ 部分工具安装失败"
    exit 1
fi
EOF

chmod +x $LFS/verify_basic_utils.sh
```

## 🚨 常见问题

### 编译失败
```bash
# 检查常见问题：
# 1. 依赖库
echo "检查依赖..."

for tool in grep sed gawk; do
    if [ -x "/usr/bin/$tool" ]; then
        ldd "/usr/bin/$tool" 2>/dev/null || echo "$tool 静态链接"
    fi
done

# 2. 环境变量
echo $PATH
echo $LFS_TGT

# 3. 源码完整性
ls -la $LFS/sources/grep-* $LFS/sources/sed-*
```

### 功能异常
```bash
# 测试工具功能
echo "测试grep..."
echo "hello world" | grep "world"

echo -e "\n测试sed..."
echo "hello world" | sed 's/world/universe/'

echo -e "\n测试awk..."
echo "1 2 3" | awk '{print $2}'

echo -e "\n测试find..."
find /usr/bin -name "grep" 2>/dev/null
```

### 网络工具问题
```bash
# 测试网络工具
echo "测试wget..."
wget --version

echo -e "\n测试curl..."
curl --version

# 测试网络连接
echo -e "\n测试网络连接..."
ping -c 1 8.8.8.8 >/dev/null 2>&1 && echo "网络连接正常" || echo "网络连接异常"
```

## 📊 工具统计

### 工具大小统计
```bash
# 统计工具大小
echo "=== 基本工具大小统计 ==="
echo "工具 | 大小"
echo "----|-----"

for tool in grep sed gawk diff find file tar gzip patch wget curl less; do
    if [ -x "/usr/bin/$tool" ]; then
        size=$(ls -lh "/usr/bin/$tool" | awk '{print $5}')
        printf "%-8s | %s\n" "$tool" "$size"
    fi
done
```

## 📚 相关资源

- [LFS官方文档 - 基本工具](http://www.linuxfromscratch.org/lfs/view/stable/chapter06/chapter06.html)
- [GNU工具文档](https://www.gnu.org/software/)
- [网络工具文档](https://curl.se/docs/)

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*