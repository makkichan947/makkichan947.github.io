+++
title = "目录结构规划"
date = "2025-10-28"
description = "LFS系统的目录结构设计和规划"
weight = 4
+++

# 目录结构规划

LFS系统的目录结构遵循Linux文件系统层次标准（FHS）。正确的目录结构规划对于系统的稳定运行和维护至关重要。本章将详细介绍LFS目录结构的创建和配置。

## 📁 Linux文件系统层次标准 (FHS)

### FHS概述

FHS定义了Linux系统中文件和目录的标准组织方式：

- **/**：根目录，所有其他目录的起点
- **/bin**：基本命令二进制文件
- **/boot**：引导加载程序文件
- **/dev**：设备文件
- **/etc**：系统配置文件
- **/home**：用户主目录
- **/lib**：基本共享库
- **/media**：可移动媒体挂载点
- **/mnt**：临时挂载点
- **/opt**：附加应用程序软件包
- **/proc**：虚拟文件系统（进程信息）
- **/root**：root用户主目录
- **/sbin**：系统二进制文件
- **/srv**：系统提供的服务数据
- **/sys**：虚拟文件系统（设备信息）
- **/tmp**：临时文件
- **/usr**：二级层次结构
- **/var**：可变数据

## 🏗️ LFS目录结构创建

### 创建基础目录结构
```bash
# 切换到LFS用户
su - lfs

# 创建基础目录
mkdir -pv $LFS/{etc,var,usr/{bin,lib,sbin},tools}

# 创建64位系统所需的lib64目录
case $(uname -m) in
  x86_64) mkdir -pv $LFS/lib64 ;;
esac

# 创建其他标准目录
mkdir -pv $LFS/{bin,boot,dev,home,media,mnt,opt,proc,root,srv,sys,tmp,var/{cache,lib,local,lock,log,opt,run,spool}}

# 创建usr下的子目录
mkdir -pv $LFS/usr/{include,lib,share/{color,dict,doc,info,locale,man,misc,terminfo,zoneinfo}}
mkdir -pv $LFS/usr/{libexec,local/{bin,etc,include,lib,sbin,share,var},sbin,src}

# 创建var下的子目录
mkdir -pv $LFS/var/{lib/{color,misc,locate},cache/{local,man}}
```

### 设置目录权限
```bash
# 设置正确的目录权限
chmod -v 0750 $LFS/root
chmod -v 1777 $LFS/{var,}/tmp
chmod -v 0750 $LFS/home

# 创建必要的符号链接
ln -sv usr/bin $LFS/bin
ln -sv usr/lib $LFS/lib
ln -sv usr/sbin $LFS/sbin

# 64位系统创建lib64链接
case $(uname -m) in
  x86_64) ln -sv usr/lib64 $LFS/lib64 ;;
esac
```

### 验证目录结构
```bash
# 检查目录结构
ls -la $LFS

# 验证符号链接
ls -l $LFS/bin $LFS/lib $LFS/sbin

# 检查目录权限
ls -ld $LFS/{root,tmp,home}
```

## 📋 详细目录说明

### 根目录 (/)
```bash
# 根目录包含：
# - bin -> usr/bin (基本命令)
# - boot/ (引导文件)
# - dev/ (设备文件)
# - etc/ (配置文件)
# - home/ (用户目录)
# - lib -> usr/lib (基本库)
# - lib64 -> usr/lib64 (64位库)
# - media/ (可移动媒体)
# - mnt/ (临时挂载)
# - opt/ (可选软件)
# - proc/ (进程信息)
# - root/ (root用户目录)
# - run/ (运行时数据)
# - sbin -> usr/sbin (系统命令)
# - srv/ (服务数据)
# - sys/ (系统信息)
# - tmp/ (临时文件)
# - usr/ (用户程序)
# - var/ (可变数据)
```

### /usr 目录结构
```bash
# /usr 包含二级层次结构：
# - bin/ (用户命令)
# - include/ (头文件)
# - lib/ (库文件)
# - lib64/ (64位库文件)
# - libexec/ (可执行库)
# - local/ (本地软件)
# - sbin/ (系统管理命令)
# - share/ (架构无关数据)
#   - dict/ (词典)
#   - doc/ (文档)
#   - info/ (info文档)
#   - locale/ (本地化)
#   - man/ (手册页)
#   - misc/ (杂项)
#   - terminfo/ (终端信息)
#   - zoneinfo/ (时区信息)
# - src/ (源码)
```

### /var 目录结构
```bash
# /var 包含可变数据：
# - cache/ (缓存文件)
#   - local/ (本地缓存)
#   - man/ (手册页缓存)
# - lib/ (可变状态信息)
#   - color/ (颜色数据库)
#   - locate/ (locate数据库)
#   - misc/ (杂项)
# - local/ (本地软件的可变数据)
# - lock/ (锁文件)
# - log/ (日志文件)
# - opt/ (可选软件的可变数据)
# - run/ (运行时变量数据)
# - spool/ (应用程序假脱机文件)
# - tmp/ (临时文件，系统重启时保留)
```

## 🔧 特殊文件和设备

### 创建设备节点
```bash
# 创建基本的设备节点
sudo mknod -m 600 $LFS/dev/console c 5 1
sudo mknod -m 666 $LFS/dev/null c 1 3
sudo mknod -m 666 $LFS/dev/zero c 1 5
sudo mknod -m 666 $LFS/dev/ptmx c 5 2
sudo mknod -m 666 $LFS/dev/tty c 5 0
sudo mknod -m 444 $LFS/dev/random c 1 8
sudo mknod -m 444 $LFS/dev/urandom c 1 9

# 创建/dev/shm目录
mkdir -pv $LFS/dev/shm

# 创建/dev/pts目录
mkdir -pv $LFS/dev/pts
```

### 创建必要的符号链接
```bash
# 创建日志文件的符号链接
ln -sv /run $LFS/var/run
ln -sv /run/lock $LFS/var/lock

# 创建其他必要的链接
ln -sv /proc/self/mounts $LFS/etc/mtab
```

## 📝 配置文件创建

### 创建passwd文件
```bash
# 创建基本的/etc/passwd文件
cat > $LFS/etc/passwd << "EOF"
root:x:0:0:root:/root:/bin/bash
bin:x:1:1:bin:/dev/null:/bin/false
daemon:x:6:6:daemon:/dev/null:/bin/false
messagebus:x:18:18:D-Bus Message Daemon User:/var/run/dbus:/bin/false
systemd-bus-proxy:x:72:72:systemd Bus Proxy:/:/bin/false
systemd-journal-gateway:x:73:73:systemd Journal Gateway:/:/bin/false
systemd-journal-remote:x:74:74:systemd Journal Remote:/:/bin/false
systemd-journal-upload:x:75:75:systemd Journal Upload:/:/bin/false
systemd-network:x:76:76:systemd Network Management:/:/bin/false
systemd-resolve:x:77:77:systemd Resolver:/:/bin/false
systemd-timesync:x:78:78:systemd Time Synchronization:/:/bin/false
systemd-coredump:x:79:79:systemd Core Dumper:/:/bin/false
uuidd:x:80:80:UUID daemon:/dev/null:/bin/false
nobody:x:99:99:Unprivileged User:/dev/null:/bin/false
EOF
```

### 创建group文件
```bash
# 创建基本的/etc/group文件
cat > $LFS/etc/group << "EOF"
root:x:0:
bin:x:1:daemon
sys:x:2:
kmem:x:3:
tape:x:4:
tty:x:5:
daemon:x:6:
floppy:x:7:
disk:x:8:
lp:x:9:
dialout:x:10:
audio:x:11:
video:x:12:
utmp:x:13:
usb:x:14:
cdrom:x:15:
adm:x:16:
messagebus:x:18:
systemd-journal:x:23:
input:x:24:
mail:x:34:
kvm:x:61:
systemd-bus-proxy:x:72:
systemd-journal-gateway:x:73:
systemd-journal-remote:x:74:
systemd-journal-upload:x:75:
systemd-network:x:76:
systemd-resolve:x:77:
systemd-timesync:x:78:
systemd-coredump:x:79:
uuidd:x:80:
wheel:x:97:
nogroup:x:99:
users:x:999:
EOF
```

### 创建基本的配置文件
```bash
# 创建/etc/hostname
echo "lfs" > $LFS/etc/hostname

# 创建/etc/hosts
cat > $LFS/etc/hosts << "EOF"
127.0.0.1 localhost lfs
::1       localhost lfs
EOF

# 创建/etc/resolv.conf
cat > $LFS/etc/resolv.conf << "EOF"
nameserver 8.8.8.8
nameserver 8.8.4.4
EOF

# 创建/etc/inputrc
cat > $LFS/etc/inputrc << "EOF"
set horizontal-scroll-mode Off
set meta-flag On
set input-meta On
set convert-meta Off
set output-meta On
set bell-style Off
"\eOd": backward-word
"\eOc": forward-word
EOF
```

## 🛠️ 构建工具设置

### 创建构建日志目录
```bash
# 创建日志目录
mkdir -pv $LFS/logs

# 设置日志文件权限
touch $LFS/logs/build.log
chmod 644 $LFS/logs/build.log
```

### 创建构建脚本框架
```bash
# 创建构建工具函数库
cat > $LFS/lib/build_functions.sh << 'EOF'
#!/bin/bash
# LFS构建工具函数库

# 日志函数
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*" | tee -a $LFS/logs/build.log
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2 | tee -a $LFS/logs/build.log
}

log_warn() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $*" | tee -a $LFS/logs/build.log
}

# 包构建函数
build_package() {
    local package_name=$1
    local package_url=$2
    local configure_options=${3:-""}

    log_info "开始构建 $package_name"

    # 创建构建目录
    local build_dir="$LFS/sources/build_$package_name"
    mkdir -pv "$build_dir"
    cd "$build_dir"

    # 下载源码
    if [ ! -f "$LFS/sources/$package_name.tar.xz" ]; then
        log_info "下载 $package_name"
        wget -P "$LFS/sources" "$package_url" || {
            log_error "下载 $package_name 失败"
            return 1
        }
    fi

    # 解压源码
    log_info "解压 $package_name"
    tar -xf "$LFS/sources/$package_name.tar.xz" || {
        log_error "解压 $package_name 失败"
        return 1
    }

    # 进入源码目录
    cd "$package_name"*/

    # 配置
    log_info "配置 $package_name"
    ./configure --prefix=/usr $configure_options || {
        log_error "配置 $package_name 失败"
        return 1
    }

    # 编译
    log_info "编译 $package_name"
    make $LFS_MAKEFLAGS || {
        log_error "编译 $package_name 失败"
        return 1
    }

    # 安装
    log_info "安装 $package_name"
    make install || {
        log_error "安装 $package_name 失败"
        return 1
    }

    # 清理
    cd "$LFS/sources"
    rm -rf "$build_dir"

    log_info "$package_name 构建完成"
    return 0
}

# 错误处理
set -e
trap 'log_error "构建失败于第 $LINENO 行"' ERR
EOF

# 设置执行权限
chmod +x $LFS/lib/build_functions.sh
```

## 📊 目录结构验证

### 完整性检查脚本
```bash
# 创建目录验证脚本
cat > $LFS/verify_structure.sh << 'EOF'
#!/bin/bash
# LFS目录结构验证脚本

set -e

LFS=${LFS:-/mnt/lfs}
errors=0

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"
}

check_dir() {
    local dir=$1
    if [ ! -d "$LFS$dir" ]; then
        log "ERROR: 目录不存在: $dir"
        errors=$((errors + 1))
    else
        log "OK: 目录存在: $dir"
    fi
}

check_link() {
    local link=$1
    local target=$2
    if [ ! -L "$LFS$link" ]; then
        log "ERROR: 符号链接不存在: $link"
        errors=$((errors + 1))
    elif [ "$(readlink $LFS$link)" != "$target" ]; then
        log "ERROR: 符号链接目标错误: $link -> $(readlink $LFS$link) (期望: $target)"
        errors=$((errors + 1))
    else
        log "OK: 符号链接正确: $link -> $target"
    fi
}

check_file() {
    local file=$1
    if [ ! -f "$LFS$file" ]; then
        log "ERROR: 文件不存在: $file"
        errors=$((errors + 1))
    else
        log "OK: 文件存在: $file"
    fi
}

log "开始验证LFS目录结构..."

# 检查基本目录
log "检查基本目录..."
check_dir "/bin"
check_dir "/boot"
check_dir "/dev"
check_dir "/etc"
check_dir "/home"
check_dir "/lib"
check_dir "/media"
check_dir "/mnt"
check_dir "/opt"
check_dir "/proc"
check_dir "/root"
check_dir "/run"
check_dir "/sbin"
check_dir "/srv"
check_dir "/sys"
check_dir "/tmp"
check_dir "/usr"
check_dir "/var"

# 检查符号链接
log "检查符号链接..."
check_link "/bin" "usr/bin"
check_link "/lib" "usr/lib"
check_link "/sbin" "usr/sbin"

case $(uname -m) in
  x86_64) check_link "/lib64" "usr/lib64" ;;
esac

# 检查usr子目录
log "检查usr子目录..."
check_dir "/usr/bin"
check_dir "/usr/include"
check_dir "/usr/lib"
check_dir "/usr/sbin"
check_dir "/usr/share"
check_dir "/usr/src"

# 检查配置文件
log "检查配置文件..."
check_file "/etc/passwd"
check_file "/etc/group"
check_file "/etc/hostname"
check_file "/etc/hosts"

# 检查设备节点
log "检查设备节点..."
check_file "/dev/null"
check_file "/dev/zero"
check_file "/dev/console"

# 检查权限
log "检查目录权限..."
if [ "$(stat -c %a $LFS/root)" != "750" ]; then
    log "ERROR: /root 权限不正确"
    errors=$((errors + 1))
else
    log "OK: /root 权限正确"
fi

if [ "$(stat -c %a $LFS/tmp)" != "1777" ]; then
    log "ERROR: /tmp 权限不正确"
    errors=$((errors + 1))
else
    log "OK: /tmp 权限正确"
fi

# 总结
log "验证完成"
if [ $errors -eq 0 ]; then
    log "所有检查通过！LFS目录结构正确。"
    exit 0
else
    log "发现 $errors 个错误。请检查上述错误信息。"
    exit 1
fi
EOF

# 设置执行权限并运行验证
chmod +x $LFS/verify_structure.sh
$LFS/verify_structure.sh
```

## 🚨 常见问题

### 权限问题
```bash
# 如果遇到权限问题，确保lfs用户有正确的权限
sudo chown -R lfs:lfs $LFS

# 或者重新创建目录
sudo rm -rf $LFS
sudo mkdir -pv $LFS
sudo chown -v lfs:lfs $LFS
```

### 符号链接问题
```bash
# 检查并修复符号链接
ls -l $LFS/bin
# 如果链接不正确，删除并重新创建
rm $LFS/bin
ln -sv usr/bin $LFS/bin
```

### 配置文件问题
```bash
# 如果配置文件丢失，可以重新创建
# 参考上面的配置创建命令
```

## 📚 相关资源

- [LFS官方文档 - 目录结构](http://www.linuxfromscratch.org/lfs/view/stable/chapter06/chapter06.html)
- [Linux FHS标准](https://refspecs.linuxfoundation.org/FHS_3.0/fhs/index.html)
- [Filesystem Hierarchy Standard](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard)

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*