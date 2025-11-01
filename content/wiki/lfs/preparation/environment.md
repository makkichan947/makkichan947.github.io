+++
title = "环境变量设置"
date = "2025-10-28"
description = "LFS构建环境的环境变量和脚本配置"
weight = 5
+++

# 环境变量设置

正确的环境变量配置是LFS构建成功的关键。本章将详细介绍如何设置LFS构建所需的环境变量、构建脚本和开发环境。

## 🌍 基本环境变量

### LFS环境变量
```bash
# 设置LFS根目录
export LFS=/mnt/lfs

# 设置目标架构
export LFS_TGT=$(uname -m)-lfs-linux-gnu

# 添加工具链路径
export PATH=$LFS/tools/bin:$PATH

# 设置配置站点
export CONFIG_SITE=$LFS/usr/share/config.site

# 设置语言环境
export LC_ALL=POSIX

# 设置并行编译参数
export LFS_MAKEFLAGS=-j$(nproc)

# 验证设置
echo "LFS=$LFS"
echo "LFS_TGT=$LFS_TGT"
echo "PATH=$PATH"
echo "MAKEFLAGS=$LFS_MAKEFLAGS"
```

### 永久设置环境变量
```bash
# 在~/.bashrc中添加LFS环境变量
cat >> ~/.bashrc << "EOF"

# LFS环境变量设置
export LFS=/mnt/lfs
export LFS_TGT=$(uname -m)-lfs-linux-gnu
export PATH=$LFS/tools/bin:$PATH
export CONFIG_SITE=$LFS/usr/share/config.site
export LC_ALL=POSIX
export LFS_MAKEFLAGS=-j$(nproc)

EOF

# 重新加载bashrc
source ~/.bashrc
```

## 🛠️ 构建脚本框架

### 通用构建函数
```bash
# 创建构建函数库
cat > $LFS/lib/lfs_build.sh << 'EOF'
#!/bin/bash
# LFS通用构建函数库

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] INFO: $*${NC}" | tee -a $LFS/logs/build.log
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN: $*${NC}" | tee -a $LFS/logs/build.log
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $*${NC}" >&2 | tee -a $LFS/logs/build.log
}

# 环境检查函数
check_environment() {
    log_info "检查构建环境..."

    # 检查LFS变量
    if [ -z "$LFS" ]; then
        log_error "LFS环境变量未设置"
        exit 1
    fi

    # 检查LFS目录
    if [ ! -d "$LFS" ]; then
        log_error "LFS目录不存在: $LFS"
        exit 1
    fi

    # 检查工具链
    if [ ! -d "$LFS/tools" ]; then
        log_error "工具链目录不存在: $LFS/tools"
        exit 1
    fi

    # 检查必要工具
    local required_tools="bash sh gcc g++ make ld ar as nm strip ranlib"
    for tool in $required_tools; do
        if ! command -v $tool >/dev/null 2>&1; then
            log_error "缺少必要工具: $tool"
            exit 1
        fi
    done

    log_info "环境检查通过"
}

# 包构建函数
build_package() {
    local package_name=$1
    local package_version=$2
    local configure_options=${3:-""}

    log_info "开始构建 $package_name-$package_version"

    # 创建构建目录
    local build_dir="$LFS/sources/build_$package_name"
    mkdir -pv "$build_dir"
    cd "$build_dir"

    # 下载源码（如果不存在）
    local package_file="$LFS/sources/$package_name-$package_version.tar.xz"
    if [ ! -f "$package_file" ]; then
        log_warn "源码包不存在: $package_file"
        return 1
    fi

    # 解压源码
    log_info "解压源码..."
    if ! tar -xf "$package_file"; then
        log_error "解压失败: $package_file"
        return 1
    fi

    # 进入源码目录
    cd "$package_name-$package_version"

    # 配置阶段
    log_info "配置 $package_name..."
    if [ -n "$configure_options" ]; then
        log_info "配置选项: $configure_options"
    fi

    if ! ./configure --prefix=/usr $configure_options; then
        log_error "配置失败"
        return 1
    fi

    # 编译阶段
    log_info "编译 $package_name..."
    if ! make $LFS_MAKEFLAGS; then
        log_error "编译失败"
        return 1
    fi

    # 安装阶段
    log_info "安装 $package_name..."
    if ! make install; then
        log_error "安装失败"
        return 1
    fi

    # 清理
    cd "$LFS/sources"
    rm -rf "$build_dir"

    log_info "$package_name-$package_version 构建成功"
    return 0
}

# 交叉编译构建函数
build_cross_package() {
    local package_name=$1
    local package_version=$2
    local configure_options=${3:-""}

    log_info "交叉编译 $package_name-$package_version"

    # 创建构建目录
    local build_dir="$LFS/sources/cross_$package_name"
    mkdir -pv "$build_dir"
    cd "$build_dir"

    # 解压源码
    local package_file="$LFS/sources/$package_name-$package_version.tar.xz"
    if ! tar -xf "$package_file"; then
        log_error "解压失败: $package_file"
        return 1
    fi

    # 进入源码目录
    cd "$package_name-$package_version"

    # 配置（使用交叉编译选项）
    log_info "配置交叉编译..."
    local cross_options="--target=$LFS_TGT --host=$LFS_TGT --build=$(./config.guess)"

    if ! ./configure $cross_options --prefix=$LFS/tools $configure_options; then
        log_error "交叉编译配置失败"
        return 1
    fi

    # 编译和安装
    if ! make $LFS_MAKEFLAGS && make install; then
        log_error "交叉编译失败"
        return 1
    fi

    # 清理
    cd "$LFS/sources"
    rm -rf "$build_dir"

    log_info "交叉编译 $package_name 成功"
    return 0
}

# 进度跟踪
init_progress() {
    local total_steps=$1
    echo 0 > $LFS/.build_progress
    echo $total_steps > $LFS/.build_total
}

update_progress() {
    local current_step=$1
    echo $current_step > $LFS/.build_progress

    local total=$(cat $LFS/.build_total)
    local percentage=$((current_step * 100 / total))

    log_info "构建进度: $current_step/$total ($percentage%)"
}

# 错误处理
error_handler() {
    local exit_code=$?
    local line_number=$1

    log_error "构建失败于第 $line_number 行，退出码: $exit_code"

    # 保存错误信息
    echo "失败时间: $(date)" > $LFS/logs/error_info.txt
    echo "失败行号: $line_number" >> $LFS/logs/error_info.txt
    echo "退出码: $exit_code" >> $LFS/logs/error_info.txt
    echo "当前目录: $(pwd)" >> $LFS/logs/error_info.txt
    echo "最后命令: $BASH_COMMAND" >> $LFS/logs/error_info.txt

    exit $exit_code
}

# 设置错误处理
trap 'error_handler $LINENO' ERR

# 导出函数
export -f log_info log_warn log_error check_environment
export -f build_package build_cross_package
export -f init_progress update_progress
EOF

# 设置执行权限
chmod +x $LFS/lib/lfs_build.sh
```

### 构建进度跟踪
```bash
# 创建进度跟踪脚本
cat > $LFS/bin/build_progress.sh << 'EOF'
#!/bin/bash
# 构建进度跟踪脚本

LFS=${LFS:-/mnt/lfs}

show_progress() {
    if [ -f "$LFS/.build_progress" ] && [ -f "$LFS/.build_total" ]; then
        local current=$(cat $LFS/.build_progress)
        local total=$(cat $LFS/.build_total)
        local percentage=$((current * 100 / total))

        echo "构建进度: $current/$total ($percentage%)"

        # 显示进度条
        local bar_length=50
        local filled_length=$((current * bar_length / total))
        local bar=$(printf "%-${bar_length}s" "=" | sed "s/ /=/g" | cut -c1-$filled_length)
        local empty=$(printf "%-$((bar_length - filled_length))s" "")

        echo -ne "[$bar$empty] $percentage%\r"
    else
        echo "进度信息不可用"
    fi
}

show_build_status() {
    echo "=== LFS构建状态 ==="
    echo "LFS目录: $LFS"
    echo "目标架构: $LFS_TGT"
    echo "并行任务数: $(nproc)"
    echo ""

    show_progress
    echo ""

    # 显示最近的日志
    if [ -f "$LFS/logs/build.log" ]; then
        echo "最近构建日志:"
        tail -10 "$LFS/logs/build.log"
    fi
}

case "$1" in
    "show")
        show_progress
        ;;
    "status")
        show_build_status
        ;;
    *)
        echo "用法: $0 {show|status}"
        echo "  show   - 显示当前进度"
        echo "  status - 显示完整状态"
        ;;
esac
EOF

chmod +x $LFS/bin/build_progress.sh
```

## 🔧 开发环境配置

### 编辑器配置
```bash
# 安装和配置vim
cat > $LFS/root/.vimrc << 'EOF'
syntax on
set number
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set background=dark
set mouse=a
EOF

# 配置nano
cat > $LFS/root/.nanorc << 'EOF'
set autoindent
set const
set mouse
set smooth
set tabsize 4
set tabstospaces
include /usr/share/nano/*.nanorc
EOF
```

### Shell配置
```bash
# 创建root用户的bashrc
cat > $LFS/root/.bashrc << 'EOF'
# root用户bashrc

# 彩色提示符
PS1='\[\e[1;32m\][\u@\h \W]\$\[\e[0m\] '

# 别名
alias ls='ls --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# 环境变量
export EDITOR=nano
export PAGER=less
export PATH=/usr/local/bin:/usr/bin:/bin

# 历史记录
HISTSIZE=1000
HISTFILESIZE=2000
shopt -s histappend

# 检查窗口大小
shopt -s checkwinsize

# 彩色man页面
export LESS_TERMCAP_mb=$'\e[1;31m'
export LESS_TERMCAP_md=$'\e[1;36m'
export LESS_TERMCAP_me=$'\e[0m'
export LESS_TERMCAP_se=$'\e[0m'
export LESS_TERMCAP_so=$'\e[01;33m'
export LESS_TERMCAP_ue=$'\e[0m'
export LESS_TERMCAP_us=$'\e[01;32m'
EOF

# 创建系统范围的profile
cat > $LFS/etc/profile << 'EOF'
# /etc/profile

# 系统范围的环境变量
export PATH=/usr/local/bin:/usr/bin:/bin

# 语言设置
export LANG=en_US.UTF-8

# 编辑器设置
export EDITOR=nano

# 分页器设置
export PAGER=less

# 加载bashrc
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# 欢迎信息
echo "欢迎使用 LFS 系统!"
echo "输入 'help' 获取帮助信息"
EOF
```

### 构建文档
```bash
# 创建构建文档
cat > $LFS/README_BUILD.txt << 'EOF'
LFS构建指南
============

本系统正在构建中，请遵循以下步骤：

1. 准备阶段
   - 宿主系统配置 ✓
   - 分区和文件系统 ✓
   - 源码包下载 ✓
   - 目录结构创建 ✓

2. 工具链构建
   - Binutils
   - GCC
   - Linux API Headers
   - Glibc

3. 基本系统
   - 核心工具
   - 系统库
   - 网络工具

4. 引导和内核
   - GRUB
   - Linux内核

5. 系统配置
   - 网络设置
   - 用户管理
   - 服务配置

构建日志: /logs/build.log
构建进度: 使用 'build_progress.sh status' 查看

如遇问题，请查看:
- /logs/error_info.txt (错误信息)
- http://www.linuxfromscratch.org/lfs/ (官方文档)
EOF
```

## 📊 构建监控

### 实时监控脚本
```bash
# 创建监控脚本
cat > $LFS/bin/monitor_build.sh << 'EOF'
#!/bin/bash
# 构建过程监控脚本

LFS=${LFS:-/mnt/lfs}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 显示系统信息
show_system_info() {
    echo -e "${BLUE}=== 系统信息 ===${NC}"
    echo "CPU核心数: $(nproc)"
    echo "总内存: $(free -h | awk 'NR==2{print $2}')"
    echo "可用内存: $(free -h | awk 'NR==2{print $7}')"
    echo "磁盘使用: $(df -h $LFS | awk 'NR==2{print $3"/"$2" ("$5" used)"}')"
    echo ""
}

# 显示构建状态
show_build_status() {
    echo -e "${BLUE}=== 构建状态 ===${NC}"

    if [ -f "$LFS/.build_progress" ]; then
        local current=$(cat $LFS/.build_progress)
        local total=$(cat $LFS/.build_total 2>/dev/null || echo "1")
        local percentage=$((current * 100 / total))

        echo "进度: $current/$total ($percentage%)"

        # 进度条
        local bar_length=50
        local filled=$((current * bar_length / total))
        printf "["
        printf "%${filled}s" | tr ' ' '='
        printf "%$((bar_length - filled))s" | tr ' ' '-'
        printf "] %d%%\n" $percentage
    else
        echo "进度信息不可用"
    fi
    echo ""
}

# 显示最近日志
show_recent_logs() {
    echo -e "${BLUE}=== 最近日志 ===${NC}"
    if [ -f "$LFS/logs/build.log" ]; then
        tail -20 "$LFS/logs/build.log" | while read line; do
            if echo "$line" | grep -q "ERROR"; then
                echo -e "${RED}$line${NC}"
            elif echo "$line" | grep -q "WARN"; then
                echo -e "${YELLOW}$line${NC}"
            else
                echo "$line"
            fi
        done
    else
        echo "日志文件不存在"
    fi
    echo ""
}

# 显示资源使用
show_resource_usage() {
    echo -e "${BLUE}=== 资源使用 ===${NC}"
    echo "CPU使用率:"
    top -bn1 | head -10 | tail -5
    echo ""
    echo "内存使用:"
    free -h
    echo ""
}

# 主函数
main() {
    clear
    echo -e "${GREEN}LFS构建监控器${NC}"
    echo "按 Ctrl+C 退出"
    echo ""

    while true; do
        show_system_info
        show_build_status
        show_recent_logs
        show_resource_usage

        echo "最后更新: $(date)"
        sleep 5
        clear
    done
}

# 检查参数
case "$1" in
    "info")
        show_system_info
        ;;
    "status")
        show_build_status
        ;;
    "logs")
        show_recent_logs
        ;;
    "resources")
        show_resource_usage
        ;;
    "monitor")
        main
        ;;
    *)
        echo "用法: $0 {info|status|logs|resources|monitor}"
        echo "  info     - 显示系统信息"
        echo "  status   - 显示构建状态"
        echo "  logs     - 显示最近日志"
        echo "  resources- 显示资源使用"
        echo "  monitor  - 实时监控模式"
        ;;
esac
EOF

chmod +x $LFS/bin/monitor_build.sh
```

### 自动化备份
```bash
# 创建备份脚本
cat > $LFS/bin/backup_build.sh << 'EOF'
#!/bin/bash
# 构建过程备份脚本

LFS=${LFS:-/mnt/lfs}
BACKUP_DIR=${BACKUP_DIR:-/mnt/backup}

# 创建备份
create_backup() {
    local backup_name="lfs_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"

    echo "创建备份: $backup_name"

    # 创建备份目录
    mkdir -p "$backup_path"

    # 备份重要文件
    cp -r "$LFS/tools" "$backup_path/"
    cp -r "$LFS/usr" "$backup_path/"
    cp -r "$LFS/etc" "$backup_path/"
    cp -r "$LFS/var" "$backup_path/"
    cp -r "$LFS/logs" "$backup_path/"

    # 备份进度信息
    cp "$LFS/.build_progress" "$backup_path/" 2>/dev/null || true
    cp "$LFS/.build_total" "$backup_path/" 2>/dev/null || true

    # 创建压缩包
    cd "$BACKUP_DIR"
    tar -czf "${backup_name}.tar.gz" "$backup_name"

    # 清理临时文件
    rm -rf "$backup_name"

    echo "备份完成: ${backup_name}.tar.gz"
}

# 恢复备份
restore_backup() {
    local backup_file="$1"

    if [ ! -f "$backup_file" ]; then
        echo "备份文件不存在: $backup_file"
        exit 1
    fi

    echo "从备份恢复: $backup_file"

    # 解压备份
    local temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"

    # 恢复文件
    cp -r "$temp_dir"/*/tools/* "$LFS/tools/" 2>/dev/null || true
    cp -r "$temp_dir"/*/usr/* "$LFS/usr/" 2>/dev/null || true
    cp -r "$temp_dir"/*/etc/* "$LFS/etc/" 2>/dev/null || true
    cp -r "$temp_dir"/*/var/* "$LFS/var/" 2>/dev/null || true
    cp -r "$temp_dir"/*/logs/* "$LFS/logs/" 2>/dev/null || true

    # 恢复进度信息
    cp "$temp_dir"/*/.build_progress "$LFS/" 2>/dev/null || true
    cp "$temp_dir"/*/.build_total "$LFS/" 2>/dev/null || true

    # 清理临时文件
    rm -rf "$temp_dir"

    echo "恢复完成"
}

# 显示备份列表
list_backups() {
    echo "可用备份:"
    ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "没有找到备份文件"
}

# 主函数
case "$1" in
    "create")
        create_backup
        ;;
    "restore")
        if [ -z "$2" ]; then
            echo "用法: $0 restore <备份文件>"
            exit 1
        fi
        restore_backup "$2"
        ;;
    "list")
        list_backups
        ;;
    *)
        echo "用法: $0 {create|restore|list}"
        echo "  create          - 创建备份"
        echo "  restore <file>  - 从备份恢复"
        echo "  list            - 列出可用备份"
        ;;
esac
EOF

chmod +x $LFS/bin/backup_build.sh
```

## 🚨 故障排除

### 环境变量问题
```bash
# 检查环境变量
echo "LFS=$LFS"
echo "LFS_TGT=$LFS_TGT"
echo "PATH=$PATH"

# 重新设置环境变量
source ~/.bashrc

# 验证工具链
which gcc
gcc --version
```

### 权限问题
```bash
# 检查目录权限
ls -ld $LFS
ls -ld $LFS/tools

# 修复权限
sudo chown -R lfs:lfs $LFS
```

### 构建失败恢复
```bash
# 查看错误日志
cat $LFS/logs/error_info.txt

# 清理失败的构建
cd $LFS/sources
rm -rf build_*

# 重新开始构建
# 参考具体包的构建步骤
```

## 📚 相关资源

- [LFS官方文档 - 环境设置](http://www.linuxfromscratch.org/lfs/view/stable/chapter04/chapter04.html)
- [Bash参考手册](https://www.gnu.org/software/bash/manual/)
- [Linux环境变量](https://wiki.archlinux.org/title/Environment_variables)

---

*最近更新: {{ .Lastmod.Format "2006-01-02" }}*