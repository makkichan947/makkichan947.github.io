+++
date = '2026-02-07T01:10:15+08:00'
draft = false
title = 'SDDM 配置'
comments = true
+++

## SDDM 配置

SDDM（Simple Desktop Display Manager）是一个轻量级的 Wayland 和 X11 显示管理器，用于登录界面。下面我将介绍如何配置 SDDM 以配合 Hyprland 使用。

### 1. 安装 SDDM

在 Arch Linux 上，您可以使用以下命令安装 SDDM：

```bash
sudo pacman -S sddm
```

### 2. 启用 SDDM

要使 SDDM 在系统启动时自动运行，您需要启用它：

```bash
sudo systemctl enable sddm --now
```

### 3. 配置 SDDM

SDDM 的主要配置文件位于 `/etc/sddm.conf`，但通常建议通过创建 `/etc/sddm.conf.d/` 目录下的配置文件来进行定制。

#### 3.1 基本配置

创建一个配置文件 `/etc/sddm.conf.d/hyprland.conf`：

```conf
[General]
# 启用自动登录（可选）
# AutoLoginEnable=true
# AutoLoginUser=your_username
# AutoLoginSession=hyprland

[Theme]
# SDDM 主题
Current=breeze

[Users]
# 显示的最大用户数
MaximumUid=60000
MinimumUid=1000

[Wayland]
# 默认的 Wayland 会话
DefaultSession=hyprland.desktop

[X11]
# 默认的 X11 会话
DefaultSession=hyprland.desktop
```

**功能解释：**
- `AutoLoginEnable`：是否启用自动登录（这里注释掉了）
- `AutoLoginUser`：自动登录的用户名（这里注释掉了）
- `AutoLoginSession`：自动登录的会话（这里是 hyprland）
- `Current`：SDDM 主题（这里是 breeze）
- `MaximumUid` 和 `MinimumUid`：显示的用户 UID 范围
- `DefaultSession`：默认的会话（这里是 hyprland.desktop）

**作用：** 配置 SDDM 的基本行为。

### 4. 配置 Hyprland 会话

SDDM 需要知道如何启动 Hyprland。您需要创建一个会话文件。

#### 4.1 创建会话文件

创建一个会话文件 `/usr/share/wayland-sessions/hyprland.desktop`：

```desktop
[Desktop Entry]
Name=Hyprland
Comment=An intelligent dynamic tiling Wayland compositor
Exec=/usr/bin/hyprland
Type=Application
Keywords=tiling;wayland;compositor;
DesktopNames=Hyprland
X-KDE-Wayland-Enabled=true
X-GNOME-Wayland-Enabled=true
```

**功能解释：**
- `Name`：会话的名称（显示在 SDDM 界面上）
- `Comment`：会话的描述
- `Exec`：启动 Hyprland 的命令
- `Type`：文件类型
- `Keywords`：关键字（用于搜索）
- `DesktopNames`：桌面名称
- `X-KDE-Wayland-Enabled`：是否支持 KDE Wayland
- `X-GNOME-Wayland-Enabled`：是否支持 GNOME Wayland

**作用：** 告诉 SDDM 如何启动 Hyprland。

### 5. 配置 SDDM 主题

SDDM 支持多种主题，您可以根据自己的喜好选择。

#### 5.1 安装主题

您可以从以下地方获取 SDDM 主题：

- [SDDM 主题网站](https://store.kde.org/browse/cat/101/)
- GitHub（搜索 "sddm theme"）

#### 5.2 应用主题

将主题文件解压到 `/usr/share/sddm/themes/` 目录下，然后在 `/etc/sddm.conf.d/hyprland.conf` 中修改 `Current` 字段：

```conf
[Theme]
Current=your-theme-name
```

### 6. 配置 SDDM 背景

#### 6.1 修改主题背景

大多数 SDDM 主题允许您自定义背景。您可以查看主题的 README 文件，了解如何修改背景。

#### 6.2 覆盖背景（适用于所有主题）

您可以创建一个 `/etc/sddm.conf.d/background.conf` 文件：

```conf
[Theme]
Background=/path/to/your/background/image.jpg
```

**作用：** 覆盖主题的默认背景。

### 7. 常见问题

#### 7.1 SDDM 无法启动 Hyprland

- 检查 `Exec` 字段是否指向正确的 Hyprland 可执行文件
- 检查会话文件的权限是否正确（应该是 644）
- 检查 Hyprland 是否安装在正确的位置

#### 7.2 SDDM 主题不显示

- 检查主题是否正确安装在 `/usr/share/sddm/themes/` 目录下
- 检查主题的名称是否正确
- 检查 `Current` 字段是否指向正确的主题

#### 7.3 SDDM 无法登录

- 检查用户名和密码是否正确
- 检查用户是否有访问权限
- 检查系统日志（使用 `journalctl -xe` 命令）

### 8. 其他配置

#### 8.1 配置语言

您可以在 `/etc/sddm.conf.d/language.conf` 中配置语言：

```conf
[General]
Language=en_US.UTF-8
```

#### 8.2 配置键盘布局

您可以在 `/etc/sddm.conf.d/keyboard.conf` 中配置键盘布局：

```conf
[Keyboard]
Layout=us
Variant=
```

#### 8.3 配置时区

您可以在 `/etc/sddm.conf.d/timezone.conf` 中配置时区：

```conf
[General]
Timezone=Asia/Shanghai
```

### 总结

配置 SDDM 以配合 Hyprland 使用需要以下步骤：

1. 安装 SDDM
2. 启用 SDDM
3. 配置 SDDM
4. 创建 Hyprland 会话文件
5. 配置 SDDM 主题
6. 配置 SDDM 背景

通过这些配置，您可以创建一个符合个人喜好的登录界面。
