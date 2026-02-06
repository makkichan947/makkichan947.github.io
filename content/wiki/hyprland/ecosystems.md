+++
date = '2026-02-01T11:22:14+08:00'
draft = false
title = 'Hyprland的生态系统'
comments = true
+++

Hyprland 有一个强大且活跃的生态系统，包含了各种工具和组件，用于增强用户体验。

## 1. 状态栏

### Waybar

Waybar 是一个高度可定制的 Wayland 状态栏，支持 Hyprland 工作区、窗口信息等。

#### 安装

```bash
sudo pacman -S waybar
```

#### 配置

配置文件位于 `~/.config/waybar/config`：

```json
{
  "layer": "top",
  "position": "top",
  "height": 18,
  "spacing": 5,
  "modules-left": [
    "custom/arch",
    "hyprland/window",
    "custom/file",
    "custom/edit",
    "custom/view",
    "custom/go",
    "custom/window"
  ],
  "modules-center": [
    "hyprland/workspaces",
    "mpd"
  ],
  "modules-right": [
    "keyboard-state",
    "tray",
    "bluetooth",
    "network",
    "battery",
    "clock"
  ],
  "hyprland/window": {
    "format": " {title}",
    "max-length": 20,
    "empty": " 访达",
    "separate-outputs": true,
    "tooltip": true,
    "tooltip-format": "当前窗口\nClass: {class}\nTitle: {title}"
  },
  "hyprland/workspaces": {
    "format": "{icon}",
    "format-icons": {
      "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
      "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
      "urgent": "",
      "active": "",
      "default": ""
    },
    "tooltip": true,
    "tooltip-format": "Workspace {name}\n{windows} 个窗口",
    "on-scroll-up": "exec hyprctl dispatch workspace e-1",
    "on-scroll-down": "exec hyprctl dispatch workspace e+1",
    "on-click": "activate",
    "all-outputs": false,
    "sort-by-number": true
  },
  "clock": {
    "format": "󰥔 {:%H:%M:%S}",
    "format-alt": "󰃭 {:%Y-%m-%d %a}",
    "tooltip-format": "<big>{:%Y %B}</big>\n<tt><small>{calendar}</small></tt>",
    "calendar": {
        "mode": "year",
        "mode-mon-col": 3,
        "weeks-pos": "right",
        "on-scroll": 1,
        "on-click-right": "mode",
        "format": {
            "months": "<span color='#89b4fa'><b>{}</b></span>",
            "days": "<span color='#a6e3a1'><b>{}</b></span>",
            "weeks": "<span color='#b4befe'>{}</span>",
            "weekdays": "<span color='#cdd6f4'><b>{}</b></span>",
            "today": "<span color='#f38ba8'><b><u>{}</u></b></span>"
        }
    },
    "interval": 1,
    "timezone": "Asia/Shanghai"
  }
}
```

#### 样式

样式文件位于 `~/.config/waybar/style.css`。

## 2. 通知守护进程

### Mako

Mako 是一个轻量级的 Wayland 通知守护进程。

#### 安装

```bash
sudo pacman -S mako
```

#### 配置

配置文件位于 `~/.config/mako/config`：

```conf
sort=-time
layer=overlay
background-color=#22222266
width=450
height=50
border-size=1
border-color=#d3e4c9
border-radius=12
icons=1
max-icon-size=64
default-timeout=5000
ignore-timeout=0
font=Noto Sans CJK TC 14
margin=12
padding=12,20

[urgency=low]
border-color=#aaaaaa

[urgency=normal]
border-color=#cceeff

[urgency=critical]
border-color=#ff7777
default-timeout=0
```

## 3. 终端

### Ghostty

Ghostty 是一个基于 WebKitGTK 的现代终端模拟器。

#### 安装

```bash
yay -S ghostty
```

#### 配置

配置文件位于 `~/.config/ghostty/config`。

## 4. 输入法

### Fcitx5

Fcitx5 是一个强大的输入法框架。

#### 安装

```bash
sudo pacman -S fcitx5 fcitx5-chinese-addons fcitx5-rime
```

#### 配置

配置文件位于 `~/.config/fcitx5/`。

## 5. 锁屏

### Hyprlock

Hyprlock 是 Hyprland 官方的锁屏工具。

#### 安装

```bash
sudo pacman -S hyprlock
```

#### 配置

配置文件位于 `~/.config/hypr/hyprlock.conf`：

```conf
general {
    lock_cmd = pidof hyprlock || hyprlock
    unlock_cmd = loginctl unlock-session $XDG_SESSION_ID
    before_sleep_cmd = loginctl lock-session $XDG_SESSION_ID
    after_sleep_cmd = hyprctl dispatch dpms on
    ignore_dbus_inhibit = false
}

background {
    monitor =
    path = ~/.config/background.png
    blur_passes = 0
    blur_size = 0
    noise = 0.0117
    contrast = 0.8916
    brightness = 0.8172
    vibrancy = 0.1696
    vibrancy_darkness = 0.0
}

input-field {
    monitor =
    size = 250, 60
    outline_thickness = 2
    dots_size = 0.2
    dots_spacing = 0.2
    dots_center = true
    outer_color = rgba(255, 255, 255, 0.1)
    inner_color = rgba(255, 255, 255, 0.05)
    font_color = rgb(255, 255, 255)
    fade_on_empty = false
    fade_timeout = 1000
    placeholder_text = <span foreground="##a0a0a0"><i>输入密码...</i></span>
    hide_input = false
    position = 0, -460
}

label {
    monitor =
    text = cmd[update:3600000] echo "上次登录: $(last -2 $USER | head -2 | tail -1 | awk '{print $4" "$5" "$6" "$7}')"
    color = rgba(200, 200, 200, 0.7)
    font_size = 14
    font_family = Inter
    position = 0, -400
    halign = center
    valign = center
}

label {
    monitor =
    text = cmd[update:1000] echo "$(date +"%H:%M")"
    color = rgba(255, 255, 255, 0.9)
    font_size = 125
    font_family = Inter
    position = 0, 370
    halign = center
    valign = center
}

label {
    monitor =
    text = cmd[update:1000] echo "$(date +"%A, %B %d")"
    color = rgba(255, 255, 255, 0.7)
    font_size = 25
    font_family = Inter
    position = 0, 470
    halign = center
    valign = center
}
```

## 6. 空闲管理

### Hypridle

Hypridle 是 Hyprland 官方的空闲管理工具。

#### 安装

```bash
sudo pacman -S hypridle
```

#### 配置

配置文件位于 `~/.config/hypr/hypridle.conf`：

```conf
general {
    lock_timeout = 600
    before_sleep_cmd = "hyprctl dispatch dpms off"
    after_sleep_cmd = "hyprctl dispatch dpms on"
    ignore_dbus_inhibit = false
}

listener {
    timeout = 600
    on-timeout = "hyprlock"
    on-resume = ""
}

listener {
    timeout = 900
    on-timeout = "hyprctl dispatch dpms off"
    on-resume = "hyprctl dispatch dpms on"
}
```

## 7. 壁纸

### Hyprpaper

Hyprpaper 是 Hyprland 官方的壁纸工具。

#### 安装

```bash
sudo pacman -S hyprpaper
```

#### 配置

配置文件位于 `~/.config/hypr/hyprpaper.conf`：

```conf
preload = ~/.config/background.png
wallpaper = eDP-1, ~/.config/background.png
```

## 8. 应用启动器

### Rofi

Rofi 是一个多功能的应用启动器。

#### 安装

```bash
sudo pacman -S rofi
```

#### 使用

```bash
rofi -show drun
```

## 9. 文件管理器

### Nautilus

Nautilus 是 GNOME 的文件管理器。

#### 安装

```bash
sudo pacman -S nautilus
```

## 10. 浏览器

### Firefox

Firefox 是一款开源的网页浏览器。

#### 安装

```bash
sudo pacman -S firefox
```

## 11. 编辑器

### VSCodium

VSCodium 是 Visual Studio Code 的开源版本。

#### 安装

```bash
yay -S vscodium-insiders
```

## 12. 其他工具

### Fastfetch

Fastfetch 是一个快速的系统信息显示工具。

#### 安装

```bash
sudo pacman -S fastfetch
```

#### 配置

配置文件位于 `~/.config/fastfetch/config.jsonc`。

### Starship

Starship 是一个快速的终端提示符。

#### 安装

```bash
sudo pacman -S starship
```

#### 配置

配置文件位于 `~/.config/starship.toml`。

## 13. 推荐组合

以下是一个推荐的 Hyprland 生态系统组合：

- **状态栏**：Waybar
- **通知守护进程**：Mako
- **终端**：Ghostty
- **输入法**：Fcitx5
- **锁屏**：Hyprlock
- **空闲管理**：Hypridle
- **壁纸**：Hyprpaper
- **应用启动器**：Rofi
- **文件管理器**：Nautilus
- **浏览器**：Firefox
- **编辑器**：VSCodium

这个组合提供了完整的功能和良好的用户体验。
