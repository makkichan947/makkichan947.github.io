+++
date = '2026-02-01T10:12:36+08:00'
draft = false
title = '默认的Hyprland'
comments = true
+++

当您第一次安装 Hyprland 时，它会创建一个默认的配置文件，提供基本的窗口管理功能。

## 默认配置文件位置

默认配置文件位于 `~/.config/hypr/hyprland.conf`，如果该文件不存在，Hyprland 会使用内置的默认配置。

## 默认配置内容

### 程序定义

默认配置中定义了以下变量：

```conf
$terminal = alacritty
$fileManager = dolphin
$menu = rofi -show drun
$browser = firefox
$lock = swaylock
$editor = kate
$screenshot = grim
```

### 自动启动

```conf
exec-once = swayidle & swaybg
exec-once = fcitx5 --replace -d
```

### 环境变量

```conf
env = XCURSOR_SIZE,24
env = HYPRCURSOR_SIZE,24
```

### 外观和感觉

```conf
general {
    gaps_in = 5
    gaps_out = 10
    border_size = 1
    col.active_border = rgba(33ccffee)
    col.inactive_border = rgba(595959aa)
    resize_on_border = true
    allow_tearing = false
    layout = dwindle
}

decoration {
    rounding = 10
    rounding_power = 2
    active_opacity = 1.0
    inactive_opacity = 0.8
    shadow {
        enabled = true
        range = 3
        render_power = 2
        color = rgba(000000aa)
    }
    blur {
        enabled = false
        size = 3
        passes = 1
        vibrancy = 0.0
        ignore_opacity = true
    }
}
```

### 动画

```conf
animations {
    enabled = true
    animation = windows, 1, 2, default
    animation = windowsIn, 1, 2, default
    animation = windowsOut, 1, 2, default
    animation = border, 1, 5, default
    animation = fade, 1, 2, default
}
```

### 输入设置

```conf
input {
    kb_layout = us
    kb_variant =
    kb_model =
    kb_options =
    kb_rules =
    follow_mouse = 1
    sensitivity = 0
    touchpad {
        natural_scroll = false
    }
}
```

### 快捷键

```conf
$mainMod = SUPER

# 基本操作
bind = $mainMod, RETURN, exec, $terminal
bind = $mainMod, C, killactive,
bind = $mainMod, M, exit,
bind = $mainMod, L, exec, $lock,
bind = $mainMod, E, exec, $fileManager
bind = $mainMod, V, togglefloating,
bind = $mainMod, R, exec, $menu

# 工作区导航
bind = $mainMod, 1, workspace, 1
bind = $mainMod, 2, workspace, 2
bind = $mainMod, 3, workspace, 3
bind = $mainMod, 4, workspace, 4
bind = $mainMod, 5, workspace, 5
bind = $mainMod, 6, workspace, 6
bind = $mainMod, 7, workspace, 7
bind = $mainMod, 8, workspace, 8
bind = $mainMod, 9, workspace, 9
bind = $mainMod, 0, workspace, 10

bind = $mainMod SHIFT, 1, movetoworkspace, 1
bind = $mainMod SHIFT, 2, movetoworkspace, 2
bind = $mainMod SHIFT, 3, movetoworkspace, 3
bind = $mainMod SHIFT, 4, movetoworkspace, 4
bind = $mainMod SHIFT, 5, movetoworkspace, 5
bind = $mainMod SHIFT, 6, movetoworkspace, 6
bind = $mainMod SHIFT, 7, movetoworkspace, 7
bind = $mainMod SHIFT, 8, movetoworkspace, 8
bind = $mainMod SHIFT, 9, movetoworkspace, 9
bind = $mainMod SHIFT, 0, movetoworkspace, 10

# 窗口调整
bind = $mainMod, left, movefocus, l
bind = $mainMod, right, movefocus, r
bind = $mainMod, up, movefocus, u
bind = $mainMod, down, movefocus, d

# 特殊工作区
bind = $mainMod, S, togglespecialworkspace, magic
bind = $mainMod SHIFT, S, movetoworkspace, special:magic

# 多媒体键
bindel = ,XF86AudioRaiseVolume, exec, wpctl set-volume -l 1 @DEFAULT_AUDIO_SINK@ 5%+
bindel = ,XF86AudioLowerVolume, exec, wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%-
bindel = ,XF86AudioMute, exec, wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle
bindel = ,XF86MonBrightnessUp, exec, brightnessctl set 5%+
bindel = ,XF86MonBrightnessDown, exec, brightnessctl set 5%-
```

## 自定义默认配置

您可以通过以下步骤自定义默认配置：

1. 复制默认配置到 `~/.config/hypr/hyprland.conf`
2. 编辑配置文件
3. 保存并重新启动 Hyprland

## 配置建议

### 1. 自定义程序定义

修改程序定义以使用您喜欢的应用程序：

```conf
$terminal = ghostty  # 替换为您喜欢的终端
$fileManager = nautilus  # 替换为您喜欢的文件管理器
$menu = rofi -show drun  # 替换为您喜欢的应用启动器
```

### 2. 优化快捷键

根据您的使用习惯调整快捷键：

```conf
# 使用 Alt 键作为主修饰符
$mainMod = ALT

# 自定义终端快捷键
bind = $mainMod, T, exec, $terminal
```

### 3. 调整外观

修改窗口外观以符合您的喜好：

```conf
decoration {
    rounding = 15  # 增大圆角
    active_opacity = 0.95  # 调整透明度
    inactive_opacity = 0.7  # 调整透明度
    shadow {
        color = rgba(1a0b1aac)  # 调整阴影颜色
    }
}
```

### 4. 启用模糊效果

```conf
decoration {
    blur {
        enabled = true
        size = 5
        passes = 2
        vibrancy = 0.2
    }
}
```

## 验证配置

使用以下命令验证配置文件的语法：

```bash
hyprctl reload
```

如果配置文件有错误，会在终端中显示。

## 学习资源

- [Hyprland 官方配置文档](https://wiki.hypr.land/Configuring)
- [Hyprland 变量列表](https://wiki.hypr.land/Configuring/Variables)
- [Hyprland 快捷键文档](https://wiki.hypr.land/Configuring/Binds)
