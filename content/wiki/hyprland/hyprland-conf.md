+++
date = '2026-02-07T01:10:30+08:00'
draft = false
title = 'Hyprland 配置详解'
comments = true
+++

## Hyprland 主配置文件详解

Hyprland 的主配置文件是 `~/.config/hypr/hyprland.conf`，它包含了窗口管理、输入、外观等所有核心设置。下面我将详细解释每个配置项的功能。

### 1. 程序定义

```conf
$terminal = ghostty
$fileManager = nautilus
$menu = rofi -show drun
$browser = firefox
$lock = hyprlock
$editor = com.vscodium.codium-insiders
$screenshot = grim
```

**功能解释：**
- 这些是自定义变量，用于在配置文件的其他地方引用常用程序
- `$terminal`：定义您使用的终端模拟器（这里是 Ghostty）
- `$fileManager`：定义您使用的文件管理器（这里是 Nautilus）
- `$menu`：定义应用启动器（这里是 Rofi）
- `$browser`：定义默认浏览器（这里是 Firefox）
- `$lock`：定义锁屏工具（这里是 Hyprlock）
- `$editor`：定义代码编辑器（这里是 VSCodium Insiders）
- `$screenshot`：定义截图工具（这里是 Grim）

**作用：** 统一管理常用程序，方便在整个配置文件中修改和引用。

### 2. 自动启动

```conf
exec-once = waybar & hyprpaper & mako
exec-once = fcitx5 --replace -d
```

**功能解释：**
- `exec-once`：用于在 Hyprland 启动时自动运行命令
- 第一个命令启动状态栏 Waybar、壁纸管理 Hyprpaper 和通知守护进程 Mako
- 第二个命令启动输入法框架 Fcitx5

**作用：** 确保必要的系统组件在 Hyprland 启动时自动运行，提供完整的用户体验。

### 3. 环境变量

```conf
env = XCURSOR_SIZE,24
env = HYPRCURSOR_SIZE,24
```

**功能解释：**
- 定义环境变量
- `XCURSOR_SIZE`：设置系统光标大小（这里是 24 像素）
- `HYPRCURSOR_SIZE`：设置 Hyprland 特定的光标大小（这里是 24 像素）

**作用：** 统一控制系统的光标显示大小。

### 4. 一般设置

```conf
general {
    gaps_in = 3
    gaps_out = 10
    border_size = 2
    col.active_border = rgba(33ccffee) rgba(00ff99ee) 45deg
    col.inactive_border = rgba(595959aa)
    resize_on_border = false
    allow_tearing = false
    layout = dwindle
}
```

**功能解释：**
- `gaps_in`：窗口之间的内部间隙（这里是 3 像素）
- `gaps_out`：屏幕边缘与窗口之间的外部间隙（这里是 10 像素）
- `border_size`：窗口边框的粗细（这里是 2 像素）
- `col.active_border`：激活窗口的边框颜色（这里是蓝绿色渐变）
- `col.inactive_border`：非激活窗口的边框颜色（这里是灰色半透明）
- `resize_on_border`：是否允许通过拖动窗口边框调整大小（这里禁止）
- `allow_tearing`：是否允许画面撕裂（这里禁止）
- `layout`：默认的窗口布局（这里是 dwindle，一种螺旋式布局）

**作用：** 控制 Hyprland 的基本外观和行为。

### 5. 窗口装饰

```conf
decoration {
    rounding = 15
    rounding_power = 2
    active_opacity = 1.0
    inactive_opacity = 0.75
    shadow {
        enabled = true
        range = 4
        render_power = 3
        color = rgba(1a0b1aac)
    }
    blur {
        enabled = true
        size = 3
        passes = 1
        vibrancy = 0.1696
        ignore_opacity = true
    }
}
```

**功能解释：**
- `rounding`：窗口圆角大小（这里是 15 像素）
- `rounding_power`：圆角的平滑度（这里是 2）
- `active_opacity`：激活窗口的不透明度（这里是 1.0，完全不透明）
- `inactive_opacity`：非激活窗口的不透明度（这里是 0.75，半透明）
- `shadow`：窗口阴影设置
  - `enabled`：是否启用阴影（这里启用）
  - `range`：阴影范围（这里是 4 像素）
  - `render_power`：阴影的渲染质量（这里是 3）
  - `color`：阴影颜色（这里是深紫色半透明）
- `blur`：窗口模糊效果设置
  - `enabled`：是否启用模糊（这里启用）
  - `size`：模糊大小（这里是 3 像素）
  - `passes`：模糊次数（这里是 1 次）
  - `vibrancy`：模糊的活力值（这里是 0.1696）
  - `ignore_opacity`：是否忽略不透明度（这里启用）

**作用：** 控制窗口的视觉效果，使界面更加美观。

### 6. 动画设置

```conf
animations {
    enabled = true
    bezier = linear, 0, 0, 1, 1
    bezier = md3_decel, 0.05, 0.7, 0.1, 1
    animation = windows, 1, 3, md3_decel, popin 60%
    animation = windowsIn, 1, 3, md3_decel, popin 60%
    animation = windowsOut, 1, 3, md3_accel, popin 60%
    animation = border, 1, 10, default
    animation = fade, 1, 3, md3_decel
    animation = layersIn, 1, 3, menu_decel, slide
    animation = layersOut, 1, 1.6, menu_accel
    animation = fadeLayersIn, 1, 2, menu_decel
    animation = fadeLayersOut, 1, 4.5, menu_accel
    animation = workspaces, 1, 7, menu_decel, slide
    animation = specialWorkspace, 1, 3, md3_decel, slidevert
}
```

**功能解释：**
- `enabled`：是否启用动画（这里启用）
- `bezier`：定义贝塞尔曲线，用于控制动画的缓动效果
- `animation`：定义具体的动画
  - `windows`：窗口显示/隐藏动画
  - `windowsIn`：窗口进入动画
  - `windowsOut`：窗口退出动画
  - `border`：边框动画
  - `fade`：淡入淡出动画
  - `layersIn`：图层进入动画
  - `layersOut`：图层退出动画
  - `workspaces`：工作区切换动画
  - `specialWorkspace`：特殊工作区动画

**作用：** 使窗口操作更加流畅和美观。

### 7. 输入设置

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
        natural_scroll = true
    }
}
```

**功能解释：**
- `kb_layout`：键盘布局（这里是 US 布局）
- `follow_mouse`：是否跟随鼠标焦点（这里启用）
- `sensitivity`：鼠标灵敏度（这里是 0，默认值）
- `touchpad`：触摸板设置
  - `natural_scroll`：是否启用自然滚动（这里启用）

**作用：** 控制输入设备的行为。

### 8. 快捷键绑定

```conf
$mainMod = SUPER

# 基本操作
bind = $mainMod, Q, exec, $terminal
bind = $mainMod, C, killactive,
bind = $mainMod, M, exit,
bind = $mainMod, L, exec, $lock,
bind = $mainMod, E, exec, $fileManager
bind = $mainMod, V, togglefloating,
bind = $mainMod, R, exec, $menu
bind = $mainMod, P, pseudo, # dwindle
bind = $mainMod, B, exec, $browser
bind = $mainMod, J, togglesplit, # dwindle
bind = $mainMod, Z, exec, $editor
bind = $mainMod, Print, exec, $screenshot

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
bindel = ,XF86MonBrightnessUp, exec, brightnessctl -e4 -n2 set 5%+
bindel = ,XF86MonBrightnessDown, exec, brightnessctl -e4 -n2 set 5%-
```

**功能解释：**
- `$mainMod`：定义主修饰键（这里是 SUPER 键，即 Windows 键）
- `bind`：绑定快捷键到特定操作
- `bindel`：绑定事件快捷键（如多媒体键）

**作用：** 提供快捷的操作方式，提高用户效率。

### 9. 手势设置

```conf
gesture = 3, horizontal, workspace
gesture = 3, down, close
gesture = 4, up, mod: SUPER, scale: 1.5, fullscreen
gesture = 4, down,  scale: 1.5, float
```

**功能解释：**
- `gesture`：定义触摸板手势
- `3, horizontal`：三指水平滑动（切换工作区）
- `3, down`：三指向下滑动（关闭窗口）
- `4, up`：四指向上滑动（全屏显示）
- `4, down`：四指向下滑动（浮动窗口）

**作用：** 提供触摸板手势操作，增强用户体验。

### 10. XWayland 设置

```conf
xwayland {
    force_zero_scaling = true
}
```

**功能解释：**
- `force_zero_scaling`：禁止 Hyprland 对 XWayland 窗口进行二次缩放

**作用：** 确保 XWayland 应用程序在高分辨率屏幕上显示正常。

### 总结

Hyprland 的配置文件提供了丰富的自定义选项，允许用户根据自己的需求调整窗口管理、输入、外观等方面的行为。通过理解每个配置项的功能，您可以创建一个完全符合您个人偏好的工作环境。
