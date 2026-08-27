+++
date = '2026-02-07T01:10:45+08:00'
draft = false
title = 'Waybar 配置详解'
comments = true
+++

## Waybar 配置详解

Waybar 是一个高度可定制的 Wayland 状态栏，支持 Hyprland 工作区、窗口信息等。下面我将详细解释每个配置项的功能。

### 1. 基本配置

```json
{
  "layer": "top",
  "position": "top",
  "height": 18,
  "spacing": 5,
```

**功能解释：**
- `layer`：状态栏所在的层级（这里是 top，顶层）
- `position`：状态栏的位置（这里是 top，顶部）
- `height`：状态栏的高度（这里是 18 像素）
- `spacing`：模块之间的间距（这里是 5 像素）

**作用：** 定义 Waybar 的基本布局和外观。

### 2. 模块配置

```json
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
```

**功能解释：**
- `modules-left`：左侧模块列表
- `modules-center`：中间模块列表
- `modules-right`：右侧模块列表

**作用：** 定义 Waybar 各个模块的布局和位置。

### 3. 窗口信息模块

```json
  "hyprland/window": {
    "format": " {title}",
    "max-length": 20,
    "empty": " 访达",
    "separate-outputs": true,
    "tooltip": true,
    "tooltip-format": "当前窗口\nClass: {class}\nTitle: {title}"
  },
```

**功能解释：**
- `format`：窗口信息的显示格式（这里显示窗口标题）
- `max-length`：窗口标题的最大长度（这里是 20 字符）
- `empty`：没有窗口时显示的信息（这里是 " 访达"）
- `separate-outputs`：是否在多显示器上显示单独的窗口信息（这里启用）
- `tooltip`：是否显示工具提示（这里启用）
- `tooltip-format`：工具提示的格式（显示窗口类和标题）

**作用：** 显示当前焦点窗口的信息。

### 4. 工作区模块

```json
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
```

**功能解释：**
- `format`：工作区的显示格式（这里显示图标）
- `format-icons`：定义工作区的图标
  - 数字 1-10：对应的工作区编号
  - `urgent`：紧急状态的图标
  - `active`：激活状态的图标
  - `default`：默认状态的图标
- `tooltip`：是否显示工具提示（这里启用）
- `tooltip-format`：工具提示的格式（显示工作区名称和窗口数量）
- `on-scroll-up`：向上滚动事件（切换到上一个工作区）
- `on-scroll-down`：向下滚动事件（切换到下一个工作区）
- `on-click`：点击事件（激活工作区）
- `all-outputs`：是否在所有显示器上显示工作区（这里禁用）
- `sort-by-number`：是否按数字排序工作区（这里启用）

**作用：** 显示和管理 Hyprland 工作区。

### 5. 自定义模块

```json
  "custom/arch": {
    "format": "",
    "tooltip": false,
    "on-click": "",
    "interval": false
  },
  "custom/file": {
    "format": "文件",
    "on-click": "nautilus",
    "on-click-right": "ghostty -e yazi",
    "tooltip": false,
    "class": "app-menu"
  },
  "custom/edit": {
    "format": "编辑",
    "on-click": "com.vscodium.codium-insiders",
    "on-click-right": "gnome-text-editor",
    "tooltip": false,
    "class": "app-menu"
  },
  "custom/view": {
    "format": "显示",
    "on-click": "",
    "on-click-right": "",
    "tooltip": false,
    "class": "app-menu"
  },
  "custom/go": {
    "format": "前往",
    "on-click": "",
    "on-click-right": "",
    "tooltip": false,
    "class": "app-menu"
  },
  "custom/window": {
    "format": "窗口",
    "on-click": "ghostty",
    "on-click-right": "",
    "tooltip": false,
    "class": "app-menu"
  },
```

**功能解释：**
- `custom/arch`：显示 Arch Linux 图标
- `custom/file`：文件菜单模块，左键点击打开 Nautilus，右键点击打开 Yazi 终端文件管理器
- `custom/edit`：编辑菜单模块，左键点击打开 VSCodium，右键点击打开 Gnome 文本编辑器
- `custom/view`：显示菜单模块（未定义操作）
- `custom/go`：前往菜单模块（未定义操作）
- `custom/window`：窗口菜单模块，左键点击打开 Ghostty 终端

**作用：** 提供快速访问常用应用程序的菜单。

### 6. 键盘状态模块

```json
  "keyboard-state": {
    "numlock": true,
    "capslock": true,
    "format": {
        "numlock": "N {icon}",
        "capslock": "C {icon}"
    },
    "format-icons": {
        "locked": "",
        "unlocked": ""
    }
  },
```

**功能解释：**
- `numlock`：是否显示 Num Lock 状态（这里启用）
- `capslock`：是否显示 Caps Lock 状态（这里启用）
- `format`：状态的显示格式
- `format-icons`：锁定和解锁状态的图标

**作用：** 显示键盘状态（Num Lock 和 Caps Lock）。

### 7. 音乐模块

```json
  "mpd": {
      "format": "  {title} - {artist} {stateIcon} [{elapsedTime:%M:%S}/{totalTime:%M:%S}] {consumeIcon}{randomIcon}{repeatIcon}{singleIcon}[{songPosition}/{queueLength}] [{volume}%]",
      "format-disconnected": " Disconnected",
      "format-stopped": " {consumeIcon}{randomIcon}{repeatIcon}{singleIcon}Stopped",
      "unknown-tag": "N/A",
      "interval": 2,
      "consume-icons": {
        "on": " "
      },
      "random-icons": {
        "on": " "
      },
      "repeat-icons": {
        "on": " "
      },
      "single-icons": {
        "on": "1 "
      },
      "state-icons": {
        "paused": "",
        "playing": ""
      },
      "tooltip-format": "MPD (connected)",
      "tooltip-format-disconnected": "MPD (disconnected)",
      "on-click": "mpc toggle",
      "on-click-right": "foot -a ncmpcpp ncmpcpp",
      "on-scroll-up": "mpc volume +2",
      "on-scroll-down": "mpc volume -2"
    },
```

**功能解释：**
- `format`：音乐信息的显示格式（显示标题、艺术家、播放状态、时间、播放模式、音量）
- `format-disconnected`：与 MPD 服务器断开连接时的显示格式
- `format-stopped`：音乐停止时的显示格式
- `unknown-tag`：未知标签时显示的内容
- `interval`：更新间隔（这里是 2 秒）
- `consume-icons`：消费模式的图标
- `random-icons`：随机播放模式的图标
- `repeat-icons`：重复播放模式的图标
- `single-icons`：单曲循环模式的图标
- `state-icons`：播放状态的图标
- `tooltip-format`：工具提示的格式
- `on-click`：左键点击事件（播放/暂停）
- `on-click-right`：右键点击事件（打开 Ncmpcpp 音乐播放器）
- `on-scroll-up`：向上滚动事件（增加音量）
- `on-scroll-down`：向下滚动事件（减少音量）

**作用：** 显示和控制音乐播放。

### 8. 蓝牙模块

```json
  "bluetooth": {
	"format": " {status}",
	"format-connected": " {device_alias}",
	"format-connected-battery": " {device_alias} {device_battery_percentage}%",
	"tooltip-format": "{controller_alias}\t{controller_address}\n\n{num_connections} connected",
	"tooltip-format-connected": "{controller_alias}\t{controller_address}\n\n{num_connections} connected\n\n{device_enumerate}",
	"tooltip-format-enumerate-connected": "{device_alias}\t{device_address}",
	"tooltip-format-enumerate-connected-battery": "{device_alias}\t{device_address}\t{device_battery_percentage}%"
  },
```

**功能解释：**
- `format`：蓝牙状态的显示格式
- `format-connected`：蓝牙连接时的显示格式
- `format-connected-battery`：蓝牙连接且设备有电池时的显示格式
- `tooltip-format`：工具提示的格式
- `tooltip-format-connected`：蓝牙连接时的工具提示格式
- `tooltip-format-enumerate-connected`：枚举连接设备时的工具提示格式
- `tooltip-format-enumerate-connected-battery`：枚举连接设备且有电池时的工具提示格式

**作用：** 显示蓝牙状态和连接的设备。

### 9. 电池模块

```json
  "battery": {
    "format": "{icon} {capacity}%",
    "format-icons": ["󰂎", "󰁺", "󰁻", "󰁼", "󰁽", "󰁾", "󰁿", "󰂀", "󰂁", "󰂂", "󰁹"],
    "format-charging": "󰂄 {capacity}%",
    "format-plugged": "󰚥 {capacity}%",
    "format-full": "󰁹 {capacity}%",
    "states": {
      "warning": 20,
      "critical": 10
    },
    "interval":1,
    "tooltip": true,
    "class": "battery"
  },
```

**功能解释：**
- `format`：电池状态的显示格式
- `format-icons`：电池容量的图标（从低到高）
- `format-charging`：充电时的显示格式
- `format-plugged`：插入电源时的显示格式
- `format-full`：电池满时的显示格式
- `states`：电池状态阈值
- `interval`：更新间隔（这里是 1 秒）
- `tooltip`：是否显示工具提示（这里启用）
- `class`：模块的 CSS 类名

**作用：** 显示电池状态。

### 10. 网络模块

```json
  "network": {
    "format-wifi": "   {essid} ({signalStrength}%)",
    "format-ethernet": "   有线",
    "format-disconnected": "⚠   离线{ifname}",
    "tooltip-format": "{ifname}\n{essid}\n{signalStrength}% • {frequency} GHz\n{ipaddr}\n{bandwidthDownBits} ↓ • {bandwidthUpBits} ↑",
    "interval": 10,
    "on-click-right": "nm-connection-editor",
  },
```

**功能解释：**
- `format-wifi`：WiFi 连接时的显示格式
- `format-ethernet`：有线连接时的显示格式
- `format-disconnected`：网络断开时的显示格式
- `tooltip-format`：工具提示的格式（显示详细网络信息）
- `interval`：更新间隔（这里是 10 秒）
- `on-click-right`：右键点击事件（打开网络连接编辑器）

**作用：** 显示网络连接状态。

### 11. 时钟模块

```json
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
    "timezone": "Asia/Shanghai",
    "actions": {
        "on-click-right": "mode",
        "on-click-middle": "mode",
        "on-scroll-up": "shift_up",
        "on-scroll-down": "shift_down"
    },
    "class": "clock"
  },
```

**功能解释：**
- `format`：时钟的显示格式（这里显示时:分:秒）
- `format-alt`：时钟的备用显示格式（这里显示年-月-日 星期）
- `tooltip-format`：工具提示的格式（显示日历）
- `calendar`：日历配置
  - `mode`：日历模式（这里是 year，年度日历）
  - `mode-mon-col`：月度日历的列数（这里是 3 列）
  - `weeks-pos`：星期的位置（这里是 right，右侧）
  - `on-scroll`：滚动事件（这里是 1，表示滚动时切换月份）
  - `on-click-right`：右键点击事件（切换模式）
  - `format`：日历的样式（设置不同部分的颜色）
- `interval`：更新间隔（这里是 1 秒）
- `timezone`：时区（这里是 Asia/Shanghai）
- `actions`：操作配置
  - `on-click-right`：右键点击事件（切换模式）
  - `on-click-middle`：中键点击事件（切换模式）
  - `on-scroll-up`：向上滚动事件（上移）
  - `on-scroll-down`：向下滚动事件（下移）
- `class`：模块的 CSS 类名

**作用：** 显示时间和日历。

### 12. 托盘模块

```json
  "tray": {
    "icon-size": 17,
    "spacing": 6,
    "show-passive-items": true,
    "passive-icon-color": "#6e738d",
    "active-icon-color": "#cdd6f4",
    "icon-path": "/usr/share/icons/Papirus-Dark/24x24/panels/",
    "smooth-scrolling-threshold": 0,
    "on-click": "",
    "on-click-middle": "",
    "on-click-right": "",
    "tooltip": true,
    "tooltip-format": "{title}\n{class}",
    "class": "tray"
  }
```

**功能解释：**
- `icon-size`：托盘图标的大小（这里是 17 像素）
- `spacing`：托盘图标之间的间距（这里是 6 像素）
- `show-passive-items`：是否显示被动项（这里启用）
- `passive-icon-color`：被动项图标的颜色（这里是 #6e738d）
- `active-icon-color`：激活项图标的颜色（这里是 #cdd6f4）
- `icon-path`：图标的路径（这里是 Papirus-Dark 主题）
- `smooth-scrolling-threshold`：平滑滚动阈值（这里是 0）
- `on-click`：左键点击事件
- `on-click-middle`：中键点击事件
- `on-click-right`：右键点击事件
- `tooltip`：是否显示工具提示（这里启用）
- `tooltip-format`：工具提示的格式（显示标题和类名）
- `class`：模块的 CSS 类名

**作用：** 显示系统托盘图标。

### 总结

Waybar 提供了丰富的模块和高度可定制的配置选项，允许用户根据自己的需求创建一个完全符合个人偏好的状态栏。通过理解每个配置项的功能，您可以优化 Waybar 的布局和外观，提高工作效率。
