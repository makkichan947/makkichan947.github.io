+++
date = '2025-09-30T11:42:47+08:00'
draft = false
title = '测试'
tags = ["杂项"]
comments = true
+++

# 一号标题

## 二号标题

### 三号标题

#### 四号标题

#### **代码块测试：**
```Rust
    // 获取日志统计信息
    pub fn get_logging_stats(&self) -> &LoggingStats {
        &self.stats
    }

    // 更新日志系统
    pub fn update(&mut self, delta_time: f32) -> GameResult<()> {
        // 刷新缓冲区
        if self.settings.enable_buffering {
            self.buffer.update(delta_time)?;
        }

        // 更新统计信息
        self.update_stats();

        Ok(())
    }
```

## **还有什么**
没了