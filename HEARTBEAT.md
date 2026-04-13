# HEARTBEAT.md

## 🎾 黄龙室内场监控（每次心跳执行，本周周二到周日）

检查周二到周日黄龙室内场是否有连续2小时以上空档：

```
cd /Users/yingyao/.openclaw/workspace/tennis && python3 check_indoor.py
```

**处理逻辑：**
1. 输出包含 `有连续2小时以上空档` → 立即通知 Ying（有空档！）
2. 输出包含 `暂无连续2小时空档` → 检查 `/tmp/tennis_indoor_last_none.txt` 中上次通知时间戳
   - 距上次通知超过 **2小时**（或文件不存在）→ 发通知 + 更新时间戳
   - 否则 → 静默

**到期：** 周日(2026-04-19)后清空此任务
