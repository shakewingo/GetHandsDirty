# Stage 06 — 合并 foundation compact 增强并复核 harness

已将远端 `origin/feat/agent-foundation` 的 **54e863d** 合并到
`empirical-study/harness-design`。合并提交 **90e1f79** 的两个父提交为原研究分支
`f12bec2` 和远端 `54e863d`；没有切换工作分支或推送远端。

[执行计划](STAGE-06-PLAN.md)记录本轮范围。原 Stage 05 的代码、任务结果和结论保留为历史证据。

## 上游实际增加了什么

本次取得共同祖先 `d7f3887` 之后的 11 个提交，其中 8 个为运行时/测试变更，3 个为文档变更。

| 上游提交 | 增强 | 对准确率/效率的意义与边界 |
|---|---|---|
| `a316e64` | `summary_max_tokens=512`，摘要独立设置测量和生成预算；actor 仍预留 2,048 | 减少摘要请求预留的空间，让此前“actor 超窗、摘要也进不去”的部分输入可以摘要；过小预算仍可能截断重要事实或输出 |
| `f2792a0` | 一次 compact 最多 2 次摘要请求；第二次附加拒绝原因，要求更短；每 turn 最多 4 次摘要 | 提供有界纠错机会；每次实际请求都计入 20 次总调用预算，保留 actor 的最后一个调用位置；摘要输入自身超窗时不重试 |
| `2d465a3` | parser 真正执行 `max_tool_calls_per_response` | 参数从仅记录变为执行约束；超额 batch 整体拒绝，不执行部分工具 |
| `4b72932` | compact 发布时重读规则；候选视图包含新规则并重新测量 | 减少长 turn 使用过期规则的风险；加载失败保留已有规则；规则增长导致无法缩小时拒绝候选；原始 trace 不改写 |
| `3b9ee31` | 增加 compact 跨边界连续性的回归测试 | 检查工具效果不重复、错误/调用计数不断档、请求和新证据保留；测试不等于模型执行成功 |
| `b6a393f` | 先保存完成 turn 的 raw delta，再保存 schema 2 summary checkpoint | checkpoint 带 covered 边界、raw 前缀摘要哈希、摘要配置与时间；不完整 turn 不成为可重放历史 |
| `09d5d69` | 重启加载 checkpoint，模型视图使用 summary + 未覆盖 suffix | 降低跨进程重复注入旧历史的成本；原始历史仍保留，前缀 digest/边界不匹配回退 raw；不是逐工具中断恢复，也不保证 summary 语义正确 |
| `6acbbb1` | actor/compact 次数、成功发布次数、actor 峰值 prompt；兼容旧 `context` 与新 `budget` 字段 | 更准确区分“摘要请求发生”“候选发布”“任务成功”，方便成本归因 |

`2615c21`、`4b8b1d3`、`54e863d` 是设计边界/阶段状态/实施计划文档。
checkpoint 的 config 当前用于来源记录，加载时没有根据当前模型/提示配置自动判定过期；不能将其描述成完整的语义或配置有效性验证。规划状态仍是当前用户 turn 的临时状态，checkpoint 不是长期 task-plan memory。

## 合并策略与实际修复

五个冲突文件为 `agent.py`、`compact.py`、`config.py`、`context.py`、`trace.py`。
采用组合双方语义的方式：保留上游预算/重试/规则/checkpoint，也保留本分支的
elision、planning、重复提醒、语法诊断和对应 trace 字段。

- `ContextState.messages()` 先使用当前规则与 summary/suffix 构建独立视图，再应用 elision、注入一次最新 plan。
- `Compactor._publish()` 的候选同时携带 `instructions`、`elided`、`plan_text`，因此 fit 检查包含实际将发送的全部内容。失败不发布任何部分状态。
- **修复 planning 协议丢失：** 原分支仅在 turn 启动时追加 `PLAN_RULES`，而上游 compact reload 直接读取文件规则。新增测试证明 planning=True 时协议在 compact 后消失。现由 `Agent._load_instructions()` 统一组装两条路径；planning=False 时不注入。
- **修复历史评估导出：** 上游聚合函数直接索引新增 compact 指标，使旧 `results.json` 在 `--review` 时 KeyError。现在旧/混合记录缺失计数保持 `null`，现代记录正常求和。CLI 回归测试验证旧结果可导出、原记录不改写、没有模型调用。

验证：文本合并后原有 **251 项通过**；新增交互测试先复现 protocol 丢失，修复后 **253 项通过**；独立审查发现历史结果兼容问题，两项测试先复现失败，修复后完整 **255 项通过（3.979 秒）**。独立审查另运行 70 项相关测试，未发现其他实质合并问题。

日志位于 `outputs/empirical-study/stage-06/`：`merge-tests.log`、
`interaction-red.log`、`interaction-green.log`、`legacy-review-red.log`、`final-tests.log`。

## 同一合并版本上的真实模型结果

运行代码冻结于 `90e1f79`。Qwen2.5-7B Q4、8,192 实际窗口、actor 输出预留 2,048、summary 预留 512、seed 11。共 8 次任务、29 次实际模型调用，总计 **122,828 tokens / 356.91 秒**（含失败，模型加载不计入 turn 时间）。没有并行模型进程或中途改源码。

| Profile | 严格成功 | 总 tokens | 秒数 | 模型调用 | 摘要生成/成功发布 | 成功更新计划 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1/2 | 35,150 | 116.06 | 8 | 2/2 | 0 |
| elision | 0/2 | 23,650 | 49.26 | 6 | 0/0 | 0 |
| planning | 0/2 | 42,069 | 146.65 | 9 | 2/2 | 0 |
| elision_planning | 0/2 | 21,959 | 44.94 | 6 | 0/0 | 0 |

| 任务 | Baseline | Elision | Planning | Elision + planning |
|---|---|---|---|---|
| history_retain | 正确值 MAPLE-41，格式失败 | 错把 entry_000 当成值 | 正确值 MAPLE-41，格式失败 | 错把 entry_000 当成值 |
| history_edit | **通过全部检查** | 版本不匹配，未写入 | old_text 三次不匹配，no_progress | 计划参数反复无效，no_progress |

**对旧结论的更新：** Stage 05 相同两题的 baseline 是 0/2，均在模型生成前 context_limit；合并后是 1/2，两题都能执行。摘要请求的实测输入为 7,190 / 7,193 tokens，加上 512 输出预算后剩余 490 / 487，满足 256 margin；旧 2,048 预算不满足。成功摘要后的首个 actor 输入为 2,134 / 2,147 tokens。这里有直接的 admission 改善证据。

但摘要本身没有保留 MAPLE-41 的具体值：baseline 随后重新读取 archive0.txt 才找到它，history_edit 再读取 config.json 并正确写入。应把结果理解为“摘要让 agent 能继续执行并恢复证据”，不是“摘要已无损保留事实”。history_retain 因附带解释而未满足只返回值的要求，不能称为语义值错误，也不事后放宽分数。

Elision 省掉了摘要调用；elision / 组合组分别省略 9 / 8 个 observation。但在这两题上，更低成本伴随零严格成功。两个 retain 轨迹仅重读第 1 行（`limit=1`，返回 `truncated=True, next_offset=2`），未继续读取就把 entry_000 当成目标值。需要改进的是省略后的读取与证据使用，不能仅凭 context 不超窗宣布提升。

Planning 组没有调用 update_plan；组合组则有 **4 次 update_plan 尝试全部被拒绝**：模型将全部步骤设为 pending，违反“恰有一个 in_progress”的约束。这是工具协议可执行性的新负例，不能将“成功计划更新为零”描述成“模型完全没尝试规划”。下一轮宜单独对比更易初始化的计划接口/更明确的示例，不把本轮失败泛化为 planning 无效。

本轮没有 summary corrective retry 触发；四次摘要首次均成功发布。重试上限得到确定性测试验证，但模型收益未测出。新 baseline 每成功题成本是 35,150 tokens / 116.06 秒（含失败题成本）；其他组没有严格成功，这个比值为 null。

逐题最终答案、failed_checks、工具错误、精确预算与新旧对照见 [机器可读结果](stage-06-measurements.json)。可用 [compare_compact.py](compare_compact.py) 从保留的原始记录重新生成；脚本检查新组之间源码/模型/seed 一致，跨版本任务提示、旧参数与初始文件一致。没有覆盖原 Stage 05 结果。

## 与论文经验的对应关系

论文 [§2.1、§2.3、Algorithm 1 与 §3](https://arxiv.org/pdf/2609.20804) 给出的经验是：组件独立消融，先用较便宜的 elision，再按压力触发摘要；保留任务/规则与近期证据；planning 的收益依赖模型能力和任务，不能只看轨迹更长。小模型依赖工具接口的可靠性，额外机制必须计入净成本。

据此，本轮保留分阶段上下文管理的方向，但不把论文的大模型/32k–128k 结果直接迁移成 Qwen2.5-7B/8k 的有效性结论。上游 checkpoint 是会话持久化增强，不是论文 recall_event 的等价实现，也不恢复被模型遗漏的事实。

| 本分支机制 | 合并后的适用性 | 仍需优化/验证 |
|---|---|---|
| Elision before summary | 仍可在不调用模型的情况下减小 actor 输入，与新摘要预算互补 | 旧实验把空 body 当成空文件；需要非空省略标记/有限预览的独立实验。新预算可能让 baseline 也能摘要，本轮已证明 baseline 也能进入生成，旧 admission 优势不再成立 |
| Plan state + update_plan | 保留；已修复 reload 协议丢失，候选预算计入 plan | 新组合组 4 次 all-pending 更新失败；应先改善初始化协议可执行性，再测规划净收益，不默认强制每题规划 |
| Structured search | 与 compact 基本独立，仍可显式注册 | 保留局部检索效率信号，但增加 schema 可能改变其他轨迹；需多题验证，不能借合并直接默认开启 |
| Repeated-success reminder | 原行为保留 | 上游 summary retry 只处理摘要失败，不解决 actor “成功读取—失败编辑”的交替循环；仍需跨多个调用的无进展检测实验 |
| Post-edit diagnostics | 原行为保留 | 语法有效不等于任务正确；需看到真实的诊断错误驱动修复链路 |
| Independent eval/trace | 继续使用；吸收上游新 compact 指标并兼容历史数据 | 分开记录实际生成、发布、重试、超窗和独立任务结果；历史 no-op 契约歧义不在本轮改规则 |

优先级：先让 read/edit/JSON 操作准确，验证摘要是否保留任务所需事实，再优化 elision 的证据恢复和跨调用循环检测。增加机制后仍要比较全部成本；不能把少生成但未完成任务称为效率提升。

## 本轮验证的边界与下一步

- 新模型 panel 只包含原 suite-v2 的 `history_retain`、`history_edit` 两题，四个 profile 共 8 次运行。模型、seed、任务提示、评分规则和初始文件保持一致；临时路径、版本号与 run ID 仍不同。没有重跑整个 8 题或 17 题基准。
- 同一新版本中的 profile 对照可用于观察机制行为；跨 Stage 05/06 的对照同时改变多项上游机制，不能把全部差异归因于某一个参数。
- 历史任务使用 fixture 提供的前史；它们验证 compact admission、事实检索和文件操作，不验证模型自主完成此前轨迹。没有将这组实验当成 checkpoint 跨进程收益的证据，也没有宣布上游 Stage 3 的所有真实模型验证缺口已关闭。
- 新摘要预算仍无法容纳任意大的 raw summary 输入；必要时应研究结构安全的分块/边界选择，而不是无限重试。先增加明确触发这一边界的固定任务。
- 下一轮应增加“原始历史和 compact 后视图都能放进窗口”的匹配任务，直接检查丢失了哪些关键事实；再对省略标记、局部预览、分页读取/精确编辑协议分别做消融。所有失败成本都进入统计，保持独立结果验证。

本轮不改实验功能的默认开关，不引入新 recall 工具、强制规划或更复杂循环检测。研究分支保留供 review；原有未跟踪 `memory.py` 不纳入提交。
