# Stage 05 — matched empirical pilot / 实验结果与采用建议

本轮完成了可独立开关的 harness 实现与验证，但**尚未证明 tiny agent 的整体正确率和效率得到可靠提升**。建议保留研究分支，所有生产默认开关继续关闭。优先继续研究工具调用与编辑可靠性、elision 后的证据恢复，以及跨成功/失败调用的循环检测。

## 实验边界与可复查证据

- 运行代码冻结于 `42c7228`；继承 `feat/agent-foundation` 的 `d7f3887`。
- Qwen2.5-7B / 本地 Q4 GGUF；模型配置、seed、源码哈希和逐题初始文件哈希一致。
- 请求 N_CTX=8,000，后端实际窗口 8,192；输出预留 2,048；每题上限 20 次模型调用、40 次工具执行尝试。完整参数见 [measurements](stage-05-measurements.json)。
- 四组各 8 题，加三个定向子集，共 **41 次正式模型任务**。顺序执行，没有并行模型进程。总计 1,343.44 秒 turn 时间、564,204 tokens，包含失败任务；不含模型加载。
- 另有原有受限工具回归集 **10/17**，记录在 [Stage 00](STAGE-00.md)，与这里的新任务不可直接比较。中止的 v1 pilot 不进入任何正式分数。
- 单元/集成测试在冻结版本上 **229 项通过**，日志 `outputs/empirical-study/stage-04-final-tests.log`。最终再次验证 **229 项通过（3.954 秒）**，日志 `outputs/empirical-study/stage-05-final-tests.log`。测试通过说明实现约束成立，不代表模型能力提升。
- 原始 evidence：`outputs/empirical-study/stage-05/<profile>/<task>/`，包含 prompt/history、运行 trace、前后文件哈希、最终 workspace 和独立评分。
- [analyze.py](analyze.py) 验证同一源码/模型/seed/fixture、完整任务列表，并导出下述逐题对照；不修改运行数据。

## 完整八题对照

“严格成功”要求最终答案、文件结果、来源读取顺序、无额外修改等相关检查全部通过；不能直接解读为语义回答正确率。

| Profile | 严格成功 | 总 tokens | 总秒数 | 模型调用 | 秒/成功题 | tokens/成功题 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1/8 | 72,441 | 156.77 | 28 | 156.77 | 72,441 |
| Elision | 1/8 | 73,674 | 173.73 | 27 | 173.73 | 73,674 |
| Planning | 1/8 | 87,948 | 186.74 | 29 | 186.74 | 87,948 |
| Elision + planning | 2/8 | 118,516 | 228.20 | 39 | 114.10 | 59,258 |

每成功题成本 = 全部题目成本 / 成功题数，包含失败开销。组合组的比值下降由单个额外成功驱动，样本不足以作为部署依据。Baseline 两个长历史任务在生成前即被阻止，几乎不消耗 tokens；因此它的低成本部分来自未执行任务，不能视为更高效地完成了相同工作。

| 任务 | Baseline | Elision | Planning | Elision + planning |
|---|---|---|---|---|
| direct | 通过 | 通过 | 通过 | 通过 |
| no_op | 多余修改尝试 | 多余修改尝试 | 修改尝试 + 答案格式 | 多余修改尝试 |
| nested | 来源/目标文件错误 | 来源读取失败 | 来源/目标错误 + 额外修改 | 来源/目标错误 + 额外修改 |
| search | 值正确，答案格式失败 | 同左 | 同左 | 同左 |
| json_repair | 文件内容不正确 | 文件内容不正确 | 文件内容不正确 | 通过 |
| missing_path | 恢复成功，答案格式失败 | 同左 | 同左 | 同左 |
| history_retain | context_limit | 执行了，但答案错 | context_limit | 执行了，但答案错 |
| history_edit | context_limit | 执行了，但编辑失败 | context_limit | 执行了，但编辑失败 |

具体 failed_checks、最终答案、修改路径和每题成本均保存在 measurements 中；表格只压缩描述主要问题。

## 单因素定向实验

每组只与相同任务子集的 baseline 比较，不能把这些分数与上面的 8 题分数横向相加。

| Profile / 子集 | Baseline → candidate 严格成功 | 总 tokens | 总秒数 | 实际触发 |
|---|---|---|---|---|
| Search / direct,nested,search,missing_path | 1/4 → 1/4 | 60,057 → 92,301 | 120.12 → 339.95 | 1 次 search_files |
| Repeat / search,missing_path | 0/2 → 0/2 | 35,824 → 36,392 | 68.23 → 72.65 | 0 次提醒 |
| Diagnostics / no_op,nested,json_repair | 0/3 → 1/3 | 35,065 → 82,932 | 82.25 → 185.40 | 1 次检查，结果为语法有效 |

Search 组每成功题成本为 92,301 tokens / 339.95 秒；Diagnostics 为 82,932 / 185.40；Repeat 零成功，分母为零，报告 null 而非 0。完整匹配成本见 JSON。

## 可指导后续 harness 的发现

1. **Elision 改善 context admission，尚未改善证据使用。** 两个 history 任务从生成前超窗变成可以执行；elision 组和组合组各省略了 9 个旧 observation。Baseline 的 summary 输入约 7,190 tokens，再加 2,048 输出预留超窗；actor 输入约 8,544。Elision 后首个 actor 输入约 3,123。可是模型把空字符串替代的旧内容理解成文件为空/没有信息，或仍未取回正确事实。下一轮应对比显式非空省略标记、有限预览与 reread 提示；这是本地推论，不是已经验证的修复。
2. **本轮没有证明 elision 能降低 summarization 成本。** 四个八题核心组均没有真实 summary 生成。Baseline/planning 的两个 history 任务各触发一次 summary 阻止和 actor 阻止；不是“完成了摘要但效果不好”。需要补充中等上下文压力、可实际完成 summary 的任务。
3. **可用的计划工具不等于被使用的计划。** Planning 组只有 1 次成功 update_plan；组合组为 0。组合组额外通过 json_repair，但这道题既未更新计划，也未触发 elision，不能宣称两者协同。当前实现也没有复刻论文单独的初始 no-plan reminder，不能据此否定其他 planning 设计。
4. **结构化 search 有局部效率信号，但没有带来整体收益。** search 单题从 9 次模型调用、28,069 tokens、53.54 秒降至 2 次、3,810 tokens、9.50 秒，约减少 86% tokens / 82% 时间；两次都找到正确值，但都违反“只返回值”的答案约束。与此同时该 profile 的 nested 题达到 20 次模型调用上限，耗时 307.43 秒，包含 3 次真实 summary。该题没有调用 search，因此只能报告工具可用/schema 条件下的轨迹变化，不能把慢循环归因于 search 执行。
5. **当前重复检测覆盖不到交替循环。** Search/nested 轨迹重复“成功读文件 → 错误编辑 → 跳过后续调用”。成功读取重置失败计数，失败编辑又打断连续相同成功读取计数。Repeat 定向组没有触发提醒，所以它只得到实现测试验证，模型收益仍未知。后续应研究跨多个调用的重复状态/无进展检测，保留预算作为兜底。
6. **Diagnostics 的额外成功不等于纠错证据。** json_repair 从失败变为通过（8,309 → 6,495 tokens；21.00 → 22.26 秒），但唯一诊断发生在正确写入之后，返回 valid。没有“诊断报错 → 模型修复”的实际链路。该组 nested 同样耗尽 20 次模型调用，包含 1 次 summary。
7. **当前 tiny model 的明显瓶颈是工具语义与精确编辑。** 多个轨迹包含把显示行号当成文件内容、转义/old_text 不匹配、版本错误、改错 manifest 而非它引用的配置；JSON 修复有时写成 Python literal。下一轮可分别测试更清楚的 read/edit 输出协议、结构化 JSON 修改工具，避免直接增加更多提示和工具后无法归因。

## 评分边界：no-op 与格式

复核发现 no_op 的提示“如果已经正确，不要写入任何文件”和 scorer 的“禁止任何 write/edit 尝试”存在结果与过程约束的歧义。**冻结分数未修改**。

最终五组 no_op 文件前后哈希都一致；其中四组答案精确为 DONE，planning 组附加了解释。所有组都尝试了修改，实际返回错误/跳过，没有成功的 `changed=False` 写入。因此这是“最终文件状态正确但调用过程有问题”，不应描述成文件被破坏。

提供一致的事后保守敏感性检查：除放宽 no_write_attempts/allowed_changes 两项外，保留其他检查，同时要求前后快照一致，且每个修改结果都明确 `ok=True, changed=False`。错误/跳过不满足这一证据条件。本批次该检查没有改变任何分数。细节在 JSON 的 `no_op_post_hoc_audit`。后续协议应明确写“不要调用 write_file 或 edit_file”，并另报最终文件状态指标。

search/missing_path 多数严格失败只来自带有解释的最终文本；所需来源证据和恢复顺序正确。报告保留这一差别，不把格式失败称为找不到答案，也不在看到结果后放宽答案标准。

## 采用决定与下一轮顺序

| 项目 | 本轮决定 | 下一轮需要的证据 |
|---|---|---|
| 独立评分、trace、固定 manifest、匹配分析 | 保留为研究基础设施 | 增加更明确的 no-op 契约与结果/过程分离指标 |
| Elision | 保持 opt-in | 更可靠的省略表示与 reread；中等压力 summary 对照 |
| Planning | 保持 opt-in | 先提高实际 plan 使用覆盖，再测净收益与提示成本 |
| Search | 保留显式注册能力，不默认加入 | 多个检索任务复现局部收益，并检查 schema 对其他任务影响 |
| Repeat | 保持 opt-in | 能实际触发的模型任务；跨调用循环检测的独立消融 |
| Diagnostics | 保持 opt-in | 真实错误诊断驱动修复的轨迹与净成本 |

建议下一轮先冻结额外任务与评分规则，再分别研究：① read/edit/JSON 工具协议；② elision 的证据恢复；③ 循环检测。每项用额外题目和重复运行确认；本轮的负例可用于开发，但不能再充当独立 held-out 证据。

限制：8 道手工开发题、每条件一次 greedy run、单个小模型、固定 profile 执行顺序、没有统计区间或外部 coding benchmark。随机 run IDs、临时路径、文件版本/时间戳使 prompt 并非逐字一致；seed 相同不等于完全可复现。两个 history 任务的前史为 fixture 提供，不是模型自主完成。Search/nested 期间进行过一次一秒 OS 采样，延迟包含这一诊断开销。不报告未人工逐条审查的 false-completion rate，也不外推为论文结论的复现。

最终独立报告审查重新计算了总数、匹配分母、成本和敏感性分析，未发现实质问题。全部 66 个源码 manifest 条目仍与实验时一致；原有未跟踪 memory.py 未修改。
