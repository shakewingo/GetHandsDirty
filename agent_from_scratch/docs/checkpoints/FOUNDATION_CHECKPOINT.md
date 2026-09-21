# Foundation review checkpoint — 2026-09-13

历史记录：下文的“当前修订”指 9 月 13 日检查点，测试数量和工具协议也仅适用于该版本。
现行工具与 schema 见 [TOOLS_CHECKPOINT.md](TOOLS_CHECKPOINT.md)，阶段状态见
[STAGE.md](../STAGE.md)。

本次 review 修订相对于 c9b30c8，代码留在工作区供审阅。
上一版实验和设计保留在 git 历史与本地报告归档，本表只描述当前修订。

## 实现

- 删除 /read、运行时 completion_check、候选驳回逻辑和 verification.py。
- 普通输入不改写；模型返回工具调用则执行并继续，返回普通答案则结束。
- 成功/失败原始响应统一进 ModelRequest，随一个 run trace 保存，run schema=2。
  session schema=1；历史 check_failed 记录和旧 trace 可读，不生成该停止原因。
- 保留模板与解析修复、严格参数校验、分类恢复、重复失败停止、总迭代上限。
- read_file 返回范围说明及 next_offset，保留 8192-byte 默认上限和 UTF-8 边界。
  工具不强制读全；覆盖率只在 eval 结束后检查，不驱动模型。

82 个确定性测试通过；运行时各模块及 eval 的 Pyright 零错误。
运行时 1202 Python 行 / 11 文件（上次交付 1357，减少 155）；
测试 1277 行 / 6 文件；eval 223 行 / 1 文件。模板不计入 Python 行数。

## 真实模型 smoke：6 次完成

Qwen2.5-7B Q4_K_M，temperature=0，max_tokens=2048，n_ctx=8000，
每轮最多20次请求、连续相同失败最多3次，read默认及上限8192bytes。
使用与上一轮相同的9562-byte冻结 llm.py；所有测试在可写的临时工作区执行。
四个工具 schema 仍暴露：calculator/list_files/read_file/write_file。
所有运行均无解析错误或工具错误。只有 edit 改动了 config.json；目标源码始终未变。

| 用例 | 目标文件读取覆盖 | 模型请求 | 秒 | 观察 |
|---|---|---|---|---|
| exact_11 | 8192/9562 | 2 | 34.90 | 首块后结束，全文未完成 |
| full_11 | 8192/9562 | 2 | 16.30 | 首块后结束，全文未完成 |
| partial_11 | 1024/9562 | 2 | 19.23 | 局部读取通过 |
| edit_11 | 0/9562 | 4 | 7.76 | 目标配置正确且回读一致 |
| exact_22 | 8192/9562 | 2 | 34.93 | 首块后结束，全文未完成 |
| full_22 | 8192/9562 | 2 | 16.14 | 首块后结束，全文未完成 |

edit 的 llm.py 覆盖率不适用，其验收依据是 config.json 产物与回读。
exact 的提示只有 Read file <path>；full 明确要求读全并短摘要，未提示工具或 offset。
两组读取都未读全；不能因 final_response 或摘要相关便记为成功。
同一读取用例两次输出相同；temperature=0 的不同 seed 不是充分独立样本，
这些 smoke 不构成泛化成功率，也不证明新范围提示改善了模型能力。
partial 的读取正确、摘要相关，但回答额外复述源码，仍偏长。
语义审查来自本次 Codex 助手，不是独立人工评测。

## 明确边界

- 模型仍会过早结束或询问用户；不会被 runtime verifier 强制继续。
- NO_PROGRESS 只检测连续相同失败，不判断成功调用是否对任务有帮助。
- trace 在轮次结束/正常中断时保存，强制终止进程可能丢失当前轮。
- schema 是模型输入与执行前验证，不是 constrained decoding。
- 本轮未实现 web/shell、多调用或其他机制；交付后等待用户 review。

证据：outputs/foundation-review-20260913/ 中的 smoke、repeat、review.json、
size.json、tests.log；每组 metadata.json 记录设置、schema和源码/fixture hash。
复跑方式见 [EVAL_HISTORY.md](EVAL_HISTORY.md)。
