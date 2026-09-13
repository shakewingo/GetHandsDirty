# Foundation results — 2026-09-13

Implementation baseline: `2f4d32c`. Delivery runtime: `3ceb413`.
88 deterministic tests passed. Runtime: 1357 Python lines; tests: 1325; eval: 200.

Qwen2.5-7B Q4_K_M, llama-cpp-python 0.3.35, temperature 0, max_tokens 2048,
n_ctx 8000, at most 20 requests per turn. Successful raw responses, parse failures,
source/template/fixture hashes and settings are retained under the ignored
`outputs/foundation` directory. The 34 completed trials span several revisions;
they are not one final-version success-rate estimate. One in-flight duplicate
trial was cancelled and excluded.

| Condition | Observation |
|---|---|
| Fixed protocol, plain request, no verifier | Three trials read only the first 2048 of 9562 bytes, then ended. No parser errors. |
| Full-read check, rejected source excerpt kept in context | Read coverage passed, but repeated output truncation prevented final delivery. Stronger retry wording alone did not fix it. |
| Rejected candidate retained in raw trace, removed from active context | Three same-prompt checked trials read all bytes and delivered relevant summaries; one truncation recovery each. |
| Explicit full read and summary | Three trials completed in three model requests each. |
| First 1024 bytes only | Three trials stopped at the requested range, without requiring EOF. |
| Read config, change output, read back | Final JSON matched `{"output":"new.txt","retries":3}`; read-back matched the file. |
| Final `/read` wording on actual current llm.py | Three trials read 11097/11097 bytes and delivered short summaries. Requests: 3, 4, 4. Seconds: 30.35, 34.77, 34.26. No parse or tool errors. |
| File containing valid tool-call examples and invalid JSON | Read and summarized; no example was executed and no file was changed. |

Semantic judgments above are Codex assistant reviews of responses against source
and tool evidence, not independent human review. Programmatic read coverage is a
separate property. Seeds at temperature zero are not fully independent samples.

The final 2048-vs-8192 default comparison did not isolate a default-size benefit:
the model explicitly requested chunk_size=8192 in both conditions. Both made two
reads and four model requests. Do not attribute their 37.97-vs-34.17-second timing
difference to the default alone. Direct tool pagination with omitted sizes does
fall from five reads to two for the frozen 9562-byte file.

The final command explicitly asks for a short summary without reproducing the
file. Earlier actual-source trials sometimes stopped mid-quotation at a chat-token
literal despite finish_reason=stop. Tool JSON is now escaped at the Qwen rendering
boundary, token strings are obtained through the public tokenizer, and the final
command's delivery format was tested separately.

Plain natural language still has no automatic task-condition extraction. The
unverified full-read gate remains open; do not treat `final_response` as universal
task success. A coverage check also cannot prove summary accuracy. Web/shell and
Stage 3/4 are not marked ready by this checkpoint.

See [README.md](README.md) for usage and reproduction. Local detailed evidence:
`outputs/foundation/review.json`, `environment.json`, `experiment_notes.json`,
and each condition's `metadata.json`, `results.json`, and `state/runs` files.
Two early `p1` fixture-directory hash changes were harness preparation actions;
their side-effect statistics are excluded. The actual read target stayed unchanged.
