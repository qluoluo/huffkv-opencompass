## Quest 与 Transformers 输出差异排查记录

场景：使用 Llama‑3.1‑8B 跑 `test_quest_vs_transformers.py`，即便给足 `token_budget`，Quest 输出仍与原生 Transformers 差距很大。

结论：Quest 的 RoPE 内核只支持统一的 `rope_scale/rope_theta`，而 Llama‑3.1 的 `rope_type=llama3` 采用非均匀缩放（高/低频不同）。原实现等价于错误的 RoPE，导致生成偏离。

修复：
- 在 `quest/models/QuestAttention.py` 增加 `rope_type` 识别；检测到 `llama3` 时使用 HuggingFace 的 `LlamaRotaryEmbedding` + `apply_rotary_pos_emb` 计算 RoPE，确保与官方实现一致。
- 其它 rope 类型（默认/linear）继续走 Quest 内核。

验证：`python test_quest_vs_transformers.py --token-budget 8192`，修复后 Quest 输出与 Transformers 对齐。

路径参考：`ffa-quest/quest/quest/models/QuestAttention.py`。
