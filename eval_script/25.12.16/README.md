# 25.12.16 Quest 评测脚本

本目录用于 Quest 模型的评测，仿照 25.12.11 的结构。

## 文件说明
- `opencompass-eval-quest.py`：OpenCompass 配置，使用 `LlamaForCausalLM_Quest_OC`，默认 NeedleBench 32K，提供若干 Quest 参数组合。
- `eval-quest.sh`：运行脚本，激活 `oc` 环境后调用 `opencompass-eval-quest.py`，输出到 `oc-eval-result/quest`。
- `oc-eval-result/quest/`：评测结果输出目录（已创建空目录）。

## 使用示例
```bash
bash eval-quest.sh
# 或附加 OpenCompass 额外参数
bash eval-quest.sh --max-partition-size 1
```

如需修改模型路径或 Quest 配置，请编辑 `opencompass-eval-quest.py` 中的 `MODEL_PATH`、`MODEL_CONFIG_LIST`。
