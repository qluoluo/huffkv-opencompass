# 25.12.19 Twilight 评测脚本

本目录用于 Twilight（Quest + Top-p Pruning）模型的评测，结构与 25.12.16 的 Quest 目录一致。

## 文件说明
- `opencompass-eval-twilight.py`：OpenCompass 配置，使用 `LlamaForCausalLM_Twilight_OC`，默认 NeedleBench 32K，提供若干 Twilight 参数组合。
- `eval-twilight.sh`：运行脚本，激活 `twi` 环境后调用 `opencompass-eval-twilight.py`，输出到 `oc-eval-result/twilight`。
- `oc-eval-result/twilight/`：评测结果输出目录（已创建空目录）。

## 使用示例
```bash
bash eval-twilight.sh
# 或附加 OpenCompass 额外参数
bash eval-twilight.sh -r
```

如需修改模型路径或 Twilight 配置，请编辑 `opencompass-eval-twilight.py` 中的 `MODEL_PATH`、`MODEL_CONFIG_LIST`，或在运行脚本时覆盖 OpenCompass 参数。
