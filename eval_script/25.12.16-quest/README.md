# 25.12.16 Quest 评测脚本

本目录用于 Quest 模型的 NeedleBench 与 RULER 评测，结构与 25.12.11 保持一致。

## 文件说明
- `opencompass-eval-quest.py`：NeedleBench 32K 配置，使用 `LlamaForCausalLM_Quest_OC`，含一组 Quest 参数与基线。
- `eval-quest.sh`：运行 NeedleBench 评测脚本，激活 `quest` 环境后输出至 `oc-eval-result/quest`。
- `opencompass-eval-ruler.py`：RULER 32K 配置，复用上方 Quest / 基线模型。
- `eval-ruler.sh`：运行 RULER 评测脚本，激活 `quest` 环境后输出至 `oc-eval-result/ruler`。
- `oc-eval-result/quest/`、`oc-eval-result/ruler/`：评测结果输出目录（运行脚本会自动创建）。

## 使用示例
```bash
# NeedleBench
bash eval-quest.sh
bash eval-quest.sh -r  # 追加 OpenCompass 运行参数

# RULER
bash eval-ruler.sh
bash eval-ruler.sh -r
```

如需修改模型路径或 Quest 参数，请编辑 `opencompass-eval-quest.py` / `opencompass-eval-ruler.py` 中的 `MODEL_PATH` 与 `MODEL_CONFIG_LIST`。
