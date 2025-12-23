# 25.12.11 FFA q2fp8 NeedleBench & RULER 评测脚本

本目录包含针测（NeedleBench NIAH）与 RULER 的 OpenCompass 评测配置（q2fp8 FFA decode）。

## 文件说明
- `opencompass-eval-niah.py`：NeedleBench 32K 配置，包含多组 q2fp8 FFA decode 变体与基线。
- `eval-niah.sh`：激活 `oc` 环境后运行 NIAH 评测，输出到 `oc-eval-result/niah`。
- `opencompass-eval-ruler.py`：RULER 32K 配置，复用 Llama-3.1-8B，默认多组 q2fp8 FFA decode 及基线。
- `eval-ruler.sh`：激活 `oc` 环境后运行 RULER 评测，输出到 `oc-eval-result/ruler`。
- `oc-eval-result/niah/`、`oc-eval-result/ruler/`：评测输出目录（运行脚本会自动创建）。

## 使用示例
```bash
# NIAH
bash eval-niah.sh
bash eval-niah.sh -r  # 追加 OpenCompass 运行参数

# RULER
bash eval-ruler.sh
bash eval-ruler.sh -r
```

## 配置调整
- 修改 `opencompass-eval-*.py` 中的 `MODEL_PATH` 以切换模型。
- 通过各脚本内的 `MODEL_CONFIG_LIST` 增删 FFA 配置或调节参数。
