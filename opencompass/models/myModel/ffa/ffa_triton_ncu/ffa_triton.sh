ncu_path="/remote-home1/zgliu/cudas/cuda-12.1/bin/ncu"
python_path="/remote-home1/zgliu/anaconda3/envs/ffa/bin/python"
script_path="/remote-home1/zgliu/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/ffa_triton_ncu/ffa_triton_v1017_ncu.py"

# 提取脚本目录和基本文件名
script_dir=$(dirname "$script_path")
script_name=$(basename "$script_path" .py)

# 在脚本所在目录生成同名报告
"$ncu_path" --set full --target-processes all -o "${script_dir}/${script_name}" "$python_path" "$script_path"