import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import LlamaForCausalLM_TaylorKV_OC as LlamaForCausalLM_OC

# 导入数据集和汇总器配置
with read_base():
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_32k_summarizer as summarizer

# 全局配置
# MODEL_PATH = "/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B"
MODEL_PATH = "/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B"
MAX_SEQ_LEN = 32 * 1024
MAX_OUT_LEN = 50
BATCH_SIZE = 1
RUN_CFG = dict(num_gpus=1, num_procs=1)

# 默认模型参数
DEFAULT_MODEL_KWARGS = dict(
    device_map='cuda',
    trust_remote_code=True,
    torch_dtype="bfloat16",
    attn_implementation='flash_attention_2',
)

# 数据集配置
datasets = needlebench_origin_en_datasets

# 模型量化配置
QUANT_CONFIGS = [
    {"abbr": "CrucialKV-w16-sn512", "use_remain": False, "window_size": 16, "sparse_num": 512,},
    {"abbr": "CrucialKV-w8-sn512", "use_remain": False, "window_size": 8, "sparse_num": 512,},
    {"abbr": "CrucialKV-w6-sn512", "use_remain": False, "window_size": 6, "sparse_num": 512,},
    {"abbr": "CrucialKV-w4-sn512", "use_remain": False, "window_size": 4, "sparse_num": 512,},
    {"abbr": "CrucialKV-w2-sn512", "use_remain": False, "window_size": 2, "sparse_num": 512,},
    {"abbr": "CrucialKV-w1-sn512", "use_remain": False, "window_size": 1, "sparse_num": 512,},
    {"abbr": "CrucialKV-w128-sn512", "use_remain": False, "window_size": 128, "sparse_num": 512,},

    {"abbr": "CrucialKV-w8-sn256", "use_remain": False, "window_size": 8, "sparse_num": 256,},
    {"abbr": "CrucialKV-w256-sn256", "use_remain": False, "window_size": 256, "sparse_num": 256,},
    {"abbr": "CrucialKV-w128-sn384", "use_remain": False, "window_size": 128, "sparse_num": 384,},

]

# 通用量化参数
COMMON_QUANT_KWARGS = {
    
    "debug": True,
}

# 构建模型列表
models = []
for config in QUANT_CONFIGS:
    abbr = config.pop('abbr')
    # model_kwargs = {
    #     "k_bits": config["k_bits"],
    #     "k_quant_dim": config["k_quant_dim"],
    #     "v_bits": config["v_bits"],
    #     "v_quant_dim": config["v_quant_dim"],
    #     **COMMON_QUANT_KWARGS
    # }
    model_kwargs = COMMON_QUANT_KWARGS | config
    
    models.append(dict(
        type=LlamaForCausalLM_OC,
        # abbr=config["abbr"],
        abbr=abbr,
        path=MODEL_PATH,
        model_kwargs=model_kwargs,
    ))

# 为所有模型添加通用配置
for model in models:
    model.update(dict(
        model_kwargs=DEFAULT_MODEL_KWARGS | model.get('model_kwargs', {}),
        tokenizer_path=MODEL_PATH,
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True
        ),
        max_seq_len=MAX_SEQ_LEN,
        max_out_len=MAX_OUT_LEN,
        run_cfg=RUN_CFG,
        batch_size=BATCH_SIZE,
    ))

# 推理配置
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        retry=1
    ),
)

# 评估配置
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=160,
        task=dict(type=OpenICLEvalTask)
    ),
)