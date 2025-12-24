import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import HF_ForCausalLM_FFA_OC as HF_ForCausalLM_FFA_OC

# 导入数据集和汇总器配置
with read_base():
    from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_origin_en_datasets

# 全局配置
MODEL_PATH = "/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B"
# MODEL_PATH = "/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B"
MAX_SEQ_LEN = 32 * 1024
MAX_OUT_LEN = 50
BATCH_SIZE = 1
RUN_CFG = dict(num_gpus=1, num_procs=1)

# 默认模型参数
DEFAULT_MODEL_KWARGS = dict(
    device_map='cuda',
    trust_remote_code=True,
    torch_dtype="float16",
    attn_implementation='flash_attention_2',
)

# 数据集配置
datasets = needlebench_origin_en_datasets

# 模型量化配置
QUANT_CONFIGS = [
    {"abbr": "base", "use_ffa": False,},
    {"abbr": "ffa-bs256-sbs256-delta5", 
        "use_ffa": True,
        "BS": 256,
        "SBS": 256,
        "delta": 5.0,
    },
    {"abbr": "ffa-bs512-sbs256-delta5", 
        "use_ffa": True,
        "BS": 256,
        "SBS": 256,
        "delta": 5.0,
    },
    {"abbr": "ffa-bs256-sbs256-delta10", 
        "use_ffa": True,
        "BS": 256,
        "SBS": 256,
        "delta": 10.0,
    },
    {"abbr": "ffa-bs256-sbs256-delta3", 
        "use_ffa": True,
        "BS": 256,
        "SBS": 256,
        "delta": 3.0,
    },
]

# 通用量化参数
COMMON_QUANT_KWARGS = {

}

# 构建模型列表
models = []
for config in QUANT_CONFIGS:
    abbr = config.pop('abbr')
    model_kwargs = COMMON_QUANT_KWARGS | config
    
    models.append(dict(
        type=HF_ForCausalLM_FFA_OC,
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