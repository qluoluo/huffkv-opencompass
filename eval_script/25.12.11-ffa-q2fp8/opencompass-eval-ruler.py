import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import HF_ForCausalLM_FFA_OC
from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM

# 导入数据集和汇总器配置
with read_base():
    from ...opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets
    # from ...opencompass.configs.summarizers.ruler import (
    #     ruler_32k_summarizer as summarizer,
    # )


# 全局配置
# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B"
MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3-8B"

MAX_SEQ_LEN = 32 * 1024
MAX_OUT_LEN = 64
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
datasets = []
datasets += ruler_datasets

# 模型量化配置
MODEL_CONFIG_LIST = [
    {"abbr": "ffa-q2fp8-delta5", 
        "use_ffa_decode": True,
        "delta": 5.0,
        "ffa_decode_kernel": "q2fp8",
        "use_fp8_residual": True,
    },
    # {"abbr": "ffa-q2fp8-delta3", 
    #     "use_ffa_decode": True,
    #     "delta": 3.0,
    #     "ffa_decode_kernel": "q2fp8",
    #     "use_fp8_residual": True,
    # },
    {"abbr": "ffa-q2fp8-delta10", 
        "use_ffa_decode": True,
        "delta": 10.0,
        "ffa_decode_kernel": "q2fp8",
        "use_fp8_residual": True,
    },
]



# 构建模型列表
models = []
for model_config in MODEL_CONFIG_LIST:
    abbr = model_config.pop('abbr')
    model_kwargs = model_config
    
    models.append(dict(
        type=HF_ForCausalLM_FFA_OC,
        # abbr=config["abbr"],
        abbr=abbr,
        path=MODEL_PATH,
        model_kwargs=model_kwargs,
    ))
    
models.append(dict(
    type=HuggingFaceCausalLM,
    abbr="base",
    path=MODEL_PATH,
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
