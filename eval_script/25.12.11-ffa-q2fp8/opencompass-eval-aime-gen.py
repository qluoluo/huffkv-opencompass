import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import HF_ForCausalLM_FFA_OC
from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM

# Dataset config imports
with read_base():
    from ...opencompass.configs.datasets.aime2024.aime2024_gen import aime2024_datasets


# Global config
# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B"
# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Qwen3-4B"

# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3-8B"

MAX_SEQ_LEN = 32 * 1024 
MAX_OUT_LEN = 64
BATCH_SIZE = 1
RUN_CFG = dict(num_gpus=1, num_procs=1)

# Default model args
DEFAULT_MODEL_KWARGS = dict(
    device_map='cuda',
    trust_remote_code=True,
    torch_dtype="float16",
    attn_implementation='flash_attention_2',
)

# Dataset config
datasets = []
datasets += aime2024_datasets

# Model quantization config
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



# Build model list
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

# Common model settings
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

# Inference config
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        retry=1
    ),
)

# Eval config
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=160,
        task=dict(type=OpenICLEvalTask)
    ),
)
