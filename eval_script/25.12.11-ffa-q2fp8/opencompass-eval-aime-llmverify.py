import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import HF_ForCausalLM_FFA_OC
from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from opencompass.models import OpenAISDK

# Dataset config imports
with read_base():
    from ...opencompass.configs.datasets.aime2024.aime2024_llmverify_repeat8_gen_e8fcee import aime2024_datasets


# Global config
# MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3_2-3B"
MODEL_PATH = "/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B"

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

# LLM judge config (vLLM OpenAI API)
verifier_cfg = dict(
    abbr='qwen2.5-32B-Instruct',
    type=OpenAISDK,
    path='Qwen2.5-32B-Instruct',
    key='EMPTY',
    openai_api_base=[
        'http://127.0.0.1:8000/v1',
    ],
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
    ),
    query_per_second=16,
    batch_size=1024,
    temperature=0.001,
    max_out_len=16384,
    max_seq_len=49152,
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = verifier_cfg

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

summary_groups = [
    {
        'name': 'AIME2024-Average8',
        'subsets': [[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['AIME2024-Average8', 'naive_average'],
    ],
    summary_groups=summary_groups,
)
