import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from opencompass.models import LlamaForCausalLM_Quest_OC as LlamaForCausalLM_OC

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
)

# 数据集配置
datasets = []
datasets += ruler_datasets

# 模型量化配置
MODEL_CONFIG_LIST = [
    {
        "abbr": "quest-ps16-b256",
        "quest_page_size": 16,
        "quest_token_budget": 256,
        "quest_max_seq_len": 32 * 1024,
    },
    
    {
        "abbr": "quest-ps16-b2048",
        "quest_page_size": 16,
        "quest_token_budget": 2048,
        "quest_max_seq_len": 32 * 1024,
    },
    
    {
        "abbr": "quest-ps16-b4096",
        "quest_page_size": 16,
        "quest_token_budget": 4096,
        "quest_max_seq_len": 32 * 1024,
    },
]

MODEL_CONFIG_LIST.append(
    {"abbr": "base",}
)

# 构建模型列表
models = []
for model_config in MODEL_CONFIG_LIST:
    abbr = model_config.pop('abbr')
    model_kwargs = model_config
    
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