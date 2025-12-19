import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import LlamaForCausalLM_FFA_OC as LlamaForCausalLM_FFA_OC

# 导入数据集和汇总器配置
with read_base():
    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_origin_en_datasets
    # from opencompass.configs.summarizers.needlebench import needlebench_32k_summarizer as summarizer
    
    from ...opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_origin_en_datasets
    from ...opencompass.configs.summarizers.needlebench import needlebench_32k_summarizer as summarizer
    
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

# 全局配置
MODEL_PATH = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/models/Llama-3_2-3B"

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
datasets = []
datasets += needlebench_origin_en_datasets
# datasets += longbench_datasets

# 模型量化配置
MODEL_CONFIG_LIST = [
    {"abbr": "ffa-prefill-delta5", 
        "use_ffa_prefill": True,
        "delta": 5.0,
    },
    {"abbr": "ffa-prefill-delta3", 
        "use_ffa_prefill": True,
        "delta": 3.0,
    },
    {"abbr": "ffa-prefill-delta8", 
        "use_ffa_prefill": True,
        "delta": 8.0,
    },
    
    {"abbr": "ffa-prefill-delta5-firsthalf", 
        "use_ffa_prefill": True,
        "delta": 5.0,
        "pattern_layers": list(range(0, 14))
    },
    {"abbr": "ffa-prefill-delta8-firsthalf", 
        "use_ffa_prefill": True,
        "delta": 8.0,
        "pattern_layers": list(range(0, 14))
    },
    
    {"abbr": "ffa-prefill-delta5-lasthalf", 
        "use_ffa_prefill": True,
        "delta": 5.0,
        "pattern_layers": list(range(14, 28))
    },
    {"abbr": "ffa-prefill-delta8-lasthalf", 
        "use_ffa_prefill": True,
        "delta": 8.0,
        "pattern_layers": list(range(14, 28))
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
        type=LlamaForCausalLM_FFA_OC,
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