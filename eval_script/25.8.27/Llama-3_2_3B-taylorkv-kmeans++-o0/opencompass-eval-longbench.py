import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from mmengine.config import read_base

from opencompass.models import LlamaForCausalLM_TaylorKV_OC as LlamaForCausalLM_OC

# 导入数据集和汇总器配置
with read_base():
    # longbench
    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets
    
    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

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
    # {"abbr": "CrucialKV", "use_remain": False},
    {"abbr": "TaylorKV-Cluster-o0-ck4", "use_remain": True, "remain_cluster_k": 4, "remain_order": 0},
    {"abbr": "TaylorKV-Cluster-o0-ck8", "use_remain": True, "remain_cluster_k": 8, "remain_order": 0},
    # {"abbr": "TaylorKV-Cluster-o0-ck16", "use_remain": True, "remain_cluster_k": 16, "remain_order": 0},
    # {"abbr": "TaylorKV-Cluster-o0-ck32", "use_remain": True, "remain_cluster_k": 32, "remain_order": 0},
    {"abbr": "TaylorKV-Cluster-o0-ck64", "use_remain": True, "remain_cluster_k": 64, "remain_order": 0},

    # {"abbr": "TaylorKV-Cluster-o1-ck4", "use_remain": True, "remain_cluster_k": 4, "remain_order": 1},
    # {"abbr": "TaylorKV-Cluster-o1-ck8", "use_remain": True, "remain_cluster_k": 8, "remain_order": 1},
    # {"abbr": "TaylorKV-Cluster-o1-ck16", "use_remain": True, "remain_cluster_k": 16, "remain_order": 1},
    # {"abbr": "TaylorKV-Cluster-o1-ck32", "use_remain": True, "remain_cluster_k": 32, "remain_order": 1},
    # {"abbr": "TaylorKV-Cluster-o1-ck64", "use_remain": True, "remain_cluster_k": 64, "remain_order": 1},
]

# 通用量化参数
COMMON_QUANT_KWARGS = {
    "window_size": 8, 
    "sparse_num": 512,
    "debug": True,
    "remain_kmeans_args": {
        "iters": 50,
        "init_method": "k-means++",
        "random_state": 0,
    },
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

        ####################
        # Longbench 的基础配置
        drop_middle=True,
        max_seq_len=31500
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