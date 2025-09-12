import os
import torch
from tqdm import tqdm


def load_qkvh(load_dir: str, device='cpu'):
    # 获取所有以'layer'开头的子目录并按数字排序
    dirname_list = sorted(
        [x for x in os.listdir(load_dir) if x.startswith("layer")],
        key=lambda x: int(x.split("_")[1]),
    )
    layer_num = len(dirname_list)
    ret_list = []

    # 验证目录命名是否符合layer_0, layer_1...的格式
    assert dirname_list == [
        f"layer_{i}" for i in range(layer_num)
    ], "Layer directories must be named layer_0, layer_1, ..."

    for i in range(layer_num):
        layer_dir = os.path.join(load_dir, f"layer_{i}")

        load_data_list = ["q_rope", "k_rope", "q_unrope", "k_unrope", "v", "h"]

        data = {}
        for data_name in load_data_list:
            data_path = os.path.join(layer_dir, f"{data_name}.pt")
            data[data_name] = torch.load(
                data_path, weights_only=True, map_location=device
            )

        yield data


def tokenize_text(tokenizer_path, text):
    """
    使用传入的 tokenizer 对 text 进行分词并返回分词列表
    :param tokenizer: 一个 HuggingFace tokenizer 对象
    :param text: 要分词的文本
    :return: 分词后的 token 列表
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 编码并转换为 token ID
    # token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    # 将 token ID 转回 token 字符串
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    tokens = [token.lstrip("Ġ") for token in tokens]
    return tokens


def load_from_longbench_jsonl(jsonl_path, line_idx=0):
    import datasets

    dataset_path = jsonl_path
    dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
    raw_text = dataset[line_idx]["context"]

    return (
        raw_text,
        f"longbench_{os.path.splitext(os.path.basename(dataset_path))[0]}_{line_idx}",
    )


def load_from_babilong_json(json_path, line_idx=0):
    import datasets

    dataset_path = json_path
    dataset = datasets.load_dataset("json", data_files=dataset_path)
    raw_text = dataset[line_idx]["input"]

    return (
        raw_text,
        f"babilong_{os.path.splitext(os.path.basename(dataset_path))[0]}_{line_idx}",
    )

if __name__ == "__main__":
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')
    from transformers.models.llama.modeling_llama import repeat_kv
    
    # Iterate through the layers and plot the attention weights and their distribution
    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v = layer_qkvh_data["v"].to('cuda')
        
        # sample_seq_len = -1
        sample_seq_len = 8 * 1024
        if sample_seq_len > 0:
            q_rope = q_rope[..., :sample_seq_len, :]
            k_rope = k_rope[..., :sample_seq_len, :]

        q_rope = q_rope[..., -1:, :]  # Focusing on the last position (query part)

        bsz, num_heads, seq_len, head_dim = q_rope.shape
        _, num_kv_heads, _, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"