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

    for i in tqdm(range(layer_num)):
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


def load_from_longbench_jsonl(jsonl_path, line_start=0, line_end=-1):
    import datasets
    import os

    # 确保 line_start 和 line_end 是整数
    line_start = int(line_start)
    line_end = int(line_end)
    
    # 如果只指定了一个行号，则只读取该行
    if line_end < line_start:
        line_end = line_start

    dataset_path = jsonl_path
    dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
    
    # 生成需要读取的行索引列表
    line_indices = list(range(line_start, line_end + 1))
    
    # 获取多个文本行并用换行符连接
    raw_texts = [dataset[i]["context"] for i in line_indices]
    raw_text = "\n".join(raw_texts)

    # 生成包含所有索引范围的标识符
    indices_str = f"{line_start}_{line_end}" if line_end > line_start else f"{line_start}"
    return (
        raw_text,
        f"longbench_{os.path.splitext(os.path.basename(dataset_path))[0]}_{indices_str}",
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
