#!/usr/bin/env python3
"""Tiny Quest text-generation demo that can run outside the Quest repo."""

import sys
from pathlib import Path

from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Point to the Quest repo so the custom attention patch can be imported without pip install.
QUEST_REPO = Path("/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest")
if str(QUEST_REPO) not in sys.path:
    sys.path.append(str(QUEST_REPO))

from evaluation import quest_attention  # noqa: E402
from evaluation.quest_attention import enable_quest_attention_eval  # noqa: E402


def main():
    model_path = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/models/Llama-3_2-3B"
    prompt = (
        "### Instruction:\n"
        "Please write a story about a robot learning how to love.\n"
        "### Response:"
    )
    max_new_tokens = 64

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    dtype = torch.float16 if use_gpu else torch.float32

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
        factor = config.rope_scaling.get("factor", 1.0)
        config.rope_scaling = {"type": "linear", "factor": factor}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto" if use_gpu else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if not use_gpu:
        model = model.to(device)
    model.eval()

    quest_attention.layer_id = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else quest_attention.layer_id
    quest_args = SimpleNamespace(token_budget=256, chunk_size=16)
    enable_quest_attention_eval(model, quest_args)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
