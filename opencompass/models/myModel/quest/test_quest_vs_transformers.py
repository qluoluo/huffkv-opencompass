#!/usr/bin/env python3
"""Quick check of Quest vs vanilla Transformers outputs."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_FILE = Path(__file__).resolve().parent / "test_text.txt"

# Try installed quest package first, then fall back to the local ffa-quest checkout.
try:
    from quest import LlamaForCausalLM as QuestLlama
except ImportError:
    quest_repo = Path(__file__).resolve().parents[5] / "ffa-quest" / "quest"
    sys.path.append(str(quest_repo))
    from quest import LlamaForCausalLM as QuestLlama


def _pick_device(user_device: str) -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_prompt_text() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def _generate_with_transformers(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
):
    tokenizer = _load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    # Drop the prompt so we only decode the new tokens
    gen_ids = output_ids[:, input_len:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text


def _generate_with_quest(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    token_budget: int,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    tokenizer = _load_tokenizer(model_path)
    quest_model = QuestLlama.from_pretrained(model_path, torch_dtype=dtype)
    quest_model.to(device)
    quest_model.eval()
    quest_model.generation_config.do_sample = False
    quest_model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    # Only allocate KV cache for the actually needed context to avoid OOM on large-context models
    cfg_max_len = getattr(quest_model.config, "max_position_embeddings", None)
    max_seq_len = min(cfg_max_len, seq_len + max_new_tokens) if cfg_max_len else (seq_len + max_new_tokens)

    quest_model.quest_init(
        page_size=page_size,
        max_seq_len=max_seq_len,
        token_budget=token_budget,
        dtype=dtype,
        device=device,
    )

    inputs = inputs.to(device)
    with torch.no_grad():
        quest_ids = quest_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    quest_model.quest_clear()

    # Drop the prompt so we only decode the new tokens
    gen_ids = quest_ids[:, seq_len:]
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text


def _run(args) -> Tuple[str, str]:
    device = _pick_device(args.device)
    if device.type != "cuda":
        print("Warning: Quest is optimized for CUDA; CPU runs may be slow or unsupported.", file=sys.stderr)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    vanilla = _generate_with_transformers(args.model_path, args.prompt, args.max_new_tokens, device, dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    quest = _generate_with_quest(
        args.model_path,
        args.prompt,
        args.max_new_tokens,
        args.token_budget,
        args.page_size,
        device,
        dtype,
    )
    return vanilla, quest


def main():
    parser = argparse.ArgumentParser(
        description="Compare Quest Llama output with vanilla Transformers output on the same prompt."
    )
    parser.add_argument(
        "--model_path",
        default="/inspire/hdd/global_user/liuzhigeng-253108120105/models/Llama-3.1-8B",
        help="Local path or HF repo id for a Llama model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Number of tokens to generate.")
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1024,
        help="Quest token budget (smaller values make differences easier to spot).",
    )
    parser.add_argument("--page-size", type=int, default=16, help="Quest page size.")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    args.prompt = _load_prompt_text()

    vanilla, quest = _run(args)

    print(f"Prompt (truncated to 200 chars): {args.prompt[:200]}{'...' if len(args.prompt) > 200 else ''}")
    print("\n=== Vanilla Transformers ===")
    print(vanilla)
    print("\n=== Quest ===")
    print(quest)
    if vanilla == quest:
        print("\nOutputs are identical.")
    else:
        print("\nOutputs differ.")


if __name__ == "__main__":
    main()
