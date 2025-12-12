#!/usr/bin/env python3
"""
Quick Quest attention generation demo that can live outside the ffa-quest repo.

Place this script next to the repository folder (e.g. projects/ffa/test_quest_generate.py
with projects/ffa/ffa-quest/quest containing evaluation/quest_attention.py) and run:
  python test_quest_generate.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_demo_args() -> SimpleNamespace:
    """Return a ready-to-run config with no CLI required."""
    repo_root = Path(__file__).resolve().parents[5] / "ffa-quest" / "quest"
    if not repo_root.exists():
        raise SystemExit(
            f"Quest repo not found at {repo_root}. Adjust `repo_root` in build_demo_args()."
        )

    use_gpu = torch.cuda.is_available()

    return SimpleNamespace(
        # A tiny Llama-based model so the download is quick.
        # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model="/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/models/Llama-3_2-3B",
        prompt=(
            "### Instruction:\n"
            "Explain briefly what Quest attention does and why it helps during text generation.\n"
            "### Response:"
        ),
        max_new_tokens=48,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        quest=True,
        token_budget=256,
        chunk_size=16,
        quest_repo=str(repo_root),
        device_map="auto" if use_gpu else "none",
        device="cuda" if use_gpu else "cpu",
        dtype="float16" if use_gpu else "float32",
        seed=0,
    )


def str_to_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(
            f"Quest repo not found at {repo_root}. Update `quest_repo` in build_demo_args()."
        )
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def load_model_and_tokenizer(args: SimpleNamespace) -> tuple:
    dtype = str_to_dtype(args.dtype)
    if args.device.startswith("cpu") and dtype != torch.float32:
        print("Switching dtype to float32 for CPU execution.")
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = model.eval()

    if model_kwargs["device_map"] is None:
        model.to(args.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    return model, tokenizer, dtype


def run_generation(model, tokenizer, prompt: str, args: SimpleNamespace) -> str:
    target_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cache=True,
    )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    args = build_demo_args()
    torch.manual_seed(args.seed)

    repo_root = Path(args.quest_repo)
    add_repo_to_path(repo_root)

    from evaluation import quest_attention
    from evaluation.quest_attention import enable_quest_attention_eval

    model, tokenizer, dtype = load_model_and_tokenizer(args)
    print(
        f"Loaded {args.model} on device_map={args.device_map} with dtype={dtype}.",
        flush=True,
    )

    if args.quest:
        # Reset the layer_id counter to match the current model depth.
        if hasattr(model.config, "num_hidden_layers"):
            quest_attention.layer_id = model.config.num_hidden_layers
        quest_args = SimpleNamespace(
            token_budget=args.token_budget,
            chunk_size=args.chunk_size,
        )
        enable_quest_attention_eval(model, quest_args)
        print(
            f"Quest attention enabled (token_budget={args.token_budget}, "
            f"chunk_size={args.chunk_size}).",
            flush=True,
        )
    else:
        print("Quest attention disabled; using vanilla attention.", flush=True)

    full_text = run_generation(model, tokenizer, args.prompt, args)
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Output ===")
    print(full_text)


if __name__ == "__main__":
    main()
