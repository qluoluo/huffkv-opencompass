#!/usr/bin/env python3
"""
Export tensors to pixel-perfect PNGs (one value = one pixel) without matplotlib.
"""
import argparse
import glob
import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from matplotlib import cm

DEFAULT_TENSOR_NAMES = ["q_rope", "k_rope", "q_unrope", "k_unrope", "v", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render torch tensors to grayscale PNGs with 1 pixel per value."
    )
    default_result_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "result")
    )
    parser.add_argument(
        "--result-dir",
        default=default_result_dir,
        help="Root directory that contains model outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to store generated images. Defaults to <result-dir>/visualizations/pixel_png.",
    )
    parser.add_argument(
        "--tensor-path",
        default=None,
        help="Optional path to a single .pt file (skip discovery).",
    )
    parser.add_argument(
        "--tensor-names",
        nargs="+",
        default=DEFAULT_TENSOR_NAMES,
        help="Only render tensors whose filename (without .pt) is in this list.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to include, e.g. --layers 0 1 2.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Head index to visualize; if omitted, heads are averaged.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch index to visualize when a batch dimension exists.",
    )
    parser.add_argument(
        "--seq-start",
        type=int,
        default=None,
        help="Sequence start index (supports negative). Leave unset to keep full length.",
    )
    parser.add_argument(
        "--seq-end",
        type=int,
        default=None,
        help="Sequence end index (exclusive, supports negative). Leave unset to keep full length.",
    )
    parser.add_argument(
        "--seq-step",
        type=int,
        default=1,
        help="Stride when slicing sequence dimension. Use 1 for full resolution.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap name for RGB rendering (e.g., coolwarm, bwr, turbo, viridis).",
    )
    parser.add_argument(
        "--norm",
        choices=["absmax", "minmax", "none"],
        default="absmax",
        help="Value scaling: absmax (default) maps [-max,+max] to [0,255], minmax maps [min,max] to [0,255], none clamps to [0,255].",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Do not swap height/width before saving. By default the matrix is transposed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after rendering this many tensors.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/layer_data/layer_*/*.pt",
        help="Glob used to discover tensor files relative to --result-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering if the output image already exists.",
    )
    return parser.parse_args()


def discover_tensor_files(
    result_dir: str,
    tensor_names: Iterable[str],
    layers: Optional[Iterable[int]],
    glob_pattern: str,
) -> List[Tuple[str, int, str]]:
    """Return list of (tensor_path, layer_idx, tensor_name)."""
    pattern = os.path.join(result_dir, glob_pattern)
    candidates = glob.glob(pattern, recursive=True)
    tensor_names = set(tensor_names)
    allowed_layers = set(layers) if layers is not None else None
    results: List[Tuple[str, int, str]] = []

    for path in sorted(candidates):
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        if ext != ".pt" or (tensor_names and name not in tensor_names):
            continue

        match = re.search(r"layer_(\d+)", path)
        layer_idx = int(match.group(1)) if match else -1
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue

        results.append((path, layer_idx, name))
    return results


def _safe_slice_length(length: int, start: Optional[int], end: Optional[int]) -> tuple[int, int]:
    start_idx = 0 if start is None else (length + start if start < 0 else start)
    end_idx = length if end is None else (length + end if end < 0 else end)
    start_idx = max(0, min(length, start_idx))
    end_idx = max(start_idx, min(length, end_idx))
    return start_idx, end_idx


def prepare_matrix(
    tensor: torch.Tensor,
    batch_idx: int,
    head_idx: Optional[int],
    seq_start: Optional[int],
    seq_end: Optional[int],
    seq_step: int,
) -> np.ndarray:
    """
    Reduce to a 2D numpy array.
    - Accept shapes like [B, H, S, D] or [H, S, D].
    - If multiple heads/batches, select or average appropriately.
    - No truncation by default; seq_step controls downsampling.
    """
    t = tensor.detach().to(torch.float32)

    # Flatten any leading dims beyond 4 to make selection predictable
    if t.ndim > 4:
        t = t.flatten(0, t.ndim - 4)

    if t.ndim == 4:
        b_idx = min(batch_idx, t.shape[0] - 1)
        t = t[b_idx]
        if head_idx is not None and t.shape[0] > 1:
            h_idx = min(head_idx, t.shape[0] - 1)
            t = t[h_idx]
        elif t.shape[0] > 1:
            t = t.mean(dim=0)
    elif t.ndim == 3:
        if head_idx is not None and t.shape[0] > 1:
            h_idx = min(head_idx, t.shape[0] - 1)
            t = t[h_idx]
        elif t.shape[0] > 1:
            t = t.mean(dim=0)
    elif t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim == 0:
        t = t.view(1, 1)

    while t.ndim > 2:
        t = t.mean(dim=0)

    seq_len = t.shape[0]
    start_idx, end_idx = _safe_slice_length(seq_len, seq_start, seq_end)
    t = t[start_idx:end_idx:seq_step]

    return t.cpu().numpy()


def normalize_to_uint8(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "absmax":
        max_abs = np.max(np.abs(matrix))
        if max_abs == 0:
            return np.zeros_like(matrix, dtype=np.uint8)
        scaled = matrix / max_abs  # in [-1,1]
        return np.clip((scaled + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if mode == "minmax":
        mn, mx = float(np.min(matrix)), float(np.max(matrix))
        if mx == mn:
            return np.zeros_like(matrix, dtype=np.uint8)
        scaled = (matrix - mn) / (mx - mn)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    # mode == "none"
    return np.clip(matrix, 0, 255).astype(np.uint8)


def colorize_uint8(gray: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(gray.astype(np.float32) / 255.0, bytes=True)
    return rgba[..., :3]  # drop alpha


def save_png(matrix: np.ndarray, output_path: str, mode: str, cmap_name: str, transpose: bool) -> None:
    if transpose:
        matrix = matrix.T
    arr = normalize_to_uint8(matrix, mode=mode)
    rgb = colorize_uint8(arr, cmap_name=cmap_name)
    image = Image.fromarray(rgb, mode="RGB")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def build_output_path(output_dir: str, result_dir: str, tensor_path: str) -> str:
    rel = os.path.relpath(tensor_path, result_dir)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(output_dir, f"{rel_no_ext}.png")


def main() -> None:
    args = parse_args()
    result_dir = os.path.abspath(args.result_dir)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(result_dir, "visualizations", "pixel_png")
    )

    if args.tensor_path:
        tensor_files = [
            (
                os.path.abspath(args.tensor_path),
                -1,
                os.path.splitext(os.path.basename(args.tensor_path))[0],
            )
        ]
    else:
        tensor_files = discover_tensor_files(
            result_dir=result_dir,
            tensor_names=args.tensor_names,
            layers=args.layers,
            glob_pattern=args.glob_pattern,
        )

    if not tensor_files:
        print("No tensor files found. Check --result-dir, --tensor-names, or --layers.")
        return

    print(f"Found {len(tensor_files)} tensor file(s). Saving images to {output_dir}")

    processed = 0
    for tensor_path, layer_idx, tensor_name in tensor_files:
        if args.limit is not None and processed >= args.limit:
            break

        out_path = build_output_path(output_dir, result_dir, tensor_path)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] {tensor_path}")
            continue

        try:
            tensor = torch.load(tensor_path, weights_only=True, map_location="cpu")
            matrix = prepare_matrix(
                tensor=tensor,
                batch_idx=args.batch,
                head_idx=args.head,
                seq_start=args.seq_start,
                seq_end=args.seq_end,
                seq_step=args.seq_step,
            )
            save_png(
                matrix,
                out_path,
                mode=args.norm,
                cmap_name=args.cmap,
                transpose=not args.no_transpose,
            )
            print(
                f"[ok] {tensor_path} -> {out_path} ({matrix.shape[0]}x{matrix.shape[1] if matrix.ndim>1 else 1})"
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {tensor_path}: {exc}")

    print(f"Done. Rendered {processed} image(s).")


if __name__ == "__main__":
    main()
