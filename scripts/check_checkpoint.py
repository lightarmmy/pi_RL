#!/usr/bin/env python3
"""Check a Pi0.5 checkpoint layout without importing Torch or loading weights."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def safetensors_info(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("file is shorter than a safetensors header")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length > 64 * 1024 * 1024:
            raise ValueError(f"implausibly large safetensors header: {header_length}")
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise ValueError("safetensors header is incomplete")
    header = json.loads(raw)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError("safetensors file has no tensors")
    final_offset = max(value["data_offsets"][1] for value in tensors.values())
    expected_bytes = 8 + header_length + final_offset
    actual_bytes = path.stat().st_size
    keys = list(tensors)
    if any(key.startswith("paligemma_with_expert.") for key in keys):
        layout = "openpi_pytorch"
    elif any(key.startswith(("img.", "llm.", "action_in_proj.")) for key in keys):
        layout = "openpi_rlinf"
    else:
        layout = "unknown"
    return {
        "path": str(path.resolve()),
        "layout": layout,
        "tensor_count": len(tensors),
        "sample_keys": keys[:8],
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "complete": actual_bytes == expected_bytes,
    }


def find_model(path: Path) -> Path:
    if path.is_file():
        return path
    for relative in ("model.safetensors", "actor/model_state_dict/full_weights.pt", "full_weights.pt"):
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no supported model file in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--expected-layout", choices=("openpi_pytorch", "openpi_rlinf"))
    args = parser.parse_args()
    model = find_model(args.checkpoint)
    if model.suffix != ".safetensors":
        print(json.dumps({"path": str(model.resolve()), "layout": "pytorch_pt_unchecked", "note": "use a Torch environment to inspect .pt weights"}, indent=2))
        return 0
    info = safetensors_info(model)
    if args.norm_stats:
        stats = json.loads(args.norm_stats.read_text())
        if not isinstance(stats, dict) or not stats:
            raise ValueError("norm_stats must be a nonempty JSON object")
        info["norm_stats"] = str(args.norm_stats.resolve())
        info["norm_keys"] = list(stats)
    print(json.dumps(info, indent=2))
    if args.expected_layout and info["layout"] != args.expected_layout:
        raise ValueError(f"expected {args.expected_layout}, found {info['layout']}")
    if not info["complete"]:
        raise ValueError("checkpoint file is incomplete or has unexpected trailing bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
