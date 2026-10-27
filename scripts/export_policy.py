#!/usr/bin/env python3
"""Export an RLinf Pi0.5 full_weights.pt for OpenPI PyTorch serving."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RLinf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="RLinf full_weights.pt")
    parser.add_argument("--reference", required=True, type=Path, help="matching OpenPI PyTorch SFT directory")
    parser.add_argument("--norm-stats", required=True, type=Path, help="task norm_stats.json")
    parser.add_argument("--asset-id", required=True, help="OpenPI data asset id, e.g. physical-intelligence/libero")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()

    from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import convert_trained_ckpt

    checkpoint = args.checkpoint.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    norm_stats = args.norm_stats.resolve(strict=True)
    output = args.output.resolve()
    if checkpoint.name != "full_weights.pt":
        parser.error("--checkpoint must point to RLinf full_weights.pt")
    if not (reference / "model.safetensors").is_file():
        parser.error("--reference needs model.safetensors")
    if not norm_stats.is_file():
        parser.error("--norm-stats must be a file")
    asset_parts = Path(args.asset_id).parts
    if not asset_parts or any(part in ("", ".", "..") for part in asset_parts) or Path(args.asset_id).is_absolute():
        parser.error("--asset-id must be a relative path without . or ..")
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        # RLinf validates every converted key and shape against the PyTorch
        # reference and restores the action-expert lm_head it does not train.
        convert_trained_ckpt(str(checkpoint), str(staging), str(reference))
        stats_dst = staging / "assets" / Path(*asset_parts) / "norm_stats.json"
        stats_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(norm_stats, stats_dst)
        if not (staging / "model.safetensors").is_file():
            raise RuntimeError("conversion did not produce model.safetensors")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"exported policy: {output}", flush=True)


if __name__ == "__main__":
    main()
