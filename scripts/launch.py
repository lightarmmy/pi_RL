#!/usr/bin/env python3
"""Launch the pinned RLinf Pi0.5 recipes with project-specific preflight checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from check_checkpoint import find_model, safetensors_info


ROOT = Path(__file__).resolve().parents[1]
RLINF = ROOT / "RLinf"
EMBODIED = RLINF / "examples" / "embodiment"
PUBLIC_MODEL = ROOT / "artifacts" / "public_pi05_libero"


def check_model(path: Path) -> None:
    model = find_model(path)
    if model.suffix == ".safetensors" and not safetensors_info(model)["complete"]:
        raise ValueError(f"incomplete checkpoint: {model}")
    if not model.is_file():
        raise FileNotFoundError(model)


def site_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = ("robot_ip", "target_ee_pose", "camera_names", "main_image_key", "model_path")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"missing site fields: {', '.join(missing)}")
    if len(data["target_ee_pose"]) != 6 or not all(isinstance(x, (int, float)) for x in data["target_ee_pose"]):
        raise ValueError("target_ee_pose must contain six numbers")
    if not isinstance(data["camera_names"], dict):
        raise ValueError("camera_names must map camera serials to model image keys")
    if data["main_image_key"] not in data["camera_names"].values():
        raise ValueError("main_image_key must be one of camera_names values")
    if any("REPLACE" in str(value) for value in (data["robot_ip"], *data["camera_names"].keys())):
        raise ValueError("replace the example robot IP and camera serial before use")
    return data


def as_override(key: str, value: object) -> str:
    if isinstance(value, dict):
        # Hydra's override grammar rejects JSON's quoted dictionary keys.
        # Robot camera serials and OpenPI image names only need simple tokens.
        for item_key, item_value in value.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(item_key)) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(item_value)):
                raise ValueError(f"unsupported Hydra mapping token: {item_key}={item_value}")
        encoded = "{" + ",".join(f"{item_key}:{item_value}" for item_key, item_value in value.items()) + "}"
    elif isinstance(value, list):
        encoded = json.dumps(value, separators=(",", ":"))
    elif isinstance(value, bool):
        encoded = str(value).lower()
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def validate_hydra(config_name: str, overrides: list[str]) -> None:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("Hydra is not installed in this Python environment") from error
    os.environ["EMBODIED_PATH"] = str(EMBODIED)
    with initialize_config_dir(config_dir=str(EMBODIED / "config"), version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)
    print(json.dumps({
        "config_name": config_name,
        "model_path": OmegaConf.select(cfg, "actor.model.model_path"),
        "rollout_model_path": OmegaConf.select(cfg, "rollout.model.model_path"),
        "pi05": OmegaConf.select(cfg, "actor.model.pi05"),
        "noise_method": OmegaConf.select(cfg, "actor.model.openpi.noise_method"),
        "only_eval": OmegaConf.select(cfg, "runner.only_eval"),
        "rollout_model_type": OmegaConf.select(cfg, "rollout.model.model_type"),
    }, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim-smoke", "sim-train", "real-eval", "real-rl"))
    parser.add_argument("--checkpoint", type=Path, help="checkpoint directory; public Pi0.5 SFT is the sim default")
    parser.add_argument("--site", type=Path, help="JSON site configuration for a Franka peg insertion task")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--extra", action="append", default=[], help="additional Hydra key=value override")
    parser.add_argument("--validate-config", action="store_true", help="compose the RLinf Hydra config and exit")
    parser.add_argument("--execute", action="store_true", help="run after preflight; default prints the command")
    args = parser.parse_args()

    if not (EMBODIED / "train_embodied_agent.py").is_file():
        raise FileNotFoundError("RLinf checkout is missing; restore the locked upstream revision")

    mode = args.mode
    real = mode.startswith("real-")
    site = site_config(args.site) if args.site else None
    if real and site is None:
        parser.error("real modes require --site with robot and checkpoint settings")
    if site and not real:
        parser.error("--site is only used for real modes")
    if real and args.execute and (args.site.name.endswith(".mock.json") or site["robot_ip"].startswith("127.")):
        parser.error("mock site config cannot be used with --execute")

    checkpoint = args.checkpoint or (Path(site["model_path"]) if real else PUBLIC_MODEL)
    checkpoint = checkpoint.resolve()
    if args.execute:
        check_model(checkpoint)

    config_name = "realworld_peginsertion_async_ppo_pi05" if real else "libero_spatial_ppo_openpi_pi05"
    script = "train_async.py" if real else "train_embodied_agent.py"
    run_dir = args.run_dir or ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}"
    run_dir = run_dir.resolve()
    overrides = [
        as_override("runner.logger.log_path", str(run_dir)),
        as_override("actor.model.model_path", str(checkpoint)),
        as_override("rollout.model.model_path", str(checkpoint)),
    ]
    if mode == "sim-smoke":
        overrides.extend([
            "runner.max_epochs=1", "runner.val_check_interval=1", "runner.save_interval=1",
            "env.train.rollout_epoch=1", "env.train.total_num_envs=8",
            "env.train.max_episode_steps=10", "env.train.max_steps_per_rollout_epoch=10",
            "env.eval.total_num_envs=8", "env.eval.max_episode_steps=10",
            "env.eval.max_steps_per_rollout_epoch=10", "actor.micro_batch_size=1",
            "actor.global_batch_size=8",
        ])
    if real:
        overrides.append(as_override("runner.only_eval", mode == "real-eval"))
        overrides.append(as_override("cluster.node_groups.1.hardware.configs.0.robot_ip", site["robot_ip"]))
        if mode == "real-eval":
            # The training recipe has only a rollout model stub. RLinf explicitly
            # requires a full rollout model spec in eval-only mode.
            overrides.extend([
                "++rollout.model.model_type=openpi",
                "++rollout.model.num_action_chunks=5",
                "++rollout.model.num_steps=4",
                "++rollout.model.action_dim=7",
                "++rollout.model.openpi.config_name=pi05_franka",
                "++rollout.model.openpi.task=eval",
                "++rollout.model.openpi.num_images_in_input=1",
                "++rollout.model.openpi.noise_method=flow_ode",
                "++rollout.model.openpi.action_horizon=8",
                "++rollout.model.openpi.action_chunk=5",
            ])
        for split in ("train", "eval"):
            overrides.extend([
                as_override(f"++env.{split}.override_cfg.target_ee_pose", site["target_ee_pose"]),
                as_override(f"++env.{split}.override_cfg.camera_names", site["camera_names"]),
                as_override(f"env.{split}.main_image_key", site["main_image_key"]),
            ])
        if "reward_threshold" in site:
            overrides.append(as_override("env.train.override_cfg.reward_threshold", site["reward_threshold"]))
    overrides.extend(args.extra)
    command = [
        sys.executable, str(EMBODIED / script), "--config-path", str(EMBODIED / "config"),
        "--config-name", config_name, *overrides,
    ]
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if args.validate_config:
        validate_hydra(config_name, overrides)
        return 0
    if not args.execute:
        return 0
    validate_hydra(config_name, overrides)
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["EMBODIED_PATH"] = str(EMBODIED)
    env["REPO_PATH"] = str(RLINF)
    env["PYTHONPATH"] = str(RLINF) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO" if not real else "FRANKA")
    return subprocess.call(command, cwd=RLINF, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
