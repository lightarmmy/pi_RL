#!/usr/bin/env python3
"""Serve an exported RLinf Pi0.5 policy with the compatible OpenPI runtime."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="exported policy directory")
    parser.add_argument("--config", default="pi05_libero", help="OpenPI TrainConfig name")
    parser.add_argument("--discrete-state-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 only on a trusted network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    parser.add_argument("--check-only", action="store_true", help="load weights and transforms, then exit")
    parser.add_argument("--self-test", action="store_true", help="run one synthetic LIBERO inference, then exit")
    args = parser.parse_args()

    if not (args.checkpoint / "model.safetensors").is_file():
        parser.error("checkpoint must contain model.safetensors; run scripts/export_policy.py first")

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.config)
    if not hasattr(train_config.model, "discrete_state_input"):
        parser.error(f"config {args.config} has no discrete_state_input setting")
    model_config = dataclasses.replace(
        train_config.model,
        discrete_state_input=args.discrete_state_input,
    )
    train_config = dataclasses.replace(train_config, model=model_config)
    logging.info("Loading %s with discrete_state_input=%s", args.checkpoint, args.discrete_state_input)
    policy = policy_config.create_trained_policy(
        train_config,
        str(args.checkpoint.resolve()),
        default_prompt=args.default_prompt,
    )
    logging.info("Policy loaded; metadata=%s", policy.metadata)
    if args.self_test:
        if args.config != "pi05_libero":
            parser.error("--self-test currently supports only pi05_libero")
        import numpy as np
        from openpi.policies.libero_policy import make_libero_example

        result = policy.infer(make_libero_example())
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid LIBERO actions: shape={actions.shape}")
        logging.info("Self-test inference passed: action_shape=%s", actions.shape)
        return
    if args.check_only:
        return
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
