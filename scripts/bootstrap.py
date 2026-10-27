#!/usr/bin/env python3
"""Restore the upstream source revisions recorded in UPSTREAM_LOCK.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    lock = json.loads((ROOT / "UPSTREAM_LOCK.json").read_text())
    for name in ("RLinf", "openpi"):
        spec = lock[name]
        target = ROOT / name
        if not target.exists():
            # Populate the worktree even when the remote HEAD already matches
            # the lock; --no-checkout would leave such a fresh clone empty.
            git("clone", "--filter=blob:none", spec["url"], str(target))
        if not (target / ".git").exists():
            raise RuntimeError(f"{target} exists but is not a Git checkout")
        head = git("rev-parse", "HEAD", cwd=target)
        if head != spec["commit"]:
            git("fetch", "--depth=1", "origin", spec["commit"], cwd=target)
            git("checkout", "--detach", spec["commit"], cwd=target)
        print(f"{name}: {git('rev-parse', 'HEAD', cwd=target)}")


if __name__ == "__main__":
    main()
