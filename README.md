# πRL / π₀.₅ replication and real robot handoff

This project uses the paper authors' RLinf implementation of Flow-SDE and
Flow-Noise PPO with a pinned OpenPI source checkout. The separate `openpi/`
checkout is for upstream model and inference reference. RLinf installs its
compatible `rlinf-openpi==0.1.1` package for training; do not replace that
package with the current upstream OpenPI checkout without testing compatibility.
Use Python 3.11 for the project and RLinf environment.

`RLinf/`, `openpi/`, `artifacts/`, and `runs/` are local and excluded from this
project's Git repository. Their revisions are recorded in `UPSTREAM_LOCK.json`.
Restore source checkouts after a GitHub clone with `python scripts/bootstrap.py`.

## Verified cluster smoke run

The public SFT checkpoint completed one LIBERO Spatial Flow-SDE PPO step on
8 H800 GPUs in `runs/h800-smoke-20261027-0030/`. The run collected 8 trajectories,
performed an actor/critic update, evaluated 8 trajectories, and saved
`libero_spatial_ppo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt`.
This is a 10-step integration test, **not** a task-success or paper-reproduction
result. Its evaluation success was 0/8. The cluster image needs the EGL packages
shown in `k8s/h800_smoke_job.yaml`; 64 simultaneous simulator environments
exceeded a 256 GiB container limit, so the smoke launcher uses 8 environments.
The renderer is pinned to EGL device 0 for this image. Use a proper NVIDIA EGL
image before scaling up simulation throughput.

## Current substitute checkpoint

The public SFT model is
[`RLinf/RLinf-Pi05-LIBERO-SFT`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT)
at the revision in `UPSTREAM_LOCK.json`. Its 7.47 GB `model.safetensors` and
`physical-intelligence/libero/norm_stats.json` belong under
`artifacts/public_pi05_libero/`. The model download is intentionally outside Git.

Check the file before training:

```bash
python scripts/check_checkpoint.py artifacts/public_pi05_libero \
  --norm-stats artifacts/public_pi05_libero/physical-intelligence/libero/norm_stats.json
```

The checker recognizes the PyTorch OpenPI key layout and the RLinf OpenPI key
layout without loading the 7.47 GB tensor data. A `.pt` checkpoint is detected
but needs a Torch environment for deeper validation. The future robot checkpoint
must be checked for its exact model variant, image ordering, action space, action
horizon, tokenizer and matching normalization statistics before rollout.

## Training and evaluation entry points

Install RLinf's OpenPI + LIBERO environment on a GPU machine as described in
`RLinf/requirements/install.sh` and `RLinf/docs/source-en/rst_source/examples/embodied/pi0.rst`.
The installer uses a patched Transformers/OpenPI stack. Run these commands in
that environment:

```bash
python scripts/launch.py sim-smoke
python scripts/launch.py sim-smoke --validate-config
python scripts/launch.py sim-smoke --execute
python scripts/launch.py sim-train --execute
```

Without `--execute`, the launcher prints the exact RLinf command. `sim-smoke`
performs one short online PPO iteration and saves its result. `sim-train` uses
the upstream π₀.₅ LIBERO Spatial recipe. Both use the public SFT checkpoint by
default; `--checkpoint` selects another checkpoint. The upstream recipe uses
Flow-SDE; Flow-Noise can be selected through Hydra overrides after a Flow-SDE
baseline passes.

The reproducible cluster manifests are under `k8s/`. `h800_install_job.yaml`
builds the compatible Python environment (the project-local `.venv_h800` is
excluded from Git), and `h800_smoke_job.yaml` runs the short GPU integration
test. These manifests contain cluster-specific image, namespace, storage, and
GPU settings; change them for another site. The installer links existing
LIBERO assets at a path outside this repository. On another machine, provide
your own LIBERO assets and tokenizer before running the manifests.

## RL checkpoint export and inference

RLinf saves a training checkpoint as `full_weights.pt`; OpenPI's WebSocket
server expects `model.safetensors` in the OpenPI PyTorch layout. Use the
matching pre-RL SFT model as the conversion reference (it provides the
action-expert head, which RLinf does not retain) and the task's normalization
statistics:

```bash
python scripts/export_policy.py \
  --checkpoint /path/to/global_step_N/actor/model_state_dict/full_weights.pt \
  --reference artifacts/public_pi05_libero \
  --norm-stats artifacts/public_pi05_libero/physical-intelligence/libero/norm_stats.json \
  --asset-id physical-intelligence/libero \
  --output /path/to/exported_policy
python scripts/serve_policy.py --checkpoint /path/to/exported_policy \
  --config pi05_libero --discrete-state-input --self-test
python scripts/serve_policy.py --checkpoint /path/to/exported_policy \
  --config pi05_libero --discrete-state-input --host 127.0.0.1 --port 8000
```

The export uses RLinf's converter, which checks every weight key and shape
against the reference. The smoke checkpoint exported successfully and the
compatible OpenPI runtime loaded it with its LIBERO norm stats and returned a
finite `(10, 7)` action chunk for a synthetic observation. The official
`pi05_libero` service config defaults to *continuous* state input, whereas
this RLinf π₀.₅ recipe uses *discrete* state input; the explicit flag above
is required. For another robot, substitute its own matching reference model,
OpenPI config, asset ID, normalization stats, observation transform, and state
mode. Do not point the stock OpenPI server directly at `full_weights.pt`.

RLinf also provides `realworld_peginsertion_async_ppo_pi05.yaml` with a
Franka peg insertion online PPO configuration. An example of the robot-specific
values needed by the launcher is in `configs/franka_peg.example.json`:

```bash
python scripts/launch.py real-eval --site /path/to/site.json
python scripts/launch.py real-rl --site /path/to/site.json
```

The printed commands are for review. `--execute` starts the RLinf runtime and
requires an installed robot/GPU environment and a complete checkpoint. The
example JSON deliberately contains placeholders. The real environment,
controller, reward, reset, cameras, network layout, and normalization must be
validated for the actual robot before use. The bundled real recipe is Franka
peg insertion; another robot or task needs its own RLinf environment/config.
`configs/franka_peg.mock.json` is only for Hydra configuration validation and
cannot be passed with `--execute`.

For action-chunk rollout from a robot process, `pi_rl_real.rollout` provides
`RemoteOpenPIPolicy` and `run_episode`. The robot-specific adapter supplies
`reset`, `observe`, `validate_action`, `step`, and `stop`; the policy server
returns an action chunk, and `run_episode` executes a configurable prefix before
replanning. The `on_step` callback receives raw observations, action, and
feedback for a LeRobot recorder or task-specific logger. Run
`python scripts/mock_rollout.py` to exercise this boundary without hardware.
The robot adapter must implement units, coordinate frames, hard action limits,
controller timing, reset, and emergency stop for its own platform.
On the robot machine, install this small package with `pip install -e .` and
the official WebSocket client with `pip install -e openpi/packages/openpi-client`.
The GPU-side server is this project's `scripts/serve_policy.py` entry point for
the exported checkpoint; the robot process passes its server address to
`RemoteOpenPIPolicy`.

## Completion criteria

1. Public checkpoint layout and normalization files pass validation. **Done.**
2. A LIBERO smoke run completes rollout, log-prob/reward collection, one PPO
   update, checkpoint save, and evaluation. **Done** with the public SFT model.
3. A longer Flow-SDE run improves over its SFT starting checkpoint; Flow-Noise
   is checked separately. **Not yet tested.**
4. The robot site can load its own checkpoint and complete deterministic
   rollout before any stochastic real-world update. **Pending the real
   checkpoint and hardware settings.**

The paper's robot demonstration used RL in simulation followed by transfer to
the real robot. Direct real-world RL is a separate deployment validation step.
