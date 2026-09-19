# VLA on RunPod

The policy (SmolVLA, π0, ACT, or any LeRobot checkpoint) runs on a RunPod GPU. The laptop streams observations to it and
executes the returned action chunks on the arm through dimOS.

```
arm camera + joint state ─▶ RemotePolicyModule (laptop, dimOS) ──HTTP──▶ policy_server.py (RunPod GPU)
                                   │                                       loads the LeRobot checkpoint
                                   ▼                                       returns an action chunk
                     ControlCoordinator joint_trajectory ─▶ OpenYAM arm
```

- `policy_server.py`: `GET /info` (joint count, action steps, the checkpoint's action range), `POST /act`
  (image + joint state + task → action chunk). `--dummy` holds the current pose, for tests without a checkpoint.
- `remote_policy.py`: a dimOS module with the same inputs and safety checks as dimOS's `LeRobotPolicyModule`.
  It rejects stale observations and non-finite values, clips to the checkpoint's action range, and cancels the
  trajectory on stop. Only the model call goes over the network.

Observation and action names match `dimos dataprep` output: `observation.images.wrist`, `observation.state`, `action`.

## Run on RunPod

1. Start a GPU pod from a PyTorch template and expose HTTP port 8000.
2. On the pod: `pip install lerobot==0.6.0 opencv-python-headless` (the version dimOS's runtime pins), copy
   `vla/policy_server.py` and the checkpoint over, then `python policy_server.py --policy-path <checkpoint>`.
3. The laptop reaches it through RunPod's HTTP proxy: `https://<pod-id>-8000.proxy.runpod.net/info`.

Each chunk costs one round trip. With 10 steps per chunk at 30 fps, inference plus network must stay under
~330 ms, or the arm pauses between chunks.

## Test without the arm

```bash
python vla/policy_server.py --dummy --host 127.0.0.1 --port 8011 &
PYTHONPATH=. python vla/sim_test.py --server http://127.0.0.1:8011    # dimOS's python, e.g. ../dimos/.venv/bin/python
```

`sim_test.py` forces dimOS's mock OpenYAM adapter and refuses to run if it would get the real one. Result on
2026-09-19: preflight passed, 12 chunks accepted in 4 s, and stop cancelled cleanly.

## Record demos for fine-tuning

```bash
PYTHONPATH=. python vla/collect_openyam.py --real --camera-index <arm camera>   # energizes the arm; asks for 'go'
dimos dataprep build --source <session.db> --config vla/openyam_dataprep.json --output data/datasets/openyam
```

Quest teleop moves the arm (B starts/saves an episode, Y discards), or press Enter in the terminal to start or save.
Train on RunPod with LeRobot on the dataset, then serve the checkpoint with `policy_server.py`.
Mock check (2026-09-19): `--mock --test` recorded 2 episodes; dataprep wrote a LeRobot v3.0 dataset of 99 frames at
30 fps with `observation.images.wrist` (480×640×3), `observation.state` (7) and `action` (7).
