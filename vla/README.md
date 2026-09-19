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
