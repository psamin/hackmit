"""Serve a LeRobot policy over HTTP so the laptop driving the arm can run it on a RunPod GPU.

    python vla/policy_server.py --policy-path <checkpoint dir or HF repo id>     # on RunPod; port 8000
    python vla/policy_server.py --dummy --joints 7                               # no checkpoint: holds the pose

GET  /info  -> {"policy", "joint_count", "n_action_steps", "action_min", "action_max", "image_shape"}
POST /act   <- {"image_jpeg_b64", "state": [..], "task"}  -> {"actions": [[..] x n_action_steps], "infer_s"}
POST /reset -> clears the policy's action queue (call at the start of each rollout)

Loading and inference follow dimOS's native/python/lerobot runtime, so checkpoints built from
`dimos dataprep` datasets (observation.images.wrist, observation.state, action) load unchanged.
"""
import argparse, base64, json, socketserver, threading, time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

IMAGE, STATE, ACTION = "observation.images.wrist", "observation.state", "action"


class DummyPolicy:
    """Returns the current pose for every step: exercises the whole loop without a checkpoint."""

    def __init__(self, joints, steps, height, width, gripper=None):
        self.gripper = gripper  # if set, the last joint (the gripper) is commanded to this value
        self.info = {"policy": "dummy", "joint_count": joints, "n_action_steps": steps,
                     "action_min": [-np.pi] * joints, "action_max": [np.pi] * joints, "image_shape": [3, height, width]}

    def reset(self):
        pass

    def act(self, image, state, task):
        actions = np.repeat(state[None, :], self.info["n_action_steps"], axis=0)
        if self.gripper is not None:
            actions[:, -1] = self.gripper
        return actions


class LeRobotPolicy:
    def __init__(self, path, device, robot_type):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        cfg = PreTrainedConfig.from_pretrained(path)
        if device:
            cfg.device = device
        self.torch, self.device, self.robot_type, self.use_amp = torch, torch.device(cfg.device), robot_type, bool(cfg.use_amp)
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        lower, upper = action_bounds(self.post)
        self.info = {"policy": cfg.type, "joint_count": int(cfg.output_features[ACTION].shape[0]),
                     "n_action_steps": int(cfg.n_action_steps), "action_min": lower, "action_max": upper,
                     "image_shape": list(cfg.input_features[IMAGE].shape)}

    def reset(self):
        self.policy.reset()

    def act(self, image, state, task):
        from lerobot.policies.utils import prepare_observation_for_inference

        torch = self.torch
        amp = torch.autocast(device_type="cuda") if self.device.type == "cuda" and self.use_amp else nullcontext()
        with torch.inference_mode(), amp:
            obs = prepare_observation_for_inference({IMAGE: image, STATE: state}, self.device, task=task,
                                                    robot_type=self.robot_type)
            chunk = self.post(self.policy.predict_action_chunk(self.pre(obs)))
        return np.asarray(chunk.to("cpu").numpy(), dtype=np.float32)[0, : self.info["n_action_steps"]]


def action_bounds(postprocessor):
    """The checkpoint's recorded action range; the laptop clips every action to it."""
    for step in postprocessor.steps:
        s = step.state_dict()
        if "action.min" in s and "action.max" in s:
            return s["action.min"].cpu().numpy().tolist(), s["action.max"].cpu().numpy().tolist()
    raise ValueError("policy postprocessor has no action min/max statistics")


def serve(policy, host, port):
    lock = threading.Lock()  # one inference at a time: policies keep an internal action queue
    joints, (_, height, width) = policy.info["joint_count"], policy.info["image_shape"]

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, policy.info) if self.path == "/info" else self.reply(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/reset":
                    with lock:
                        policy.reset()
                    return self.reply(200, {"ok": True})
                if self.path != "/act":
                    return self.reply(404, {"error": "unknown path"})
                jpeg = np.frombuffer(base64.b64decode(req["image_jpeg_b64"]), np.uint8)
                image = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                state = np.asarray(req["state"], np.float32)
                if image.shape != (height, width, 3) or state.shape != (joints,):
                    return self.reply(400, {"error": f"expected image {(height, width, 3)} and {joints} joints, "
                                                     f"got {image.shape} and {state.shape}"})
                t0 = time.perf_counter()
                with lock:
                    actions = policy.act(image, state, req.get("task", ""))
                self.reply(200, {"actions": actions.tolist(), "infer_s": round(time.perf_counter() - t0, 4)})
            except Exception as exc:
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        def server_bind(self):  # skip HTTPServer's reverse-DNS lookup, which can hang for minutes on some networks
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = host, port

    server = Server((host, port), Handler)
    print(f"serving {policy.info['policy']} on {host}:{port}: {policy.info}", flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", help="LeRobot checkpoint directory or Hugging Face repo id")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the checkpoint's)")
    ap.add_argument("--robot-type", default="openyam")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 on RunPod; 127.0.0.1 for local tests")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dummy", action="store_true", help="hold the current pose instead of loading a checkpoint")
    ap.add_argument("--joints", type=int, default=7, help="--dummy only: OpenYAM is 6 joints + gripper")
    ap.add_argument("--steps", type=int, default=10, help="--dummy only: actions per chunk")
    ap.add_argument("--image-size", default="480x640", help="--dummy only: HxW the laptop must send")
    ap.add_argument("--dummy-gripper", type=float, default=None, help="--dummy only: command the gripper to this value")
    args = ap.parse_args()
    if args.dummy:
        h, w = map(int, args.image_size.split("x"))
        policy = DummyPolicy(args.joints, args.steps, h, w, args.dummy_gripper)
    elif args.policy_path:
        policy = LeRobotPolicy(args.policy_path, args.device, args.robot_type)
    else:
        ap.error("give --policy-path or --dummy")
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
