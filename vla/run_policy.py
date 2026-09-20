"""Run the RunPod-hosted policy on OpenYAM: the demo-day script.

    PYTHONPATH=. python vla/run_policy.py --real --server https://<pod-id>-8000.proxy.runpod.net --camera-index 1
    PYTHONPATH=. python vla/run_policy.py --mock --server http://127.0.0.1:8011     # mock arm + synthetic camera

Terminal controls: p = preflight (moves nothing), s = start, x = stop, q = quit. Quitting disables the motors and the
arm has no brakes: support it first. Other processes (the voice loop) use vla/arm_client.py, served on 127.0.0.1:8020.
"""
import argparse, functools, json, socketserver, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve_control(actions, status, port):
    """Localhost-only HTTP control: POST /preflight, /start, /stop and GET /status, each returning the status."""

    class Handler(BaseHTTPRequestHandler):
        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(status()) if self.path == "/status" else self.send_error(404)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            action = actions.get(self.path.strip("/"))
            self.reply(action()) if action else self.send_error(404)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        def server_bind(self):  # skip HTTPServer's reverse-DNS lookup, which can hang for minutes on some networks
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", port

    server = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true", help="drive the real arm (energizes the motors)")
    mode.add_argument("--mock", action="store_true", help="dimOS's in-memory OpenYAM adapter")
    ap.add_argument("--server", required=True, help="policy_server.py URL")
    ap.add_argument("--task", default="pick up the pill bottle")
    ap.add_argument("--fps", type=float, default=30.0, help="the training dataset's rate")
    ap.add_argument("--camera-index", type=int, default=0, help="--real: OpenCV index of the arm camera")
    ap.add_argument("--control-port", type=int, default=8020, help="localhost control for arm_client.py; 0 = off")
    args = ap.parse_args()

    from dimos.core.global_config import global_config

    if args.mock:
        global_config.simulation = "mock"  # before building OpenYAM hardware: selects the in-memory adapter
    from dimos.control.coordinator import ControlCoordinator, TaskConfig
    from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from dimos.robot.manipulators.openyam.config import (OPENYAM_GRIPPER_JOINT, OPENYAM_HARDWARE_ID, OPENYAM_JOINTS,
                                                         openyam_hardware)

    from vla.remote_policy import RemotePolicyModule

    hardware = openyam_hardware()
    if args.mock and hardware.adapter_type != "mock_whole_body":
        raise SystemExit(f"refusing to run: expected the mock adapter, got {hardware.adapter_type!r}")
    if args.real:
        print(f"REAL ARM ({hardware.adapter_type}): the motors will be energized. Clear the workspace, "
              "keep the power switch in reach.")
        if input("Type 'go' to continue: ").strip() != "go":
            raise SystemExit("aborted")

    if args.mock:
        from vla.sim_test import SyntheticCamera

        camera = SyntheticCamera.blueprint()
    else:
        from dimos.hardware.sensors.camera.module import CameraModule
        from dimos.hardware.sensors.camera.webcam import Webcam

        camera = CameraModule.blueprint(  # a factory: dimOS builds the webcam inside its worker process
            hardware=functools.partial(Webcam, camera_index=args.camera_index, width=640, height=480, fps=30.0))

    blueprint = autoconnect(
        ControlCoordinator.blueprint(hardware=[hardware], tasks=[
            joint_trajectory_task(OPENYAM_JOINTS),
            TaskConfig(name=f"{OPENYAM_HARDWARE_ID}_gripper", type="gripper", joint_names=[OPENYAM_GRIPPER_JOINT],
                       priority=20),
        ]),
        camera,
        RemotePolicyModule.blueprint(server_url=args.server, task=args.task, joint_names=list(OPENYAM_JOINTS),
                                     fps=args.fps),
    )
    coordinator = ModuleCoordinator.build(blueprint)
    policy = coordinator.get_instance(RemotePolicyModule)
    actions = {"p": policy.preflight_rollout, "s": policy.start_rollout, "x": policy.stop_rollout}
    if args.control_port:
        serve_control({"preflight": policy.preflight_rollout, "start": policy.start_rollout,
                       "stop": policy.stop_rollout}, policy.rollout_status, args.control_port)
        print(f"arm control on http://127.0.0.1:{args.control_port}", flush=True)
    try:
        for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("p/s/x/q: "), None):
            key = line.strip().lower()
            if key == "q":
                break
            status = actions[key]() if key in actions else policy.rollout_status()
            print({k: status[k] for k in ("active", "policy_ready", "observations_ready", "chunks_accepted", "last_error")},
                  flush=True)
    finally:
        policy.stop_rollout()
        coordinator.stop()


if __name__ == "__main__":
    main()
