"""Run the RunPod-hosted policy on OpenYAM: the demo-day script.

    PYTHONPATH=. python vla/run_policy.py --real --server https://<pod-id>-8000.proxy.runpod.net --camera-index 1
    PYTHONPATH=. python vla/run_policy.py --mock --server http://127.0.0.1:8011     # mock arm + synthetic camera

Terminal controls: p = preflight (moves nothing), s = start, x = stop, q = quit. Quitting disables the motors and the
arm has no brakes: support it first. Other processes (the voice loop) use vla/arm_client.py, served on 127.0.0.1:8020.
"""
import argparse, functools, json, socketserver, sys, threading, time
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


def watch_for_grasp(control, policy, args):
    """Hand the bottle over the moment the policy closes on it, then stand down.

    The policy is trained on the pick alone and its episodes end when the gripper shuts, so it has nothing sensible
    to say after that. A thread watches the gripper; once it has been closed for GRASP_SETTLE_S - long enough that a
    single noisy sample cannot trigger it - the rollout stops and the taught preset plays: lift, swing round, hold
    the bottle out, open. Returning to home is the caller's next rollout, which drives there anyway.
    """
    import json as _json
    import threading
    import time as _time

    from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
    from dimos.robot.manipulators.openyam.config import OPENYAM_JOINTS

    from vla.scripted_demos import build_trajectory

    poses = _json.load(open(args.handover))["poses"]
    gripper_joint = OPENYAM_JOINTS[-1]
    GRASP_SETTLE_S = 0.8      # the jaws must be this long without moving before the lift starts
    GRIPPER_STILL = 0.02      # rad of travel over that window that still counts as "stopped"
    SHUT_ON_AIR = 0.05        # fully shut means it missed the bottle; lifting nothing is worse than not lifting
    done = threading.Event()

    def play(motion, speed=None):
        start = [control.get_joint_positions()[n] for n in OPENYAM_JOINTS]
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, start, motion, speed or args.speed, 0.0,
                                                __import__("numpy").random.default_rng(0))
        if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
            _time.sleep(duration + 0.5)
            return True
        return False

    def say(text_or_command):
        """Speak through the voice agent when --voice-url is set, otherwise run it as a shell command."""
        import subprocess
        import urllib.request

        if not args.voice_url:
            subprocess.run(text_or_command, shell=True, timeout=20, check=False)
            return
        body = _json.dumps({"type": "speak", "text": text_or_command}).encode()
        request = urllib.request.Request(args.voice_url, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        try:  # the agent not being up must never strand the arm holding a bottle
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
        except Exception as exc:
            print(f"voice agent unreachable ({exc}), carrying on", flush=True)

    def run():
        """Lift and turn holding the bottle, present it, cue the person, wait for their hand, then open."""

        policy.stop_rollout()
        hold = [p for p in poses if not p["gripper"]]
        release = [p for p in poses if p["gripper"]]
        if hold:
            play(hold)
        print(f"presenting the bottle for {args.present_s:.0f}s", flush=True)
        _time.sleep(args.present_s)
        if args.announce:  # the voice agent speaks while the arm holds still; the release waits for it to finish
            print(f"announcing: {args.announce}", flush=True)
            say(args.announce)
        if args.wait_for_hand:
            from vla.hand_release import wait_for_hand

            default_prompts = (["Please take your pills.", "Take your time, I have got it."] if args.voice_url
                               else ['say "please take your pills"', 'say "take your time, I have got it"'])
            prompts = args.nag or default_prompts

            def nag(n):
                say(prompts[n % len(prompts)])

            wait_for_hand(camera_index=args.camera_index, timeout_s=args.catch_s, nag_s=args.nag_s, on_nag=nag)
        else:
            print(f"waiting {args.catch_s:.0f}s for a hand underneath", flush=True)
            _time.sleep(args.catch_s)
        if release:
            play(release)
        print("handover done - bottle released", flush=True)
        if args.home:  # back to where the next rollout starts, gently, with the bottle already gone
            _time.sleep(args.return_after_s)
            print(f"returning to home at {args.return_speed} rad/s", flush=True)
            play([_json.load(open(args.home))["poses"][0]], speed=args.return_speed)
            print("home", flush=True)

    def watch():
        """Fire only once the jaws have stopped moving, not the moment they pass the threshold.

        The gripper sweeps through every value on its way shut, so a plain threshold triggers mid-close and the arm
        lifts before it has the bottle. Wait until it is both below the threshold and no longer changing: that means
        it has finished travelling, either stalled on the bottle or shut on air.
        """
        history = []
        while not done.is_set():
            _time.sleep(0.1)
            if not policy.rollout_status().get("active"):
                history.clear()
                continue
            value = control.get_joint_positions().get(gripper_joint)
            if value is None or value > args.grip_closed:
                history.clear()
                continue
            history.append(value)
            if len(history) < int(GRASP_SETTLE_S / 0.1):
                continue
            recent = history[-int(GRASP_SETTLE_S / 0.1):]
            if max(recent) - min(recent) > GRIPPER_STILL:  # still travelling
                continue
            if value < SHUT_ON_AIR:
                print(f"gripper shut to {value:.2f} - nothing in the jaws, not handing over", flush=True)
                history.clear()
                continue
            print(f"gripper settled at {value:.2f} holding the bottle - handing over", flush=True)
            history.clear()
            run()

    threading.Thread(target=watch, daemon=True).start()
    return run


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
    ap.add_argument("--home", default=None,
                    help="spot file whose first pose the arm drives to before each rollout, e.g. vla/spots/left_01.json")
    ap.add_argument("--speed", type=float, default=0.3, help="--handover and --home: rad/s cap for the preset motion")
    ap.add_argument("--handover", default=None,
                    help="preset poses to play the moment the policy closes the gripper, e.g. vla/spots/handover.json")
    ap.add_argument("--present-s", type=float, default=2.0,
                    help="--handover: seconds to hold the bottle out before the announcement")
    ap.add_argument("--announce", default=None,
                    help="--handover: shell command run once the bottle is presented, e.g. a voice agent saying "
                         "'catch, grab the bottle'. The release waits for it to finish, then --catch-s longer.")
    ap.add_argument("--catch-s", type=float, default=2.0,
                    help="--handover: seconds after the announcement before the gripper opens, to get a hand under it")
    ap.add_argument("--wait-for-hand", action="store_true",
                    help="--handover: hold the bottle until the camera sees a hand, nagging every --nag-s. "
                         "--catch-s becomes the timeout it releases on anyway.")
    ap.add_argument("--nag-s", type=float, default=3.0,
                    help="--wait-for-hand: seconds between spoken prompts while waiting for a hand")
    ap.add_argument("--voice-url", default=None,
                    help="the voice agent's push endpoint, e.g. http://127.0.0.1:8000/api/push. With it, "
                         "--announce and --nag are spoken by the agent as plain text rather than run as commands.")
    ap.add_argument("--return-after-s", type=float, default=3.0,
                    help="--handover: seconds to wait after releasing before the arm goes back to home")
    ap.add_argument("--return-speed", type=float, default=0.15,
                    help="--handover: rad/s for the trip back to home; slower than --speed, nothing is being carried")
    ap.add_argument("--nag", action="append", default=None,
                    help="--wait-for-hand: a command to run for each prompt, repeatable; cycles through them")
    ap.add_argument("--grip-closed", type=float, default=0.85,
                    help="--handover: gripper below this counts as closed on the bottle")
    ap.add_argument("--replay-episode", type=int, default=None,
                    help="--mock: feed this dataset episode's real frames instead of a grey image")
    ap.add_argument("--dataset", default="openyam_dataset", help="--replay-episode: the LeRobot dataset directory")
    ap.add_argument("--viz", action="store_true",
                    help="add dimOS's Viser 3D view of the arm, served on http://127.0.0.1:8080")
    args = ap.parse_args()

    from dimos.core.global_config import global_config

    # dimOS starts its workers with multiprocessing's forkserver. On macOS that child cannot initialise Metal, and
    # dimos/models/base.py calls torch.backends.mps.is_available() during import, so the worker dies in
    # MPSLibrary::MPSKey_Compile with SIGSEGV and the coordinator only sees a broken pipe. A spawned child starts
    # clean, so Metal initialises normally there.
    if sys.platform == "darwin":
        import multiprocessing

        from dimos.core.coordination import python_worker

        python_worker.get_forkserver_context = lambda: multiprocessing.get_context("spawn")

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

    if args.mock and args.replay_episode is not None:
        from vla.replay_camera import ReplayCamera

        camera = ReplayCamera.blueprint(dataset=args.dataset, episode=args.replay_episode)
    elif args.mock:
        from vla.sim_test import SyntheticCamera

        camera = SyntheticCamera.blueprint()  # a flat grey frame: proves the plumbing, tells the policy nothing
    else:
        from dimos.hardware.sensors.camera.module import CameraModule
        from dimos.hardware.sensors.camera.webcam import Webcam

        camera = CameraModule.blueprint(  # a factory: dimOS builds the webcam inside its worker process
            hardware=functools.partial(Webcam, camera_index=args.camera_index, width=640, height=480, fps=30.0))

    viz = []
    if args.viz:  # the same planner the openyam-planner-coordinator blueprint uses, purely to get its Viser view
        from dimos.robot.manipulators.common.blueprints import planner
        from dimos.robot.manipulators.openyam.config import make_openyam_model_config

        viz = [planner(model=make_openyam_model_config(), visualization={"backend": "viser"})]
    blueprint = autoconnect(
        *viz,
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
    control = coordinator.get_instance(ControlCoordinator)
    handover = watch_for_grasp(control, policy, args) if args.handover else (lambda: None)

    def start():
        """Drive to the taught home first: every training episode began there, so the policy expects it."""
        if args.home:
            import json as _json

            import numpy as _np

            from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus

            from vla.scripted_demos import build_trajectory
            home = _json.load(open(args.home))["poses"][0]
            print(f"driving to home from {args.home} before the rollout", flush=True)
            begin = [control.get_joint_positions()[n] for n in OPENYAM_JOINTS]
            trajectory, duration = build_trajectory(OPENYAM_JOINTS, begin, [home], args.speed, 0.0,
                                                    _np.random.default_rng(0))
            if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
                time.sleep(duration + 0.5)
        return policy.start_rollout()

    actions = {"p": policy.preflight_rollout, "s": start,
               "x": policy.stop_rollout, "h": lambda: (handover(), policy.rollout_status())[1]}
    if args.control_port:
        serve_control({"preflight": policy.preflight_rollout, "start": start,
                       "stop": policy.stop_rollout,
                       "handover": lambda: (handover(), policy.rollout_status())[1]},
                      policy.rollout_status, args.control_port)
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
