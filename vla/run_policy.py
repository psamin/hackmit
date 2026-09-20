"""Run the RunPod-hosted policy on OpenYAM: the demo-day script.

    PYTHONPATH=. python vla/run_policy.py --real --server https://<pod-id>-8000.proxy.runpod.net --camera-index 1
    PYTHONPATH=. python vla/run_policy.py --mock --server http://127.0.0.1:8011     # mock arm + synthetic camera

Terminal controls: p = preflight (moves nothing), s = start, x = stop, q = quit. Quitting disables the motors and the
arm has no brakes: support it first. Other processes (the voice loop) use vla/arm_client.py, served on 127.0.0.1:8020.
"""
import argparse, functools, json, socketserver, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve_control(actions, status, port):
    """Localhost-only HTTP control: POST /preflight, /start, /stop, /handover, /open, /close, /home,
    and GET /status."""

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


def post_stage(voice_url, name, detail=""):
    """Tell the voice agent where the arm is, so it can answer "what is it doing" at any moment.

    A separate event type from "speak": the agent should know the stage without saying it out loud.
    """
    import json as _json
    import urllib.request

    print(f"[stage] {name} {detail}".rstrip(), flush=True)
    if not voice_url:
        return
    body = _json.dumps({"type": "arm_stage", "stage": name, "detail": detail}).encode()
    req = urllib.request.Request(voice_url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            r.read()
    except Exception:
        pass  # the agent not listening must never hold up the arm


def watch_for_grasp(control, policy, args):
    """Give the policy a fixed window to reach the bottle, then run the hand-over as a fixed sequence.

    The policy is trained on the pick alone and has nothing sensible to say after it, so everything past the grasp
    is scripted: shut the jaws, turn round, announce, pause, open. Nothing here reads the gripper to decide what
    happened. An earlier version did, and refused to hand over whenever the jaws closed past a "shut on air"
    threshold - which is exactly what a narrow bottle looks like, so the arm gripped the pills and then stood
    still. A fixed window cannot make that mistake, and 'h' fires the same sequence by hand at any moment.
    """
    import json as _json
    import threading
    import time as _time

    from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus
    from dimos.robot.manipulators.openyam.config import OPENYAM_JOINTS

    from vla.scripted_demos import build_trajectory, smooth_corners

    poses = _json.load(open(args.handover))["poses"]
    gripper_joint = OPENYAM_JOINTS[-1]
    running = threading.Event()  # set while a rollout is going; no RPC needed to ask the module
    TICK_S = 0.3              # dimOS calls are cross-process RPC; polling faster than this starves the rollout
    SETTLE_S = 0.6            # the jaws must be shut and no longer moving for this long before the hand-over
    GRIPPER_STILL = 0.02      # rad of travel across that window that still counts as stopped
    done = threading.Event()

    TAIL_S = 0.15  # a margin so the arm has certainly finished, not half a second of standing about

    def play(motion, speed=None):
        start = [control.get_joint_positions()[n] for n in OPENYAM_JOINTS]
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, start, motion, speed or args.speed, 0.0,
                                                __import__("numpy").random.default_rng(0))
        trajectory = smooth_corners(trajectory, blend_s=args.blend_s)
        if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
            _time.sleep(duration + TAIL_S)
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
        """Shut on the bottle, turn round, announce, pause, drop. Fixed order, no decisions."""

        running.clear()
        policy.stop_rollout()
        # Close deliberately rather than trusting whatever the policy left the jaws at: its last commanded
        # gripper value is wherever the chunk happened to end, which is not necessarily gripping anything.
        hold = [p for p in poses if not p["gripper"]]
        release = [p for p in poses if p["gripper"]]
        post_stage(args.voice_url, "grasped", "holding the bottle")
        try:
            # Shut the jaws and carry straight on into the lift and the turn as ONE trajectory. As three separate
            # calls the arm stopped dead between them: each one re-read the joints, re-planned, and padded the end.
            # The gripper still gets its full close-and-settle - that is inside the trajectory, not between them.
            print("closing on the bottle, then turning round", flush=True)
            here = control.get_joint_positions()
            play([{"name": "grasp", "q": [here[n] for n in OPENYAM_JOINTS[:-1]], "gripper": 0.0}] + hold)
            post_stage(args.voice_url, "presenting", "turned to the person, holding the bottle out")
            print(f"presenting the bottle for {args.present_s:.1f}s", flush=True)
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

                post_stage(args.voice_url, "waiting_for_hand", "gripper still closed, watching for a hand")
                if args.hand_url:
                    from vla.hand_release import wait_for_hand_remote
                    saw = wait_for_hand_remote(args.hand_url, timeout_s=args.catch_s, nag_s=args.nag_s, on_nag=nag)
                else:
                    saw = wait_for_hand(camera_index=args.camera_index, timeout_s=args.catch_s,
                                        nag_s=args.nag_s, on_nag=nag)
                post_stage(args.voice_url, "hand_seen" if saw else "hand_timeout", "releasing")
            else:
                print(f"waiting {args.catch_s:.1f}s before opening the jaws", flush=True)
                _time.sleep(args.catch_s)
            if release:
                play(release)
            post_stage(args.voice_url, "released", "the bottle is in their hand")
            print("handover done - bottle released", flush=True)
        finally:
            # Always end where the next run starts, even if the hand-over threw halfway. Otherwise one bad
            # cycle leaves the arm parked mid-turn and every later run begins from somewhere the policy has
            # never seen. 'r' does the same thing on demand.
            if args.home:
                try:
                    _time.sleep(args.return_after_s)
                    post_stage(args.voice_url, "returning", "going back to home")
                    print(f"returning to home at {args.return_speed} rad/s", flush=True)
                    play([_json.load(open(args.home))["poses"][0]], speed=args.return_speed)
                    post_stage(args.voice_url, "idle", "back at home, ready")
                    print("home", flush=True)
                except Exception as exc:
                    post_stage(args.voice_url, "stuck", f"could not get back to home: {exc}")
                    print(f"could not get back to home: {exc} - press 'r' or POST /home", flush=True)

    def watch():
        """Hand over once the jaws have shut on something, or when the grasp window runs out - whichever first.

        Waiting for the gripper is what makes the pick reliable: the policy takes as long as it takes to reach
        the bottle. What it must NOT do is judge what it is holding - an earlier version refused to hand over
        whenever the jaws closed past a "shut on air" threshold, which is just what a narrow bottle looks like,
        so it gripped the pills and stood there. Any settled closure counts now. --grasp-s is only a backstop so
        a pick that never closes still ends somewhere predictable.
        """
        while not done.is_set():
            _time.sleep(TICK_S)
            if not running.is_set():
                continue
            began, history, why = _time.time(), [], None
            while running.is_set() and not done.is_set():
                _time.sleep(TICK_S)
                if _time.time() - began >= args.grasp_s:
                    why = f"grasp window of {args.grasp_s:.0f}s is up"
                    break
                try:
                    joints = control.get_joint_positions()
                except Exception as exc:
                    print(f"grasp watcher: {exc}", flush=True)
                    history.clear()
                    continue
                value = joints.get(gripper_joint) if joints else None
                if value is None or value > args.grip_closed:
                    history.clear()
                    continue
                history.append(value)
                need = max(2, round(SETTLE_S / TICK_S))
                if len(history) >= need and max(history[-need:]) - min(history[-need:]) <= GRIPPER_STILL:
                    why = f"jaws settled at {value:.2f}"
                    break
            if not running.is_set() or done.is_set():
                continue  # stopped, or handed over by hand, in the meantime
            print(f"{why} - handing over", flush=True)
            try:
                run()
            except Exception as exc:  # a failed hand-over must not take the watcher down with it
                print(f"hand-over failed: {exc}", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    return run, running


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
    ap.add_argument("--blend-s", type=float, default=0.4,
                    help="seconds either side of a corner over which one leg's velocity gives way to the next's. "
                         "0 turns blending off and the arm hesitates at each taught pose.")
    ap.add_argument("--handover", default=None,
                    help="preset poses played after the grasp window, e.g. vla/spots/handover.json")
    ap.add_argument("--grasp-s", type=float, default=35.0,
                    help="--handover: backstop only. The hand-over normally starts the moment the jaws shut on "
                         "the bottle; this caps how long a pick that never closes can run. 'h' runs it now.")
    ap.add_argument("--grip-closed", type=float, default=0.85,
                    help="--handover: a gripper reading below this counts as shut on something")
    ap.add_argument("--present-s", type=float, default=0.5,
                    help="--handover: seconds to hold still after turning round, before the announcement")
    ap.add_argument("--announce", default=None,
                    help="--handover: shell command run once the bottle is presented, e.g. a voice agent saying "
                         "'catch, grab the bottle'. The release waits for it to finish, then --catch-s longer.")
    ap.add_argument("--catch-s", type=float, default=1.0,
                    help="--handover: seconds after the announcement before the jaws open. With --wait-for-hand "
                         "they open sooner if the camera sees a hand; they never stay shut longer than this.")
    ap.add_argument("--hand-url", default=None,
                    help="poll this for hand detection instead of opening a camera here, e.g. "
                         "http://127.0.0.1:8000/api/hand-visible - the phone sees the catch, the wrist camera "
                         "looks past it")
    ap.add_argument("--wait-for-hand", action="store_true",
                    help="--handover: hold the bottle until the camera sees a hand, nagging every --nag-s. "
                         "--catch-s becomes the timeout it releases on anyway.")
    ap.add_argument("--nag-s", type=float, default=7.0,  # longer than --catch-s by default, so it stays quiet
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
    def set_gripper(value, label):
        """Drive the gripper alone, leaving the arm where it is. 1 is open, 0 is shut.

        The voice agent needs this for "let go" and "hold on to it" - the person may want the bottle released
        before the detector is convinced, or the grip kept while they get a better hold.
        """
        import numpy as _np

        from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus

        from vla.scripted_demos import build_trajectory
        here = control.get_joint_positions()
        arm_now = [here[n] for n in OPENYAM_JOINTS[:-1]]
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, [here[n] for n in OPENYAM_JOINTS],
                                                [{"name": label, "q": arm_now, "gripper": value}],
                                                args.speed, 0.0, _np.random.default_rng(0))
        accepted = control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED
        if accepted:
            time.sleep(duration + 0.3)
        post_stage(args.voice_url, "gripper_open" if value else "gripper_closed", label)
        return {"gripper": label, "accepted": accepted}

    handover, rollout_running = (watch_for_grasp(control, policy, args) if args.handover
                                 else (lambda: None, threading.Event()))

    starting = threading.Event()

    def go_home(speed):
        """Drive to the taught home pose. Returns False if there is no --home to go to."""
        if not args.home:
            return False
        import json as _json

        import numpy as _np

        from dimos.control.tasks.trajectory_task.trajectory_task import TrajectoryExecutionStatus

        from vla.scripted_demos import build_trajectory
        home = _json.load(open(args.home))["poses"][0]
        print(f"driving to home from {args.home} at {speed} rad/s", flush=True)
        post_stage(args.voice_url, "homing", "moving to the start position")
        begin = [control.get_joint_positions()[n] for n in OPENYAM_JOINTS]
        trajectory, duration = build_trajectory(OPENYAM_JOINTS, begin, [home], speed, 0.0,
                                                _np.random.default_rng(0))
        if control.execute_trajectory(trajectory).status is TrajectoryExecutionStatus.ACCEPTED:
            time.sleep(duration + 0.5)
        return True

    def reset_home():
        """Put the arm back at home on demand, whatever it was doing.

        Stops the rollout first: a policy still writing targets would fight the trajectory the whole way.
        Goes at --return-speed rather than --speed, because a reset is a manual, unhurried move and there is
        nothing in the jaws to keep level.
        """
        rollout_running.clear()
        starting.clear()
        try:
            policy.stop_rollout()
        except Exception as exc:
            print(f"reset: could not stop the rollout ({exc})", flush=True)
        went = go_home(args.return_speed)
        post_stage(args.voice_url, "idle", "back at home, ready" if went else "no home pose configured")
        if not went:
            print("no --home pose given, so there is nowhere to reset to", flush=True)
        return {"home": went, "spot": args.home}

    def drive_home_then_roll():
        """Drive to the taught home first: every training episode began there, so the policy expects it."""
        try:
            go_home(args.speed)
            policy.start_rollout()
            rollout_running.set()
            post_stage(args.voice_url, "reaching", "going for the bottle")
        except Exception as exc:  # a thread now, so a failure here would otherwise be silent
            print(f"start failed: {exc}", flush=True)
        finally:
            starting.clear()

    def start():
        """Kick the rollout off and answer at once.

        The drive home takes seconds and the caller is a voice agent on a short HTTP timeout: blocking here made
        it give up and tell the person the arm was unreachable while the arm was in fact moving. Progress goes out
        on the stage channel instead.
        """
        status = policy.rollout_status()
        if not status.get("active") and not starting.is_set():
            starting.set()
            threading.Thread(target=drive_home_then_roll, daemon=True).start()
        return {**status, "starting": True}

    actions = {"p": policy.preflight_rollout, "s": start,
               "x": policy.stop_rollout, "h": lambda: (handover(), policy.rollout_status())[1],
               "r": lambda: (reset_home(), policy.rollout_status())[1]}
    if args.control_port:
        serve_control({"preflight": policy.preflight_rollout, "start": start,
                       "stop": lambda: (rollout_running.clear(), starting.clear(), policy.stop_rollout())[2],
                       "handover": lambda: (handover(), policy.rollout_status())[1],
                       "open": lambda: set_gripper(1.0, "open"),
                       "close": lambda: set_gripper(0.0, "closed"),
                       "home": reset_home},
                      policy.rollout_status, args.control_port)
        print(f"arm control on http://127.0.0.1:{args.control_port}", flush=True)
    try:
        for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("p/s/x/r/q: "), None):
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
