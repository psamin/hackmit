"""Read-only check of the Damiao arm: find the USB-CAN adapter, find which motor IDs answer, print their state.
Never enables a motor (refresh queries only), so nothing moves.

    python arm/arm_probe.py                 # scan send IDs 1-8 under the common feedback-ID layouts
    python arm/arm_probe.py --ids 1,2,3,4,5 --recv-offset 0x10 --watch   # stream positions; move joints by hand
    python arm/arm_probe.py --mock          # no hardware: checks the script itself

Motors are DM-J4340P-2EC (24 V). Per the manual, a solid red LED = powered and disabled (normal),
solid green = enabled, blinking red = fault (8 overvoltage, 9 undervoltage, A overcurrent, B/C overtemp,
D communication loss, E overload).
"""
import argparse, sys, time

import can_motor_control as cmc
from can_motor_control import damiao

GS_USB = dict(vendor_id=0x1D50, product_id=0x606F)  # the only adapter type dimOS drives on macOS
# Feedback-ID layouts to try: dimOS/OpenYAM use send + 0x10; the factory default feedback ID (MST_ID) is 0.
LAYOUTS = {"send+0x10": lambda s: s + 0x10, "send": lambda s: s, "0 (factory)": lambda s: 0}


def open_robot(ids, recv, mock):
    bus = cmc.MockCanBus("probe") if mock else cmc.GsUsbBus(**GS_USB)
    motors = [cmc.MotorSpec(f"m{i}", damiao.MotorType.DM4340, i, recv(i)) for i in ids]
    robot = (cmc.Robot.builder().add_bus("arm", bus, damiao.DamiaoCodec())
             .add_arm("motors", bus="arm", motors=motors).build())
    robot.connect()
    return robot, bus


def poll(robot, rounds=20):
    for _ in range(rounds):
        robot.refresh()
        robot.tick(5000)
        time.sleep(0.01)


def answered(m):
    # A motor that never replied keeps its zeroed defaults; a live one reports a real MOSFET temperature.
    return m.temperature_mos > 0 or m.position != 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="1,2,3,4,5,6,7,8", help="send IDs to try")
    ap.add_argument("--recv-offset", type=lambda s: int(s, 0), default=None, help="use only feedback ID = send + this")
    ap.add_argument("--watch", action="store_true", help="keep printing positions (Ctrl-C to stop)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    ids = [int(i, 0) for i in args.ids.split(",")]

    if not args.mock:
        found = cmc.list_gs_usb_devices(**GS_USB)
        print(f"gs_usb adapters (1d50:606f): {[(d.index, d.serial_number) for d in found] or 'none'}")
        if not found:
            sys.exit("No gs_usb CAN adapter. Check the USB cable and dongle, and that the adapter runs candleLight/gs_usb firmware.")

    layouts = {"send+offset": lambda s: s + args.recv_offset} if args.recv_offset is not None else LAYOUTS
    best = None
    for name, recv in layouts.items():
        if name == "0 (factory)":  # every motor replies on ID 0, so the library can only route one at a time
            shared = []
            for i in ids:
                robot, _ = open_robot([i], recv, args.mock)
                poll(robot)
                shared += [f"m{i}"] if answered(robot["motors"][f"m{i}"]) else []
                robot.__exit__(None, None, None)
            print(f"feedback ID = {name:12s}: {len(shared)} motors answered {shared}")
            if shared and best is None:
                sys.exit("These motors all reply on feedback ID 0. Give each a unique MST_ID (register 0x07) "
                         "with Damiao's debugging tool, or ask Dimensional for their ID layout.")
            continue
        robot, bus = open_robot(ids, recv, args.mock)
        poll(robot)
        group = robot["motors"]
        live = [f"m{i}" for i in ids if answered(group[f"m{i}"])]
        rx = getattr(bus, "rx_received", "n/a")
        print(f"feedback ID = {name:12s}: {len(live)} motors answered {live} (frames received: {rx})")
        if live and (best is None or len(live) > len(best[2])):
            best = (name, robot, live)
        else:
            robot.__exit__(None, None, None)
    if best is None:
        sys.exit("No motor answered. Check 24 V power (LED solid red) and the CAN wires in the XT30 cable.")

    name, robot, live = best
    group = robot["motors"]
    print(f"\nUsing feedback ID = {name}. Motors stay disabled.")
    try:
        while True:
            poll(robot, rounds=3)
            print("  ".join(f"{n}: {group[n].position:+.3f} rad {group[n].temperature_mos}C"
                            f"{' FAULT ' + hex(group[n].fault) if group[n].fault else ''}"
                            f"{' ENABLED' if group[n].is_enabled else ''}" for n in live), flush=True)
            if not args.watch:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        robot.__exit__(None, None, None)


if __name__ == "__main__":
    main()
