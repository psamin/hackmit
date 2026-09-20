"""Disable every motor over CAN, so the arm goes back to solid red without power-cycling the 24 V.

    python arm/disarm.py            # disable all 7, then report each LED state
    python arm/disarm.py --status   # report only, change nothing

Solid red is powered and disabled; solid green is enabled and holding position. dimOS's own shutdown asks for this
and logs "Hardware openyam deactivate returned False" when it does not take, which is why this exists separately.

THE ARM GOES LIMP THE INSTANT IT IS DISABLED. Hold it, or make sure it is resting somewhere it can fall safely.
"""
import argparse, sys, time

import can_motor_control as cmc
from can_motor_control import damiao

GS_USB = dict(vendor_id=0x1D50, product_id=0x606F)
ARM = tuple(range(1, 7))  # joints 1-6; the gripper is 8, feedback on send + 0x10
GRIPPER = 8


def build():
    types = [damiao.MotorType.DM4340] * 3 + [damiao.MotorType.DM4310] * 3
    motors = [cmc.MotorSpec(f"yam_joint{i}", t, i, i + 0x10) for i, t in zip(ARM, types)]
    motors.append(cmc.MotorSpec("yam_gripper", damiao.MotorType.DM4310, GRIPPER, GRIPPER + 0x10))
    bus = cmc.GsUsbBus(**GS_USB)
    robot = (cmc.Robot.builder().add_bus("openyam", bus, damiao.DamiaoCodec())
             .add_arm("arm", bus="openyam", motors=motors).build())
    robot.connect()
    return robot


def report(robot):
    for _ in range(5):
        robot.refresh()
        robot.tick(5000)
        time.sleep(0.01)
    arm, enabled = robot["arm"], []
    for spec in [f"yam_joint{i}" for i in ARM] + ["yam_gripper"]:
        motor = arm[spec]
        led = "GREEN (enabled)" if motor.is_enabled else "red (disabled)"
        fault = f"  FAULT {hex(motor.fault)}" if motor.fault else ""
        print(f"  {spec:<12} {led}{fault}")
        if motor.is_enabled:
            enabled.append(spec)
    return enabled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report the LED state without changing it")
    args = ap.parse_args()

    if not cmc.list_gs_usb_devices(**GS_USB):
        sys.exit("No gs_usb CAN adapter. Check the USB cable and the dongle.")
    robot = build()
    try:
        if args.status:
            report(robot)
            return
        print("Disabling every motor - THE ARM WILL GO LIMP. Hold it if it is not resting.")
        robot["arm"].disable_all()
        time.sleep(0.3)
        still_on = report(robot)
    finally:
        robot.__exit__(None, None, None)
    if still_on:
        sys.exit(f"\n{len(still_on)} motor(s) still enabled: {', '.join(still_on)}. "
                 "Power-cycle the 24 V instead.")
    print("\nAll motors disabled - every LED should be solid red.")


if __name__ == "__main__":
    main()
