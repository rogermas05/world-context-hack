"""
One-time interactive calibration for an SO-101 arm (leader or follower).

Run this YOURSELF in your own terminal -- it will ask you to physically move
the arm to the middle of its range, press Enter, then move each joint through
its full range of motion and press Enter again. That's a physical step, so it
can't be automated or run on your behalf.

    python hardware/calibrate.py --type leader --port /dev/tty.usbmodemXXXX --id hackathon_l1_right
    python hardware/calibrate.py --type follower --port /dev/tty.usbmodemXXXX --id hackathon_f1_head

Saves a calibration file lerobot will reuse automatically on future connects
(keyed by --id), so you only need to do this once per physical arm.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["leader", "follower"], required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()

    if args.type == "leader":
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        arm = SO101Leader(SO101LeaderConfig(port=args.port, id=args.id))
        read = arm.get_action
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        arm = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
        read = arm.get_observation

    arm.connect(calibrate=True)
    print("Calibrated and connected. Current position:", read())
    arm.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
