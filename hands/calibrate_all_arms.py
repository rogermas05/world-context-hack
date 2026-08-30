"""Interactive one-time calibration for the three-arm rig.

Calibration is a PHYSICAL step -- lerobot asks you to move each arm to the
middle of its range, then sweep every joint through its full travel. It cannot
be done on your behalf, so run this yourself:

    python hands/calibrate_all_arms.py            # all three, in turn
    python hands/calibrate_all_arms.py --only left

Calibration is stored per arm id under
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json and is
reused automatically afterwards, so you only do this once per physical arm
(unless you rebuild one or the servos get re-homed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

from arm_config import ARMS  # noqa: E402


def calibrate(role: str) -> None:
    port, arm_id = ARMS[role]
    print("=" * 70)
    print(f"CALIBRATING {role.upper()}  (id={arm_id}, port={port})")
    print("=" * 70)
    robot = SO101Follower(SO101FollowerConfig(port=port, id=arm_id))
    robot.connect(calibrate=True)
    obs = {k: round(v, 1) for k, v in robot.get_observation().items() if k.endswith(".pos")}
    print(f"{role} calibrated. Current position: {obs}")
    robot.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(ARMS), help="calibrate just one arm")
    args = ap.parse_args()

    roles = [args.only] if args.only else ["head", "left", "right"]
    print("Keep the workspace clear -- torque is released during calibration and "
          "the arms will go limp.\n")
    for role in roles:
        calibrate(role)
    print("\nAll requested arms calibrated. Next: python hands/play_full_body.py --dry-run")


if __name__ == "__main__":
    main()
