"""Rebuild lerobot calibration files by reading them back off the servos.

SO-101 servos keep their homing offsets and range limits in EEPROM, so an arm
that was calibrated on one laptop is still calibrated when you plug it into
another -- only the host-side JSON is missing. This reads that back and writes
the JSON, which turns the interactive range-of-motion sweep into a no-op on any
machine after the first.

    python hands/recover_calibration.py            # all three arms
    python hands/recover_calibration.py --check    # report only, write nothing

If an arm reports a suspiciously narrow span (never calibrated, or its EEPROM
was reset), this refuses to write that arm and tells you to run
calibrate_all_arms.py for it instead.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

from arm_config import ARMS  # noqa: E402

# A real calibration sweeps a joint over a decent fraction of its travel. Much
# less than this means the EEPROM never saw a calibration.
MIN_PLAUSIBLE_SPAN_TICKS = 600


def read_arm(role: str, port: str, arm_id: str):
    robot = SO101Follower(SO101FollowerConfig(port=port, id=arm_id))
    try:
        robot.bus.connect(handshake=False)
        calibration = robot.bus.read_calibration()
        return robot, calibration
    except Exception:
        try:
            robot.bus.disconnect()
        except Exception:
            pass
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--only", choices=sorted(ARMS))
    args = ap.parse_args()

    roles = [args.only] if args.only else ["head", "left", "right"]
    wrote, skipped = [], []

    for role in roles:
        port, arm_id = ARMS[role]
        print(f"--- {role}  (id={arm_id}, port={port})")
        try:
            robot, calibration = read_arm(role, port, arm_id)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR: could not read ({exc})")
            skipped.append(role)
            continue

        try:
            narrow = []
            for name, cal in calibration.items():
                d = asdict(cal)
                span = d["range_max"] - d["range_min"]
                flag = ""
                if span < MIN_PLAUSIBLE_SPAN_TICKS:
                    narrow.append(name)
                    flag = "  <-- suspiciously narrow"
                print(f"    {name:15s} homing={d['homing_offset']:6d} "
                      f"range=[{d['range_min']:5d},{d['range_max']:5d}] span={span:5d}{flag}")

            if narrow:
                print(f"    SKIPPED: {', '.join(narrow)} look uncalibrated. "
                      f"Run: python hands/calibrate_all_arms.py --only {role}")
                skipped.append(role)
            elif args.check:
                print("    OK (--check: not written)")
            else:
                robot.calibration = calibration
                robot._save_calibration()
                print(f"    wrote {robot.calibration_fpath}")
                wrote.append(role)
        finally:
            try:
                robot.bus.disconnect()
            except Exception:
                pass

    print(f"\nrecovered: {wrote or 'none'}   needs manual calibration: {skipped or 'none'}")


if __name__ == "__main__":
    main()
