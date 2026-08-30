"""
One-time interactive calibration for the SO-101 leader arm.

Run this YOURSELF in your own terminal -- it will ask you to physically move
the arm to the middle of its range, press Enter, then move each joint through
its full range of motion and press Enter again. That's a physical step, so it
can't be automated or run on your behalf.

    python hardware/calibrate_leader.py

Saves a calibration file lerobot will reuse automatically on future connects
(keyed by LEADER_ID below), so you only need to do this once per physical arm.
"""

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

PORT = "/dev/tty.usbmodem5A7A0561491"  # from `ls /dev/tty.*` -- update if it changes
LEADER_ID = "hackathon_leader"

config = SO101LeaderConfig(port=PORT, id=LEADER_ID)
leader = SO101Leader(config)
leader.connect(calibrate=True)
print("Calibrated and connected. Current position:", leader.get_action())
leader.disconnect()
print("Done")
