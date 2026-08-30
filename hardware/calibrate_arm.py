"""
One-time interactive calibration for the SO-101 follower arm.

Run this YOURSELF in your own terminal -- it will ask you to physically move
the arm to the middle of its range, press Enter, then move each joint through
its full range of motion and press Enter again. That's a physical step, so it
can't be automated or run on your behalf.

    python hardware/calibrate_arm.py

Saves a calibration file lerobot will reuse automatically on future connects
(keyed by ARM_ID below), so you only need to do this once per physical arm.
"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5A7A0547481"  # from `ls /dev/tty.*` -- update if it changes
ARM_ID = "hackathon_follower"

config = SO101FollowerConfig(port=PORT, id=ARM_ID)
robot = SO101Follower(config)
robot.connect(calibrate=True)
print("Calibrated and connected. Current position:", robot.get_observation())
robot.disconnect()
print("Done -- you can now run play_head_trajectory.py")
