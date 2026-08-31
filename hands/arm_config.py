"""Shared hardware configuration for the three-arm rig.

Physical layout (all three are SO-101 arms on USB; the two "leader" arms are
driven here as position-controlled followers, which works because they carry
the same STS3215 servos):

    follower  -> CORE / HEAD   (neck turn + nod, from the clip's head IMU)
    leader A  -> LEFT hand     (worker's left wrist, from the hand pipeline)
    leader B  -> RIGHT hand    (worker's right wrist)

Ports come from `ls /dev/tty.usbmodem*`. If the arms re-enumerate, update
PORTS here (or pass --*-port on the command line) rather than editing scripts.
"""

from __future__ import annotations

# --- ports -----------------------------------------------------------------
HEAD_PORT = "/dev/tty.usbmodem5A7A0547481"   # follower -> core/head
LEFT_PORT = "/dev/tty.usbmodem5A7A0561491"   # leader A -> left hand
RIGHT_PORT = "/dev/tty.usbmodem5A7A0556051"  # leader B -> right hand

# --- lerobot arm ids (calibration is stored per id) ------------------------
HEAD_ID = "hackathon_follower"
LEFT_ID = "hackathon_hand_left"
RIGHT_ID = "hackathon_hand_right"

ARMS = {
    "head": (HEAD_PORT, HEAD_ID),
    "left": (LEFT_PORT, LEFT_ID),
    "right": (RIGHT_PORT, RIGHT_ID),
}

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# --- safety ----------------------------------------------------------------
# Hardware cap on how far a single command may jump from the present position.
# This is a SOFTWARE ceiling we impose, not a real hardware limit -- 25deg at
# 30Hz caps requested speed at ~750deg/s, which was regularly engaging (visible
# as constant "clamped to be safe" warnings), meaning our own cap was the
# bottleneck rather than the servo's real physical speed. Raised so the servo's
# actual capability becomes the limiting factor instead.
MAX_RELATIVE_TARGET_DEG = 45.0

# Per-joint cap on the excursion from the captured neutral pose. The sim
# trajectory can swing wrist_roll through ~320 deg, which is fine in MuJoCo but
# ugly and stressful on a real arm, so every joint is clamped to a demo-safe
# envelope. Widen deliberately, not by accident.
MAX_OFFSET_DEG = {
    "shoulder_pan": 50.0,
    "shoulder_lift": 50.0,
    "elbow_flex": 55.0,
    "wrist_flex": 60.0,
    "wrist_roll": 80.0,
    "gripper": 100.0,   # gripper is RANGE_0_100, handled separately
}

# Motion sent to the arms is this fraction of the (clamped) sim excursion.
# Some joints were already saturating the old, tighter MAX_OFFSET_DEG caps
# (e.g. wrist_flex hitting exactly +45deg), so raising scale alone wouldn't
# have made the motion any more expressive -- both were raised together.
DEFAULT_MOTION_SCALE = 1.1

CONTROL_HZ = 30.0
POSE_TRANSITION_HZ = 25.0
POSE_TOLERANCE_DEG = 2.0
MAX_POSE_TRANSITION_S = 8.0
