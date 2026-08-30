"""
Plays back a derived neck motion (turn + nod) on the SO-101 FOLLOWER arm for
any task, and opens the matching source clip at the same moment so accuracy
can be judged side by side.

Body metaphor (arm posed straight up = a person standing, facing forward):
    shoulder_pan  -> waist
    shoulder_lift -> lower spine
    elbow_flex    -> upper spine
    wrist_roll    -> neck turn   (driven by the dominant rotation axis)
    wrist_flex    -> neck nod    (driven by the second rotation axis)
    gripper       -> head

Waist and spine stay fixed at the standing pose; only the neck moves.

PREREQUISITE: run calibrate_arm.py once first (interactive, physical step).

    python hardware/generate_trajectory.py <task-id>   # if not already generated
    python hardware/play_head_trajectory.py <task-id>
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from dataset_utils import find_dataset_root

PORT = "/dev/tty.usbmodem5A7A0547481"  # from `ls /dev/tty.*` -- update if it changes
ARM_ID = "hackathon_follower"
MAX_RELATIVE_TARGET_DEG = 40.0  # hardware safety cap, independent of the offsets in the data itself
POSE_TRANSITION_HZ = 25.0
POSE_TOLERANCE_DEG = 1.0
MAX_POSE_TRANSITION_S = 8.0  # hard ceiling so a stuck joint can't loop forever

MOCKUPS_DIR = Path(__file__).parent.parent / "mockups"
FOLLOWER_POSE_FILE = MOCKUPS_DIR / "follower_neutral_pose.json"


def move_to_pose_smoothly(robot, target_pose: dict[str, float]) -> None:
    """Closed-loop move: repeatedly commands the REAL target (not a small step
    toward it), letting the servo's own position controller apply full corrective
    force each cycle -- bounded only by the hardware's own max_relative_target
    safety cap. A small artificial per-tick step (tried earlier) let gravity win
    against heavier joints (shoulder_lift/elbow_flex) between corrections; sending
    the real target every tick gives it full authority to actually hold the pose."""
    deadline = time.monotonic() + MAX_POSE_TRANSITION_S
    action = {f"{joint}.pos": target for joint, target in target_pose.items()}
    while time.monotonic() < deadline:
        robot.send_action(action)
        current = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        remaining = {f"{j}.pos": abs(target_pose[j] - current[f"{j}.pos"]) for j in target_pose}
        if all(r <= POSE_TOLERANCE_DEG for r in remaining.values()):
            return
        time.sleep(1.0 / POSE_TRANSITION_HZ)
    print("WARNING: pose transition hit its time limit before settling within tolerance.")


def play_video_synced(video_path: Path) -> None:
    """Opens video_path and blocks until QuickTime confirms playback has
    actually started (rather than guessing with a fixed sleep), so the
    caller's clock can start at a verified t=0."""
    subprocess.Popen(["open", "-a", "QuickTime Player", str(video_path)])
    time.sleep(1.5)  # just long enough for the app/document to exist
    result = subprocess.run(
        ["osascript", "-e", '''
            tell application "QuickTime Player"
                play document 1
                repeat 200 times
                    if (current time of document 1) > 0 then return "started"
                    delay 0.01
                end repeat
                return "timeout"
            end tell
        '''],
        capture_output=True, text=True,
    )
    if result.stdout.strip() != "started":
        print(f"WARNING: could not confirm video playback started "
              f"(got {result.stdout.strip()!r}, {result.stderr.strip()!r}) -- proceeding anyway")


def main(task_id: str) -> None:
    traj_file = MOCKUPS_DIR / f"head_trajectory_{task_id.replace('-', '_')}.json"
    data = json.loads(traj_file.read_text())
    traj = data["trajectory"]
    duration_s = data["duration_s"]
    neutral = json.loads(FOLLOWER_POSE_FILE.read_text())["neutral_pose"]
    video_file = find_dataset_root() / data["source_video"]
    print(f"Loaded {len(traj)} steps for {task_id} ({duration_s}s)")

    config = SO101FollowerConfig(port=PORT, id=ARM_ID, max_relative_target=MAX_RELATIVE_TARGET_DEG)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)  # requires calibrate_arm.py to have been run already

    print("Moving to standing pose...")
    move_to_pose_smoothly(robot, neutral)
    print("In standing pose. Neck (wrist_flex/wrist_roll) will now drive from here.")

    try:
        for i in range(3, 0, -1):
            print(f"starting in {i}...")
            time.sleep(1)

        play_video_synced(video_file)
        print(">>> GO <<<")

        start = time.monotonic()
        for step in traj:
            now = time.monotonic() - start
            sleep_for = step["t"] - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            robot.send_action({
                "wrist_roll.pos": neutral["wrist_roll"] + step["neck_turn_offset_deg"],
                "wrist_flex.pos": neutral["wrist_flex"] + step["neck_nod_offset_deg"],
            })

        print("Returning to standing pose...")
        move_to_pose_smoothly(robot, neutral)
        print("Back at standing pose.")
    finally:
        robot.disconnect()
        print("disconnected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    args = parser.parse_args()
    main(args.task_id)
