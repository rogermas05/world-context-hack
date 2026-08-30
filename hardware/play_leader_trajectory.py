"""
Plays back a derived neck motion (turn + nod) on the SO-101 LEADER arm --
"this is the head" -- for any task, and opens the matching source clip at
the same moment so accuracy can be judged side by side.

Body metaphor (arm posed straight up = a person standing, facing forward):
    shoulder_pan  -> waist   (drives turn -- swings the WHOLE body, not just
                              the neck, since a neck-only turn barely swings
                              sideways at all)
    shoulder_lift -> lower spine
    elbow_flex    -> upper spine
    wrist_flex    -> neck nod
    wrist_roll    -> neck turn (unused during playback, held at neutral)
    gripper       -> head

Leader-specific: no send_action()/max_relative_target -- uses send_feedback()
and has no hardware safety clamp, so this script rate-limits its own
commands. Leader is torque-off by default (meant to be hand-guided) --
enable_torque() is required to hold a commanded position, and this script
disables torque again at the end so it's freely movable by hand afterward.

PREREQUISITE: run calibrate_leader.py once first (interactive, physical step).

    python hardware/generate_trajectory.py <task-id>   # if not already generated
    python hardware/play_leader_trajectory.py <task-id>
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from dataset_utils import find_dataset_root

PORT = "/dev/tty.usbmodem5A7A0561491"  # from `ls /dev/tty.*` -- update if it changes
LEADER_ID = "hackathon_leader"
MAX_STEP_DEG = 40.0  # software rate limit -- leader has no hardware max_relative_target
POSE_TOLERANCE_DEG = 1.0
MAX_POSE_TRANSITION_S = 8.0
POSE_TRANSITION_HZ = 25.0

MOCKUPS_DIR = Path(__file__).parent.parent / "mockups"
LEADER_POSE_FILE = MOCKUPS_DIR / "leader_neutral_pose.json"


def move_to_pose_smoothly(leader, target_pose: dict[str, float]) -> None:
    """Closed-loop move: repeatedly commands the real target, letting the
    servo's own position controller apply full corrective force each cycle."""
    deadline = time.monotonic() + MAX_POSE_TRANSITION_S
    feedback = {f"{joint}.pos": target for joint, target in target_pose.items()}
    while time.monotonic() < deadline:
        leader.send_feedback(feedback)
        current = leader.get_action()
        remaining = {j: abs(target_pose[j] - current[f"{j}.pos"]) for j in target_pose}
        if all(r <= POSE_TOLERANCE_DEG for r in remaining.values()):
            return
        time.sleep(1.0 / POSE_TRANSITION_HZ)
    print("WARNING: pose transition hit its time limit before settling within tolerance.")


def clamp_step(last_commanded: float, target: float) -> float:
    delta = target - last_commanded
    return last_commanded + max(-MAX_STEP_DEG, min(MAX_STEP_DEG, delta))


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
    neutral = json.loads(LEADER_POSE_FILE.read_text())["neutral_pose"]
    video_file = find_dataset_root() / data["source_video"]
    print(f"Loaded {len(traj)} steps for {task_id} ({duration_s}s) -- driving the LEADER")

    config = SO101LeaderConfig(port=PORT, id=LEADER_ID)
    leader = SO101Leader(config)
    leader.connect(calibrate=False)  # requires calibrate_leader.py to have been run already
    leader.enable_torque()  # off by default -- won't hold a commanded position otherwise

    try:
        print("Moving to standing pose...")
        move_to_pose_smoothly(leader, neutral)
        print("In standing pose. Neck (wrist_flex/wrist_roll) will now drive from here.")

        for i in range(3, 0, -1):
            print(f"starting in {i}...")
            time.sleep(1)

        play_video_synced(video_file)
        print(">>> GO <<<")

        last_turn = neutral["shoulder_pan"]
        last_nod = neutral["wrist_flex"]
        start = time.monotonic()
        for step in traj:
            now = time.monotonic() - start
            sleep_for = step["t"] - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            target_turn = neutral["shoulder_pan"] + step["neck_turn_offset_deg"]
            target_nod = neutral["wrist_flex"] + step["neck_nod_offset_deg"]
            last_turn = clamp_step(last_turn, target_turn)
            last_nod = clamp_step(last_nod, target_nod)
            leader.send_feedback({"shoulder_pan.pos": last_turn, "wrist_flex.pos": last_nod})

        print("Returning to standing pose...")
        move_to_pose_smoothly(leader, neutral)
        print("Back at standing pose.")
    finally:
        leader.disable_torque()  # so it's freely hand-movable again, as it was before this script ran
        leader.disconnect()
        print("disconnected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    args = parser.parse_args()
    main(args.task_id)
