"""Play one egocentric clip on the full three-arm rig: head + two hands.

    follower  -> CORE / HEAD : neck turn + nod integrated from the clip's head IMU
    leader A  -> LEFT hand   : worker's left wrist, retargeted by the hand pipeline
    leader B  -> RIGHT hand  : worker's right wrist

The source clip is opened in QuickTime at the same moment so the rig and the
video can be judged side by side.

    python hands/play_full_body.py --dry-run          # no hardware, prints the plan
    python hands/play_full_body.py                    # the real thing
    python hands/play_full_body.py --no-head          # hands only
    python hands/play_full_body.py --scale 0.5        # gentler motion

PREREQUISITE: python hands/calibrate_all_arms.py (once per physical arm).

Motion is played RELATIVE to whatever pose each arm is in when the script
starts (its "neutral"), not as absolute joint angles. That keeps the rig
independent of how each arm happened to be calibrated, and means a bad neutral
shows up as a small offset rather than a slam into a hard stop. Every joint is
additionally clamped to MAX_OFFSET_DEG and to the servo-level
max_relative_target cap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hardware"))

from arm_config import (  # noqa: E402
    ARMS, CONTROL_HZ, DEFAULT_MOTION_SCALE, JOINTS, MAX_OFFSET_DEG,
    MAX_POSE_TRANSITION_S, MAX_RELATIVE_TARGET_DEG, POSE_TOLERANCE_DEG, POSE_TRANSITION_HZ,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ = REPO / "hands/trajectories/z45uhnxrkwr5s_60_120.npz"

# Head motion shaping (mirrors hardware/generate_trajectory.py)
TURN_LIMIT_DEG = 30.0
NOD_LIMIT_DEG = 25.0
CLIP_FACTOR = 1.6
DEHIFT_S = 4.0


# --------------------------------------------------------------------------- data
def find_dataset_root() -> Path:
    """The dataset lives on an external drive on some machines and in the home
    directory on others; accept either, plus an explicit override."""
    env = os.environ.get("WORLDCONTEXT_ROOT")
    if env and (Path(env) / "meta/clips.jsonl").exists():
        return Path(env)
    import glob
    for pattern in ("/Volumes/WC*/WORLD_CONTEXT_EXPLORER_V3",
                    str(Path.home() / "WORLD_CONTEXT_EXPLORER_V3")):
        for match in glob.glob(pattern):
            if (Path(match) / "meta/clips.jsonl").exists():
                return Path(match)
    raise FileNotFoundError(
        "Could not find the World Context dataset. Set WORLDCONTEXT_ROOT=/path/to/it."
    )


def load_hand_trajectory(path: Path, scale: float):
    """npz from retarget_sim.py -> per-arm per-joint offsets in degrees."""
    d = np.load(path, allow_pickle=True)
    times = d["times"] - d["times"][0]
    q_deg = np.rad2deg(d["qpos"])
    names = [str(n) for n in d["joint_names"]]
    clip_id = str(d["clip_id"]) if "clip_id" in d else None

    out = {}
    for side in ("left", "right"):
        cols = {j: names.index(f"{side}/{j}") for j in JOINTS}
        ref = {j: float(np.median(q_deg[:, cols[j]])) for j in JOINTS}
        offs = {}
        for j in JOINTS:
            if j == "gripper":
                continue
            raw = q_deg[:, cols[j]] - ref[j]
            capped = np.clip(raw * scale, -MAX_OFFSET_DEG[j], MAX_OFFSET_DEG[j])
            offs[j] = capped
        # gripper: sim radians (0 closed .. ~1.75 open) -> lerobot RANGE_0_100
        g = q_deg[:, cols["gripper"]]
        lo, hi = np.percentile(g, 5), np.percentile(g, 95)
        span = max(hi - lo, 1e-6)
        offs["gripper"] = np.clip((g - lo) / span, 0, 1) * 100.0
        out[side] = offs
    return times, out, clip_id


def integrate_and_dehift(rate: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    angle = np.cumsum(rate * dt)
    win = max(3, int(DEHIFT_S * len(t) / (t[-1] - t[0])))
    win += 1 - win % 2
    pad = win // 2
    rolling = np.convolve(np.pad(angle, (pad, pad), mode="edge"), np.ones(win) / win, mode="valid")
    return angle - rolling


def load_head_trajectory(root: Path, clip_id: str, t_start: float, times: np.ndarray):
    """Neck turn/nod, in degrees, sampled at `times` (clip-relative seconds)."""
    clip = next(json.loads(l) for l in open(root / "meta/clips.jsonl")
                if json.loads(l)["clip_id"] == clip_id)
    cal = next(json.loads(l) for l in open(root / "meta/calibration.jsonl")
               if json.loads(l)["camera_id"] == clip["camera_id"])["gyro"]
    imu = json.loads((root / clip["imu_path"]).read_text())

    t = np.array(imu["t"])
    gyro_dps = np.degrees(np.array(imu["gyro"], float))
    # einsum, not matmul: Apple's Accelerate BLAS raises spurious divide-by-zero
    # FP flags here (verified identical results to 1e-14).
    calibrated = np.einsum("ij,nj->ni", np.array(cal["M"]), gyro_dps - np.array(cal["bias_dps"]))

    lo, hi = t_start, t_start + times[-1]
    m = (t >= lo) & (t <= hi)
    tw, cw = t[m], calibrated[m]
    axis_turn, axis_nod = np.argsort(cw.var(axis=0))[::-1][:2]

    def shaped(sig, limit):
        a = integrate_and_dehift(sig, tw)
        ref = np.percentile(np.abs(a), 97)
        return np.clip(a / max(ref, 1e-9), -CLIP_FACTOR, CLIP_FACTOR) * limit

    turn = np.interp(times, tw - t_start, shaped(cw[:, axis_turn], TURN_LIMIT_DEG))
    nod = np.interp(times, tw - t_start, shaped(cw[:, axis_nod], NOD_LIMIT_DEG))
    # CLIP_FACTOR lets the shaped signal exceed its nominal limit; clamp the neck
    # to the same per-joint envelope every other joint obeys.
    turn = np.clip(turn, -MAX_OFFSET_DEG["wrist_roll"], MAX_OFFSET_DEG["wrist_roll"])
    nod = np.clip(nod, -MAX_OFFSET_DEG["wrist_flex"], MAX_OFFSET_DEG["wrist_flex"])
    return turn, nod, clip["relative_path"]


# --------------------------------------------------------------------------- hardware
def joint_degree_limits(robot) -> dict[str, tuple[float, float]]:
    """Calibrated travel per joint in the degree units send_action uses.

    Mirrors lerobot's DEGREES normalization exactly: deg = (ticks - mid) * 360/(res-1)
    with mid = (range_min + range_max)/2, so travel is symmetric about zero at
    +/- half the swept span. (homing_offset lives in the servo EEPROM and is
    already baked into the raw reading -- it must NOT appear here.)
    """
    limits = {}
    for name, cal in robot.bus.calibration.items():
        motor = robot.bus.motors[name]
        if motor.norm_mode.name.startswith("RANGE"):
            limits[name] = (0.0, 100.0)
            continue
        max_res = robot.bus.model_resolution_table[motor.model] - 1
        half = (cal.range_max - cal.range_min) / 2 * 360 / max_res
        limits[name] = (-half, half)
    return limits


def fit_motion(role: str, robot, neutral: dict[str, float], offsets, margin_deg: float = 5.0):
    """Fit the planned excursion inside each joint's calibrated travel.

    Two levers, in order: shift the working neutral toward the middle of travel
    (free -- the arm just starts from a better pose), then, only if the swing
    still does not fit, scale that joint's amplitude down. Scaling preserves the
    shape of the motion, so a constrained joint moves less rather than wrongly.
    """
    limits = joint_degree_limits(robot)
    working = dict(neutral)
    fitted, notes = {}, []
    for joint, span in offsets.items():
        if joint == "gripper" or joint not in limits:
            fitted[joint] = span
            continue
        lo, hi = limits[joint]
        lo, hi = lo + margin_deg, hi - margin_deg
        neg, pos = float(np.min(span)), float(np.max(span))
        room = hi - lo
        scale = 1.0
        if (pos - neg) > room:                      # swing larger than the whole travel
            scale = room / (pos - neg)
            neg, pos = neg * scale, pos * scale
        base = min(max(neutral[joint], lo - neg), hi - pos)
        working[joint] = base
        fitted[joint] = span * scale
        if scale < 1.0 or abs(base - neutral[joint]) > 0.5:
            notes.append(f"{role}/{joint}: neutral {neutral[joint]:+.0f}->{base:+.0f}"
                         + (f", amplitude x{scale:.2f}" if scale < 1.0 else ""))
    return working, fitted, notes


def report_plan(role: str, robot, working: dict[str, float], offsets) -> None:
    limits = joint_degree_limits(robot)
    for joint, span in offsets.items():
        if joint == "gripper" or joint not in limits:
            continue
        lo, hi = limits[joint]
        c_lo = working[joint] + float(np.min(span))
        c_hi = working[joint] + float(np.max(span))
        margin = min(c_lo - lo, hi - c_hi)
        print(f"    {'ok ' if margin >= 0 else 'OVER'} {joint:15s} "
              f"start={working[joint]:+7.1f} cmd=[{c_lo:+7.1f},{c_hi:+7.1f}] "
              f"travel=[{lo:+7.1f},{hi:+7.1f}] margin={margin:+.0f}")


def move_to_pose_smoothly(robot, target: dict[str, float], label: str) -> None:
    """Command the real target every tick (not a small step toward it) so the
    servo's own controller keeps full authority against gravity."""
    deadline = time.monotonic() + MAX_POSE_TRANSITION_S
    action = {f"{j}.pos": v for j, v in target.items()}
    while time.monotonic() < deadline:
        robot.send_action(action)
        cur = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        if all(abs(target[j] - cur[f"{j}.pos"]) <= POSE_TOLERANCE_DEG for j in target):
            return
        time.sleep(1.0 / POSE_TRANSITION_HZ)
    print(f"WARNING: {label} did not settle within tolerance before the time limit.")


def play_video_synced(video_path: Path) -> None:
    subprocess.Popen(["open", "-a", "QuickTime Player", str(video_path)])
    time.sleep(1.5)
    result = subprocess.run(["osascript", "-e", '''
        tell application "QuickTime Player"
            play document 1
            repeat 200 times
                if (current time of document 1) > 0 then return "started"
                delay 0.01
            end repeat
            return "timeout"
        end tell'''], capture_output=True, text=True)
    if result.stdout.strip() != "started":
        print(f"WARNING: could not confirm playback started ({result.stdout.strip()!r}) -- continuing")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--clip-id", default=None, help="override the clip id stored in the npz")
    ap.add_argument("--clip-start", type=float, default=60.0,
                    help="clip-relative start time the trajectory was extracted from")
    ap.add_argument("--scale", type=float, default=DEFAULT_MOTION_SCALE)
    ap.add_argument("--duration", type=float, default=None, help="seconds to play (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="no hardware, no video: just the plan")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--swap-hands", action="store_true", help="if left/right come out mirrored")
    ap.add_argument("--preflight", action="store_true",
                    help="connect and check headroom, then exit without moving")
    args = ap.parse_args()

    times, hands, npz_clip = load_hand_trajectory(args.traj, args.scale)
    clip_id = args.clip_id or npz_clip
    if args.duration:
        keep = times <= args.duration
        times = times[keep]
        hands = {s: {j: v[keep] for j, v in o.items()} for s, o in hands.items()}

    if args.swap_hands:
        hands = {"left": hands["right"], "right": hands["left"]}

    # resample to the control rate
    t_ctrl = np.arange(0, times[-1], 1.0 / CONTROL_HZ)
    hands = {s: {j: np.interp(t_ctrl, times, v) for j, v in o.items()} for s, o in hands.items()}

    turn = nod = None
    video_path = None
    if clip_id:
        try:
            root = find_dataset_root()
            turn, nod, rel = load_head_trajectory(root, clip_id, args.clip_start, t_ctrl)
            video_path = root / rel
        except (FileNotFoundError, StopIteration) as exc:
            print(f"NOTE: no head motion / video ({exc}); hands only.")
    if args.no_head:
        turn = nod = None

    print(f"clip={clip_id}  {len(t_ctrl)} steps @ {CONTROL_HZ:g} Hz  "
          f"({t_ctrl[-1]:.1f}s)  motion scale={args.scale}")
    for side in ("left", "right"):
        rng = {j: f"{v.min():+.0f}..{v.max():+.0f}" for j, v in hands[side].items() if j != "gripper"}
        print(f"  {side:5s} offsets(deg): {rng}")
    if turn is not None:
        print(f"  head   turn {turn.min():+.0f}..{turn.max():+.0f}  nod {nod.min():+.0f}..{nod.max():+.0f} deg")

    if args.dry_run:
        print("\n--dry-run: nothing was sent to hardware.")
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    roles = ["left", "right"] + ([] if turn is None else ["head"])
    robots, neutral = {}, {}
    try:
        for role in roles:
            port, arm_id = ARMS[role]
            cfg = SO101FollowerConfig(port=port, id=arm_id,
                                      max_relative_target=MAX_RELATIVE_TARGET_DEG)
            robot = SO101Follower(cfg)
            robot.connect(calibrate=False)   # requires calibrate_all_arms.py to have run
            robots[role] = robot
            obs = robot.get_observation()
            neutral[role] = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
            print(f"connected {role:5s} neutral={ {j: round(v,1) for j,v in neutral[role].items()} }")

        print("\nfitting motion to each arm's calibrated travel:")
        all_notes = []
        for role in roles:
            plan = hands[role] if role in hands else {"wrist_roll": turn, "wrist_flex": nod}
            working, fitted, notes = fit_motion(role, robots[role], neutral[role], plan)
            neutral[role] = working
            if role in hands:
                hands[role] = fitted
            else:
                turn, nod = fitted["wrist_roll"], fitted["wrist_flex"]
            all_notes += notes
            print(f"  {role}:")
            report_plan(role, robots[role], working, fitted)
        for n in all_notes:
            print(f"  adjusted: {n}")

        if args.preflight:
            print("\n--preflight: nothing was moved.")
            return

        print("\nmoving to start pose...")
        for role in roles:
            move_to_pose_smoothly(robots[role], neutral[role], role)

        print("\n>>> KEEP HANDS CLEAR OF THE ARMS <<<")
        for i in range(3, 0, -1):
            print(f"starting in {i}...")
            time.sleep(1)

        if video_path and not args.no_video:
            play_video_synced(video_path)
        print(">>> GO <<<")

        start = time.monotonic()
        for k, tk in enumerate(t_ctrl):
            sleep_for = tk - (time.monotonic() - start)
            if sleep_for > 0:
                time.sleep(sleep_for)
            for side in ("left", "right"):
                base, off = neutral[side], hands[side]
                robots[side].send_action({
                    **{f"{j}.pos": base[j] + float(off[j][k]) for j in JOINTS if j != "gripper"},
                    "gripper.pos": float(off["gripper"][k]),
                })
            if turn is not None:
                h = neutral["head"]
                robots["head"].send_action({
                    "wrist_roll.pos": h["wrist_roll"] + float(turn[k]),
                    "wrist_flex.pos": h["wrist_flex"] + float(nod[k]),
                })
        print("playback complete; returning to neutral...")
        for role in roles:
            move_to_pose_smoothly(robots[role], neutral[role], role)
    except KeyboardInterrupt:
        print("\ninterrupted -- returning to neutral")
        for role, robot in robots.items():
            try:
                move_to_pose_smoothly(robot, neutral[role], role)
            except Exception as exc:  # noqa: BLE001
                print(f"  ({role}: {exc})")
    finally:
        for role, robot in robots.items():
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"disconnect {role}: {exc}")
        print("disconnected")


if __name__ == "__main__":
    main()
