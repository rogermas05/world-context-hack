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

# The real, hand-calibrated standing pose for each arm (from hardware/'s
# earlier calibration session), NOT whatever pose the arm happens to be in
# when this script connects -- that can be an arbitrary gravity-sagged pose
# if the arm has sat idle with torque off.
NEUTRAL_POSE_FILE = {
    "head": REPO / "mockups/follower_neutral_pose.json",
    "left": REPO / "mockups/leader_neutral_pose.json",
    "right": REPO / "mockups/l1_right_neutral_pose.json",
}


def load_saved_neutral(role: str) -> dict[str, float]:
    return json.loads(NEUTRAL_POSE_FILE[role].read_text())["neutral_pose"]

# Head motion shaping (mirrors hardware/generate_trajectory.py)
TURN_LIMIT_DEG = 35.0
NOD_LIMIT_DEG = 30.0
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
    turn = np.clip(turn, -MAX_OFFSET_DEG["shoulder_pan"], MAX_OFFSET_DEG["shoulder_pan"])
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


REST_FALL_S = 4.0  # seconds to ease down to rest at the end -- NEVER slam the arms down


def move_to_rest_slowly(robot, target: dict[str, float], label: str,
                         duration_s: float = REST_FALL_S) -> None:
    """Eases every joint from its current position down to `target` (neutral/
    rest) over duration_s seconds on a smooth cosine curve, sent in small
    steps computed here -- NOT a fast snap via send-full-target-and-let-the-
    servo-close-the-gap. This is deliberately independent of how responsive
    MAX_RELATIVE_TARGET_DEG is set for live playback: raising that cap for
    accurate tracking during motion must never make the end-of-run "return to
    rest" move fast too. Always use this (not move_to_pose_smoothly) for
    settling the arms at the end of a run, interrupted or not."""
    current = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    n_steps = max(1, int(round(duration_s * POSE_TRANSITION_HZ)))
    for i in range(1, n_steps + 1):
        frac = (1 - np.cos(np.pi * i / n_steps)) / 2  # 0 -> 1, ease-in-out
        action = {f"{j}.pos": current[f"{j}.pos"] + (v - current[f"{j}.pos"]) * frac
                  for j, v in target.items()}
        robot.send_action(action)
        time.sleep(1.0 / POSE_TRANSITION_HZ)
    print(f"  {label}: eased down to rest over {duration_s:.1f}s")


def play_video_synced(video_path: Path, start_s: float = 0.0, speed: float = 1.0) -> None:
    """Opens video_path, seeks to start_s (the trajectory data's own clip-relative
    start time -- the arms are performing THAT window, not the beginning of the
    clip), sets its playback rate to match the arm's --speed, then confirms
    playback is actually advancing before returning."""
    subprocess.Popen(["open", "-a", "QuickTime Player", str(video_path)])
    time.sleep(1.5)
    result = subprocess.run(["osascript", "-e", f'''
        tell application "QuickTime Player"
            set current time of document 1 to {start_s}
            set rate of document 1 to {speed}
            repeat 200 times
                if (current time of document 1) > {start_s} then return "started"
                delay 0.01
            end repeat
            return "timeout"
        end tell'''], capture_output=True, text=True)
    if result.stdout.strip() != "started":
        print(f"WARNING: could not confirm playback started ({result.stdout.strip()!r}) -- continuing")


def add_soft_fall(t_ctrl: np.ndarray, hands: dict, turn, nod, tail_s: float):
    """Extends the trajectory with a cosine ease-out tail: every offset-based
    joint (not gripper, which is an absolute 0-100 position, not an offset)
    glides smoothly from its final value down to 0 (neutral) over tail_s
    seconds, instead of the motion just stopping and the arm snapping back."""
    if tail_s <= 0:
        return t_ctrl, hands, turn, nod
    n_tail = max(1, int(round(tail_s * CONTROL_HZ)))
    ramp = (np.cos(np.linspace(0, np.pi, n_tail)) + 1) / 2  # 1 -> 0, smooth

    def fall(arr):
        return np.concatenate([arr, arr[-1] * ramp])

    t_tail = t_ctrl[-1] + (np.arange(1, n_tail + 1) / CONTROL_HZ)
    t_ctrl = np.concatenate([t_ctrl, t_tail])
    hands = {s: {j: (fall(v) if j != "gripper" else np.concatenate([v, np.full(n_tail, v[-1])]))
                 for j, v in o.items()} for s, o in hands.items()}
    if turn is not None:
        turn, nod = fall(turn), fall(nod)
    return t_ctrl, hands, turn, nod


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--clip-id", default=None, help="override the clip id stored in the npz")
    ap.add_argument("--clip-start", type=float, default=60.0,
                    help="clip-relative start time the trajectory was extracted from")
    ap.add_argument("--scale", type=float, default=DEFAULT_MOTION_SCALE)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate for BOTH arm motion and video (e.g. 0.5 = half speed). "
                         "The safety clamp regularly engages at 1.0x, meaning the intended motion "
                         "exceeds what the servos can actually track in real time -- slowing down "
                         "gives them enough real time to reach each waypoint instead of lagging.")
    ap.add_argument("--duration", type=float, default=None, help="seconds to play (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="no hardware, no video: just the plan")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--overlay-video", type=Path, default=None,
                    help="play this rendered CV-overlay video (ego + hand skeleton + MuJoCo sim, "
                         "from retarget_sim.py + encode.py) instead of the raw dataset clip. It's "
                         "already trimmed to the extraction window, so it plays from its own t=0, "
                         "not --clip-start.")
    ap.add_argument("--swap-hands", action="store_true", help="if left/right come out mirrored")
    ap.add_argument("--soft-fall", type=float, default=2.5,
                    help="seconds to ease every joint's offset smoothly down to 0 (neutral) at "
                         "the end, instead of the motion just stopping and snapping back. 0 disables.")
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

    t_ctrl, hands, turn, nod = add_soft_fall(t_ctrl, hands, turn, nod, args.soft_fall)

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
            neutral[role] = load_saved_neutral(role)
            print(f"connected {role:5s} neutral (calibrated standing pose)="
                  f"{ {j: round(v,1) for j,v in neutral[role].items()} }")

        print("\nfitting motion to each arm's calibrated travel:")
        all_notes = []
        for role in roles:
            # turn drives the WAIST (shoulder_pan), not the neck (wrist_roll) --
            # a neck-only turn barely swings sideways since it's such a short
            # lever arm; the whole body rotating from the base reads much more
            # clearly as "turning to look" (found live on the rig).
            plan = hands[role] if role in hands else {"shoulder_pan": turn, "wrist_flex": nod}
            working, fitted, notes = fit_motion(role, robots[role], neutral[role], plan)
            neutral[role] = working
            if role in hands:
                hands[role] = fitted
            else:
                turn, nod = fitted["shoulder_pan"], fitted["wrist_flex"]
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

        if args.overlay_video:
            play_video_synced(args.overlay_video, 0.0, args.speed)
        elif video_path and not args.no_video:
            play_video_synced(video_path, args.clip_start, args.speed)
        print(">>> GO <<<" + (f"  (speed={args.speed}x)" if args.speed != 1.0 else ""))

        start = time.monotonic()
        for k, tk in enumerate(t_ctrl):
            # stretched wall-clock schedule: at speed<1, each waypoint gets more
            # real time to actually arrive, instead of the servo perpetually
            # lagging behind a target it physically can't track that fast.
            sleep_for = (tk / args.speed) - (time.monotonic() - start)
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
                    # Static joints re-sent every tick too, not just the two
                    # driven neck joints -- otherwise nothing fights gravity
                    # creep on shoulder_lift/elbow_flex for the full 60s and
                    # the head sags out of position (found live on the rig).
                    "shoulder_lift.pos": h["shoulder_lift"],
                    "elbow_flex.pos": h["elbow_flex"],
                    "gripper.pos": h["gripper"],
                    "wrist_roll.pos": h["wrist_roll"],
                    "shoulder_pan.pos": h["shoulder_pan"] + float(turn[k]),
                    "wrist_flex.pos": h["wrist_flex"] + float(nod[k]),
                })
        print("playback complete; easing down to rest...")
        for role in roles:
            move_to_rest_slowly(robots[role], neutral[role], role)
    except KeyboardInterrupt:
        print("\ninterrupted -- easing down to rest")
        for role, robot in robots.items():
            try:
                move_to_rest_slowly(robot, neutral[role], role)
            except KeyboardInterrupt:
                # a second interrupt while already easing down -- KeyboardInterrupt
                # is a BaseException, not an Exception, so it needs its own clause
                # or it crashes out here uncaught instead of finishing gracefully.
                print(f"  ({role}: second interrupt -- skipping ahead to disconnect)")
                break
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
