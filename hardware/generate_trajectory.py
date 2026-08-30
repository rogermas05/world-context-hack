"""
Generates a derived neck-motion trajectory (turn + nod, from integrated,
calibrated gyro orientation) for any task in the World Context dataset.

Turn and nod are real integrated ANGLES (not raw angular velocity), high-pass
filtered against a ~4s rolling mean -- a held head-turn stays elevated
instead of decaying back to baseline the moment rotation stops.

    python hardware/generate_trajectory.py <task-id> [--duration 75] [--clip-id CLIP_ID]

Writes mockups/head_trajectory_<task-id-with-underscores>.json. This file is
arm-agnostic (just the derived signal) -- see follower_neutral_pose.json /
leader_neutral_pose.json for the per-arm standing-pose reference each play
script combines it with.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_utils import find_clip_for_task, find_dataset_root, load_gyro_calibration

MOCKUPS_DIR = Path(__file__).parent.parent / "mockups"
TURN_LIMIT_DEG = 35.0
NOD_LIMIT_DEG = 30.0
CLIP_FACTOR = 1.6
CONTROL_HZ = 25.0
DEHIFT_S = 4.0  # high-pass window -- longer than any real head-turn hold


def integrate_and_dehift(rate_series: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    angle = np.cumsum(rate_series * dt)
    win = max(3, int(DEHIFT_S * len(t) / (t[-1] - t[0])))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(angle, (pad, pad), mode="edge")
    k = np.ones(win) / win
    rolling_mean = np.convolve(padded, k, mode="valid")
    return angle - rolling_mean


def scale(sig: np.ndarray, limit_deg: float) -> np.ndarray:
    ref = np.percentile(np.abs(sig), 97)
    return np.clip(sig / ref, -CLIP_FACTOR, CLIP_FACTOR) * limit_deg


def generate(task_id: str, duration_s: float, clip_id: str | None) -> Path:
    root = find_dataset_root()
    clip_row = find_clip_for_task(task_id, root, clip_id)
    cal = load_gyro_calibration(clip_row["camera_id"], root)

    imu = json.loads((root / clip_row["imu_path"]).read_text())
    t = np.array(imu["t"])
    gyro_dps = np.degrees(np.array(imu["gyro"]))
    M = np.array(cal["M"])
    bias = np.array(cal["bias_dps"])
    calibrated_dps = (M @ (gyro_dps - bias).T).T

    mask = t <= duration_s
    t_win = t[mask]
    cal_win = calibrated_dps[mask]

    variances = cal_win.var(axis=0)
    axis_turn, axis_nod = np.argsort(variances)[::-1][:2]

    turn_angle = integrate_and_dehift(cal_win[:, axis_turn], t_win)
    nod_angle = integrate_and_dehift(cal_win[:, axis_nod], t_win)
    turn_offset = scale(turn_angle, TURN_LIMIT_DEG)
    nod_offset = scale(nod_angle, NOD_LIMIT_DEG)

    t_ctrl = np.arange(0, duration_s, 1.0 / CONTROL_HZ)
    turn_ctrl = np.interp(t_ctrl, t_win, turn_offset)
    nod_ctrl = np.interp(t_ctrl, t_win, nod_offset)

    ctrl_win = max(3, int(0.12 * CONTROL_HZ))
    ck = np.ones(ctrl_win) / ctrl_win
    turn_ctrl = np.convolve(turn_ctrl, ck, mode="same")
    nod_ctrl = np.convolve(nod_ctrl, ck, mode="same")

    traj = [{"t": round(float(tt), 3),
             "neck_turn_offset_deg": round(float(tu), 2),
             "neck_nod_offset_deg": round(float(no), 2)}
            for tt, tu, no in zip(t_ctrl, turn_ctrl, nod_ctrl)]

    out_path = MOCKUPS_DIR / f"head_trajectory_{task_id.replace('-', '_')}.json"
    out_path.write_text(json.dumps({
        "clip_id": clip_row["clip_id"],
        "task_id": task_id,
        "source_video": clip_row["relative_path"],
        "duration_s": duration_s,
        "control_hz": CONTROL_HZ,
        "note": ("Integrated, dehifted orientation (real angle, not raw angular velocity) -- "
                 "held turns/nods stay elevated instead of decaying to baseline."),
        "trajectory": traj,
    }, indent=1))
    print(f"wrote {out_path}")
    print(f"  {len(traj)} steps, turn=[{turn_ctrl.min():.1f},{turn_ctrl.max():.1f}]deg, "
          f"nod=[{nod_ctrl.min():.1f},{nod_ctrl.max():.1f}]deg")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--duration", type=float, default=75.0)
    parser.add_argument("--clip-id", default=None)
    args = parser.parse_args()
    generate(args.task_id, args.duration, args.clip_id)
