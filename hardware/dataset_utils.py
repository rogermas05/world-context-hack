"""
Shared helpers for locating the World Context dataset and its per-clip
calibration, regardless of which /Volumes/WCxx name the drive re-mounts
under (it's changed at least once already this session: WC26 -> WC27).
"""

import glob
import json
from pathlib import Path


def find_dataset_root() -> Path:
    matches = glob.glob("/Volumes/WC*/WORLD_CONTEXT_EXPLORER_V3")
    if not matches:
        raise FileNotFoundError(
            "Could not find the World Context dataset under /Volumes/WC*/WORLD_CONTEXT_EXPLORER_V3 "
            "-- is the drive plugged in?"
        )
    return Path(matches[0])


def find_clip_for_task(task_id: str, root: Path, clip_id: str | None = None) -> dict:
    """Returns a clips.jsonl row for a task, preferring one with full gyro
    calibration. Pass clip_id to pick a specific clip instead of the first match."""
    candidates = []
    with open(root / "meta" / "clips.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["canonical_task_id"] != task_id:
                continue
            if clip_id and row["clip_id"] != clip_id:
                continue
            candidates.append(row)
    if not candidates:
        raise ValueError(f"No clips found for task_id={task_id!r} (clip_id={clip_id!r})")
    calibrated = [c for c in candidates if c.get("calibration_status") == "intrinsics_and_gyro"]
    return (calibrated or candidates)[0]


def load_gyro_calibration(camera_id: str, root: Path) -> dict:
    with open(root / "meta" / "calibration.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["camera_id"] == camera_id:
                return row["gyro"]
    raise ValueError(f"No calibration found for camera_id={camera_id!r}")
