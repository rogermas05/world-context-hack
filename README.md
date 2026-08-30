# World Data Hack

Hackathon project on the World Context dataset: derives head-orientation
motion from calibrated egocentric IMU data and reenacts it on an SO-101 arm
(leader or follower), synced to the source clip for comparison.

## Layout

- `hardware/` — the pipeline
  - `dataset_utils.py` — locates the dataset drive and looks up per-clip camera calibration
  - `generate_trajectory.py` — turns one clip's calibrated gyro data into a neck turn/nod trajectory
  - `calibrate_arm.py` / `calibrate_leader.py` — one-time interactive SO-101 calibration
  - `play_head_trajectory.py` / `play_leader_trajectory.py` — plays a generated trajectory on the arm, synced with the source video
- `mockups/` — generated trajectory data, per-arm neutral poses, and exploratory visualizations

## Usage

```bash
python hardware/generate_trajectory.py <task-id>
python hardware/play_leader_trajectory.py <task-id>   # or play_head_trajectory.py for the follower
```

`<task-id>` is any canonical task id from the World Context dataset (e.g. `axle-shaft-cutting`, `metal-grinding`).
