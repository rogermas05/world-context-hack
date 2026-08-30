# hands/ — ego hand motion → bimanual SO-101 mimicry

Turns a World Context egocentric clip into smooth bimanual SO-101 arm motion:
the two sim arms mimic the camera wearer's wrist trajectories and gripper
open/close, ignoring other people's hands in frame.

**Demo:** `demo_notebook_cover_assembly.mp4` (clip_7xg6sghgjjn46, t=120–180 s) —
left panel ego video with tracked hand skeletons, right panel two SO-101s in
MuJoCo. Wrist targets shown as spheres; segments where a hand is untracked are
tagged "no track (coasting)" and the arm holds its pose.

## Pipeline

1. `extract_hands.py CLIP_ID START_S END_S` — MediaPipe HandLandmarker on a
   crop of the hand region; PnP of its metric world landmarks against
   fisheye-undistorted image points (dataset camera calibration) → 6-DoF wrist
   pose per hand per frame → `hands_*.npz`.
   - **Wearer-only filtering (track-then-filter):** strict rules only when
     *acquiring* a track — the hand must point up-screen (`dyn < 0.25`; other
     people's hands reach in from the top pointing down-screen), wrist in the
     lower frame, PnP depth ≤ 0.6 m. Once tracked, association is by 3D
     continuity (< 0.35 m/frame) so a tracked hand can be followed anywhere,
     with a 3 s / 0.40 m resume window. Duplicates (two detections on one
     physical hand) collapse by mean landmark distance.
2. `retarget_sim.py hands_*.npz` — head-IMU complementary filter (calibrated
   gyro + gravity) removes head rotation; a robust plane fit maps the worker's
   (possibly tilted) work surface onto the robot bench; per-axis autoscaling
   fits the motion into the SO-101 workspace; mink differential IK with
   velocity limits + posture prior; zero-lag low-pass on the final joint
   trajectory. Smoothness is prioritized over exact mimicry. Outputs rendered
   frames + `trajectory.npz` (timestamps + 12 joint positions — directly
   playable on real SO-101s).
   - Fast iteration: `--stride 75 --half` renders a 12-frame preview in ~5 s.
3. `encode.py frames_dir out.mp4 15` — H.264 via PyAV (no ffmpeg binary needed).

## Setup

```bash
pip install "mediapipe==0.10.21" mujoco mink opencv-python-headless scipy av
# (mediapipe 1.x crashes on macOS — pin 0.10.21)
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
git clone https://github.com/TheRobotStudio/SO-ARM100.git   # SO-101 MJCF
export WORLDCONTEXT_ROOT=/path/to/WORLD_CONTEXT_EXPLORER_V3  # dataset package root
```

The scripts expect, under `WORLDCONTEXT_ROOT`: `meta/clips.jsonl`,
`meta/calibration.jsonl`, `videos/`, `imu/`, and the `worldcontext` Python
package (all part of the dataset package — not included here), plus
`work/models/hand_landmarker.task` and
`work/so_arm100/Simulation/SO101/so101_new_calib.xml` from the downloads above.
Use clips with `calibration_status == "intrinsics_and_gyro"` (200 in the release).

## Numbers (demo clip)

- wrist tracking error vs. retargeted target: median 3.4 cm (L) / 2.4 cm (R)
- hand detection: 85% (L) / 45% (R) — the R droughts are genuine
  unavailability (hand off-frame / wrapped around a tool); the arm holds pose
- other-person hand contamination after filtering: none observed

## Known limits

- MediaPipe monocular depth is noisy (winsorized + low-passed); head
  *translation* is ignored (rotation is IMU-compensated)
- 5-DoF arms can't match full wrist orientation (orientation is a soft target)
- Upgrade path: WiLoR/HaMeR for hand pose, HaWoR/SLAM for world-frame
  trajectories — the retarget/IK stage consumes the same npz format unchanged
