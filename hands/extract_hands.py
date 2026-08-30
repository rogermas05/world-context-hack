"""Stage 1: egocentric clip -> per-frame 3D hand poses in the camera frame.

MediaPipe HandLandmarker (2D px + metric-ish world landmarks) on a crop of the hand
region, then PnP against the fisheye-undistorted normalized image points using the
dataset's calibration -> 6-DoF wrist pose in the camera frame.

Usage: python extract_hands.py CLIP_ID START_S END_S [--fps 15] [--out hands.npz]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

ROOT = Path(os.environ.get("WORLDCONTEXT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from worldcontext.calibration import load_calibration, K_and_D  # noqa: E402

MODEL = ROOT / "work/models/hand_landmarker.task"
CROP = (250, 1080, 160, 1760)  # y0,y1,x0,x1 — where hands live in ego footage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id"); ap.add_argument("start_s", type=float); ap.add_argument("end_s", type=float)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    clips = {json.loads(l)["clip_id"]: json.loads(l) for l in open(ROOT / "meta/clips.jsonl")}
    c = clips[a.clip_id]
    rec = load_calibration(ROOT)[c["camera_id"]]
    K, D = (np.array(x, float) for x in K_and_D(rec)); D = D.reshape(4, 1)

    cap = cv2.VideoCapture(str(ROOT / c["relative_path"]))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    step = max(1, int(round(src_fps / a.fps)))
    f0, f1 = int(a.start_s * src_fps), int(a.end_s * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    # IMU: per-sample camera-frame angular velocity for rotation-compensated tracking
    imu = json.loads(open(ROOT / c["imu_path"]).read())
    Mg = np.array(rec["gyro"]["M"]); bg = np.deg2rad(np.array(rec["gyro"]["bias_dps"]))
    imu_t = np.array(imu["t"]) + rec["gyro"].get("time_offset_ms", 0.0) / 1000.0
    imu_w = (Mg @ (np.array(imu["gyro"], float) - bg).T).T

    def cam_rot_between(t0, t1):
        """R such that p_cam(t1) = R @ p_cam(t0) for a world-fixed point (pure rotation)."""
        i0, i1 = np.searchsorted(imu_t, [t0, t1])
        rv = np.zeros(3)
        for i in range(max(i0, 1) - 1, min(i1, len(imu_t) - 1)):
            rv = rv + imu_w[i] * (imu_t[i + 1] - imu_t[i])
        th = np.linalg.norm(rv)
        if th < 1e-9: return np.eye(3)
        kx, ky, kz = rv / th
        Km = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        return np.eye(3) - np.sin(th) * Km + (1 - np.cos(th)) * (Km @ Km)   # exp(-[rv])

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=4,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.4, min_tracking_confidence=0.3))

    y0, y1, x0, x1 = CROP
    N = (f1 - f0) // step + 1
    times = np.zeros(N); found = np.zeros((N, 2), bool)
    px = np.full((N, 2, 21, 2), np.nan); wl = np.full((N, 2, 21, 3), np.nan)
    R_cam_hand = np.full((N, 2, 3, 3), np.nan); t_cam_hand = np.full((N, 2, 3), np.nan)
    pinch = np.full((N, 2), np.nan)

    t0 = time.time(); k = 0; fi = f0
    last_seen = [None, None]; last_t = [0.0, 0.0]; last_rot_t = [0.0, 0.0]
    cand_log = []
    while fi <= f1:
        ok, fr = cap.read()
        if not ok: break
        if (fi - f0) % step == 0:
            crop = fr[y0:y1, x0:x1]
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                                      int(1000 * fi / src_fps))
            times[k] = fi / src_fps
            cands = []
            for hl, wlm, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
                p = np.array([[q.x * (x1 - x0) + x0, q.y * (y1 - y0) + y0] for q in hl])
                w = np.array([[q.x, q.y, q.z] for q in wlm])
                und = cv2.fisheye.undistortPoints(p.reshape(-1, 1, 2), K, D).reshape(-1, 2)
                okp, rvec, tvec = cv2.solvePnP(w.astype(np.float64), und.astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
                if not okp: continue
                R, _ = cv2.Rodrigues(rvec); tv = tvec.ravel()
                if np.linalg.norm(tv) > 0.75: continue          # absolute sanity: nobody's hand
                side = 1 if hd[0].category_name == "Left" else 0  # selfie-swap: ego video unmirrored
                cands.append({"p": p, "w": w, "side": side, "score": float(hd[0].score), "R": R, "tv": tv})
            cands.sort(key=lambda c: -c["score"])
            keep = []
            for c in cands:  # de-dup: same physical hand
                if all(np.mean(np.linalg.norm(c["p"] - o["p"], axis=1)) > 70 for o in keep):
                    keep.append(c)
            tnow = fi / src_fps
            for side in (0, 1):   # rotate last-seen positions into the current camera frame
                if last_seen[side] is not None:
                    last_seen[side] = cam_rot_between(max(last_t[side], last_rot_t[side]), tnow) @ last_seen[side]
                    last_rot_t[side] = tnow
            for c in keep:
                v = c["p"][9] - c["p"][0]
                cand_log.append([tnow, c["p"][0,0], c["p"][0,1], float(np.linalg.norm(c["tv"])),
                                 float(v[1]/(np.linalg.norm(v)+1e-9)), c["score"], c["side"]])
            # --- track-then-filter: per side, prefer continuity; strict wearer rules only to ACQUIRE
            def wearer_ok(c):
                # acquisition-only rules: conservative — a wearer hand APPEARS low, near, not
                # pointing down-screen. (Once tracked, association below may follow it anywhere.)
                depth = float(np.linalg.norm(c["tv"]))
                v = c["p"][9] - c["p"][0]; dyn = float(v[1] / (np.linalg.norm(v) + 1e-9))
                wy = float(c["p"][0, 1])
                # measured on this dataset: wearer hands point up-screen (dyn<0), other
                # people's hands reach in from the top pointing down-screen (dyn>0.35)
                return dyn < 0.25 and wy >= 400 and depth <= 0.60
            taken = [False] * len(keep)
            for side in (0, 1):
                alive = last_seen[side] is not None and (tnow - last_t[side]) < 0.8
                best, bestd = -1, 1e9
                for i, c in enumerate(keep):
                    if taken[i]: continue
                    if alive:
                        dist = np.linalg.norm(c["tv"] - last_seen[side])
                        v = c["p"][9] - c["p"][0]
                        if v[1] / (np.linalg.norm(v) + 1e-9) > 0.5 or np.linalg.norm(c["tv"]) > 0.65: continue
                        if dist < 0.35 and dist < bestd: best, bestd = i, dist
                    else:
                        if last_seen[side] is not None and (tnow - last_t[side]) < 3.0 \
                                and np.linalg.norm(c["tv"] - last_seen[side]) < 0.40 \
                                and float(c["p"][0, 1]) >= 380:
                            pass                                  # resume: near where we lost it
                        elif not (wearer_ok(c) and c["side"] == side):
                            continue
                        if c["score"] > -bestd: best, bestd = i, -c["score"]
                if best < 0 and not alive:  # fall back: any wearer-ok candidate on the correct half
                    for i, c in enumerate(keep):
                        if taken[i] or not wearer_ok(c): continue
                        onleft = c["p"][0, 0] < 960
                        if (side == 0) == onleft: best = i; break
                if best >= 0:
                    c = keep[best]; taken[best] = True
                    found[k, side] = True; px[k, side] = c["p"]; wl[k, side] = c["w"]
                    R_cam_hand[k, side] = c["R"]; t_cam_hand[k, side] = c["tv"]
                    pinch[k, side] = np.linalg.norm(c["w"][4] - c["w"][8])
                    last_seen[side] = c["tv"]; last_t[side] = tnow
            k += 1
            if k % 100 == 0:
                print(f"  {k}/{N} frames, {time.time()-t0:.1f}s, detect rate L={found[:k,0].mean():.2f} R={found[:k,1].mean():.2f}", flush=True)
        fi += 1
    N = k
    out = a.out or str(ROOT / f"work/ego2so101/hands_{a.clip_id}_{int(a.start_s)}_{int(a.end_s)}.npz")
    np.savez(out, clip_id=a.clip_id, times=times[:N], found=found[:N], px=px[:N], wl=wl[:N],
             R_cam_hand=R_cam_hand[:N], t_cam_hand=t_cam_hand[:N], pinch=pinch[:N], K=K, D=D, crop=np.array(CROP), cand_log=np.array(cand_log))
    print(f"wrote {out}: {N} frames, detect rate L={found[:N,0].mean():.2f} R={found[:N,1].mean():.2f}, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
