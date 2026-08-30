"""Stage 2: hand poses (camera frame) -> gravity/gyro-stabilised world frame -> two SO-101
arms in MuJoCo via mink IK -> side-by-side video frames.

Usage: python retarget_sim.py hands.npz [--t0 S --t1 S] [--scale 0.8] [--outdir frames_dir]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "cgl")
import cv2, numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
import mujoco, mink

ROOT = Path(os.environ.get("WORLDCONTEXT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from worldcontext.calibration import load_calibration  # noqa: E402

SO101_XML = ROOT / "work/so_arm100/Simulation/SO101/so101_new_calib.xml"
BASE_Y = 0.15          # robot bases at y = +-BASE_Y, facing +x
ROBOT_CENTER = np.array([0.22, 0.0, 0.09])
OUTWARD = 0.03         # small per-arm outward offset to reduce arm crossing
SHOULDER_OFF = np.array([0.0388, 0.0, 0.0624])
REACH = (0.155, 0.33)   # usable radius from shoulder
HANDS = ("left", "right")
HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


# ----------------------------------------------------------------------------- IMU
def head_orientation(clip, rec, frame_times):
    """Complementary filter: gyro integration + gravity correction -> R_WC(t) at frame times.
    World frame W: z up (gravity), y = camera forward projected on the horizontal plane, x = right."""
    imu = json.loads(open(ROOT / clip["imu_path"]).read())
    t = np.array(imu["t"]); g = np.array(imu["gyro"], float); a = np.array(imu["accl"], float)
    M = np.array(rec["gyro"]["M"]); bias = np.deg2rad(np.array(rec["gyro"]["bias_dps"]))
    t = t + rec["gyro"].get("time_offset_ms", 0.0) / 1000.0
    w = (M @ (g - bias).T).T            # rad/s, camera frame
    acc = (M @ a.T).T                   # m/s^2, camera frame; mean(acc) points UP
    i0 = max(0, np.searchsorted(t, frame_times[0] - 1.0))
    up_c = acc[i0:i0 + 200].mean(0); up_c /= np.linalg.norm(up_c)
    fwd = np.array([0, 0, 1.0]); fwd = fwd - fwd.dot(up_c) * up_c; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_c)
    R_WC = np.stack([right, fwd, up_c])  # rows = world axes expressed in camera frame
    out = np.zeros((len(frame_times), 3, 3)); k = 0
    for i in range(i0, len(t) - 1):
        dt = t[i + 1] - t[i]
        R_WC = R_WC @ Rot.from_rotvec(w[i] * dt).as_matrix()
        # gravity correction (small gain): rotate so measured up aligns with world z
        am = acc[i]; n = np.linalg.norm(am)
        if 7 < n < 13:
            up_w = R_WC @ (am / n)
            corr = np.cross(up_w, [0, 0, 1.0]) * 0.01
            R_WC = Rot.from_rotvec(corr).as_matrix() @ R_WC
        while k < len(frame_times) and frame_times[k] <= t[i + 1]:
            out[k] = R_WC; k += 1
        if k >= len(frame_times): break
    while k < len(frame_times): out[k] = R_WC; k += 1
    return out


# ----------------------------------------------------------------------------- hands
def lowpass(x, fps, cutoff=2.5):
    b, a = butter(2, cutoff / (fps / 2)); return filtfilt(b, a, x, axis=0)


def fill_gaps(x, found, max_gap):
    x = x.copy(); N = len(x); idx = np.where(found)[0]
    if len(idx) == 0: return x, np.zeros(N, bool)
    valid = np.zeros(N, bool)
    for j in range(x.shape[1] if x.ndim > 1 else 1):
        col = x[:, j] if x.ndim > 1 else x
        col[:] = np.interp(np.arange(N), idx, col[idx])
    # mark long gaps as "held" (interpolated but unreliable)
    valid[idx] = True
    return x, valid


def hand_tracks(d, R_WC, side, fps):
    f = d["found"][:, side]; R = d["R_cam_hand"][:, side]; t = d["t_cam_hand"][:, side]; wl = d["wl"][:, side]
    N = len(f)
    pos = np.full((N, 3), np.nan); xdir = np.full((N, 3), np.nan); nout = np.full((N, 3), np.nan)
    for k in np.where(f)[0]:
        P = (R[k] @ wl[k].T).T + t[k]              # 21 landmarks in camera frame
        pos[k] = R_WC[k] @ P[0]
        v = P[9] - P[0]; xdir[k] = R_WC[k] @ (v / np.linalg.norm(v))
        vi, vp = P[5] - P[0], P[17] - P[0]
        n = np.cross(vi, vp) if side == 1 else np.cross(vp, vi)   # palm-out normal
        nout[k] = R_WC[k] @ (n / np.linalg.norm(n))
    pos, _ = fill_gaps(pos, f, 15); xdir, _ = fill_gaps(xdir, f, 15); nout, _ = fill_gaps(nout, f, 15)
    pinch, _ = fill_gaps(d["pinch"][:, side].copy(), f, 15)
    pos = lowpass(pos, fps, 1.2); xdir = lowpass(xdir, fps, 0.5); nout = lowpass(nout, fps, 0.5); pinch = lowpass(pinch, fps, 1.2)
    return pos, xdir, nout, pinch, f


# ----------------------------------------------------------------------------- sim
def build_scene():
    spec = mujoco.MjSpec(); spec.modelname = "so101_bimanual"
    spec.visual.global_.offwidth = 1280; spec.visual.global_.offheight = 720
    w = spec.worldbody
    w.add_light(pos=[0.3, 0, 1.5], dir=[0, 0, -1], castshadow=True)
    w.add_light(pos=[-0.5, 0.5, 1.0], dir=[0.5, -0.5, -1], castshadow=False)
    w.add_geom(name="bench", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.2, 0, -0.02], size=[0.45, 0.45, 0.02], rgba=[0.55, 0.42, 0.28, 1])
    for name, y in (("left", BASE_Y), ("right", -BASE_Y)):
        arm = mujoco.MjSpec.from_file(str(SO101_XML))
        w.add_frame(pos=[0, y, 0]).attach_body(arm.worldbody.first_body(), name + "/", "")
    for name, rgba in (("left", [0.2, 0.4, 1, 0.7]), ("right", [1, 0.3, 0.2, 0.7])):
        b = w.add_body(name=f"target_{name}", mocap=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    model = spec.compile()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hands"); ap.add_argument("--t0", type=float); ap.add_argument("--t1", type=float)
    ap.add_argument("--scale", type=float, default=0.55); ap.add_argument("--outdir", default=None)
    ap.add_argument("--orient_cost", type=float, default=0.06)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--half", action="store_true")
    a = ap.parse_args()

    d = np.load(a.hands)
    clip_id = str(d["clip_id"]); times = d["times"]; fps = 1.0 / np.median(np.diff(times))
    clips = {json.loads(l)["clip_id"]: json.loads(l) for l in open(ROOT / "meta/clips.jsonl")}
    clip = clips[clip_id]; rec = load_calibration(ROOT)[clip["camera_id"]]

    R_WC = head_orientation(clip, rec, times)
    tracks = [hand_tracks(d, R_WC, s, fps) for s in range(2)]

    # ---- retarget: fit the (possibly tilted) work plane and map it onto the robot bench.
    for tr in tracks:                                          # winsorize PnP outliers per axis
        got = tr[0][tr[4]]
        lo, hi = np.percentile(got, 2, axis=0), np.percentile(got, 98, axis=0)
        np.clip(tr[0], lo, hi, out=tr[0])
    allp = np.concatenate([tr[0] for tr in tracks]); mu = allp.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((allp - mu).T))
    n = evecs[:, 0]; n = n if n[2] >= 0 else -n                # smallest-variance axis = plane normal
    if n[2] < 0.5: n = np.array([0, 0, 1.0])                   # fallback: gravity if fit is wild
    u1 = np.array([1, 0, 0.0]) - n[0] * n; u1 /= np.linalg.norm(u1)   # worker-lateral projected in plane
    u2 = np.cross(n, u1)                                       # "away from worker" in plane
    A = np.stack([u2, -u1, n])                                 # rows: W -> robot (x fwd, y left, z up)
    q = (A @ (allp - mu).T).T
    q5, q95 = np.percentile(q, 5, axis=0), np.percentile(q, 95, axis=0)
    qc = (q5 + q95) / 2; span = q95 - q5
    allowed = np.array([0.16, 0.40, 0.15])
    S = float(min(0.75, *(allowed / (span + 1e-9))))
    print(f"plane normal {n.round(2)} (tilt {np.degrees(np.arccos(n[2])):.0f} deg); spans {span.round(2)} -> scale {S:.2f}")
    med = np.stack([np.median(tr[0][tr[4]], 0) for tr in tracks])
    targets = []
    for s, tr in enumerate(tracks):
        p = ROBOT_CENTER + S * ((A @ (tr[0] - mu).T).T - qc)
        p[:, 1] += OUTWARD if s == 0 else -OUTWARD
        sh = np.array([0, BASE_Y if s == 0 else -BASE_Y, 0]) + SHOULDER_OFF
        v = p - sh; r = np.linalg.norm(v, axis=1, keepdims=True); rc = np.clip(r, *REACH)
        p = sh + v / r * rc; p[:, 2] = np.maximum(p[:, 2], 0.015); p[:, 0] = np.maximum(p[:, 0], 0.12)
        xd = (A @ tr[1].T).T; zd = -(A @ tr[2].T).T
        Rt = np.zeros((len(p), 3, 3))
        for k in range(len(p)):
            x = xd[k] / np.linalg.norm(xd[k]); y = np.cross(zd[k], x); y /= np.linalg.norm(y); z = np.cross(x, y)
            Rt[k] = np.stack([x, y, z], 1)
        grip = np.clip((tr[3] - 0.035) / (0.075 - 0.035), 0, 1) * 1.2
        for k in range(1, len(grip)):                     # rate-limit the gripper (rad/frame)
            grip[k] = grip[k - 1] + np.clip(grip[k] - grip[k - 1], -0.08, 0.08)
        targets.append((p, Rt, grip))

    # ---- IK
    model = build_scene(); data = mujoco.MjData(model)
    for n in HANDS:  # keep IK out of the folded-backward basin
        model.jnt_range[model.joint(f"{n}/shoulder_lift").id] = [-1.74, 0.55]
        model.jnt_range[model.joint(f"{n}/shoulder_pan").id] = [-1.4, 1.4]
    cfg = mink.Configuration(model)
    tasks = []; ftasks = []
    for s, name in enumerate(HANDS):
        ft = mink.FrameTask(frame_name=f"{name}/gripperframe", frame_type="site", position_cost=1.0, orientation_cost=a.orient_cost, lm_damping=1.0)
        ftasks.append(ft); tasks.append(ft)
    posture = mink.PostureTask(model, cost=4e-2); tasks.append(posture)
    limits = [mink.ConfigurationLimit(model),
              mink.VelocityLimit(model, {model.joint(j).name: 1.2 for j in range(model.njnt)})]
    q0 = np.zeros(model.nq)
    for name in HANDS:
        for j, v in (("shoulder_lift", -0.6), ("elbow_flex", 1.1), ("wrist_flex", 0.6)):
            q0[model.joint(f"{name}/{j}").qposadr[0]] = v
    cfg.update(q0); posture.set_target(q0)
    gidx = [model.joint(f"{n}/gripper").qposadr[0] for n in HANDS]
    mocap_ids = [model.body(f"target_{n}").mocapid[0] for n in HANDS]

    N = len(times); Q = np.zeros((N, model.nq)); err = np.zeros((N, 2))
    t0 = time.time()
    for k in range(N):
        for s in range(2):
            p, Rt, grip = targets[s]
            ftasks[s].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(Rt[k]), p[k]))
        for _ in range(3):
            vel = mink.solve_ik(cfg, tasks, 1.0 / fps, "daqp", damping=1e-3, limits=limits)
            cfg.integrate_inplace(vel, 1.0 / fps)
        q = cfg.q.copy()
        for s in range(2): q[gidx[s]] = targets[s][2][k]
        Q[k] = q; cfg.update(q)
        for s in range(2):
            sid = model.site(f"{HANDS[s]}/gripperframe").id
            err[k, s] = np.linalg.norm(data.site_xpos[sid] - targets[s][0][k]) if k else 0
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        for s in range(2):
            sid = model.site(f"{HANDS[s]}/gripperframe").id
            err[k, s] = np.linalg.norm(data.site_xpos[sid] - targets[s][0][k])
    # offline polish: zero-lag low-pass on the joint trajectory itself, re-clamped to limits
    Q = lowpass(Q, fps, 1.8)
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        Q[:, adr] = np.clip(Q[:, adr], model.jnt_range[j][0], model.jnt_range[j][1])
    for k in range(N):
        data.qpos[:] = Q[k]; mujoco.mj_forward(model, data)
        for s in range(2):
            sid = model.site(f"{HANDS[s]}/gripperframe").id
            err[k, s] = np.linalg.norm(data.site_xpos[sid] - targets[s][0][k])
    print(f"IK done {N} frames in {time.time()-t0:.1f}s; mean pos err L={err[:,0].mean()*100:.1f}cm R={err[:,1].mean()*100:.1f}cm, p90 {np.percentile(err,90)*100:.1f}cm")

    # ---- render
    outdir = Path(a.outdir or ROOT / f"work/ego2so101/frames_{clip_id}"); outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.jpg"): f.unlink()
    np.savez(outdir / "trajectory.npz", times=times, qpos=Q, joint_names=[model.joint(i).name for i in range(model.njnt)],
             target_pos=np.stack([t[0] for t in targets]), target_rot=np.stack([t[1] for t in targets]), pos_err=err)
    H, W_ = (360, 640) if a.half else (720, 1280)
    renderer = mujoco.Renderer(model, H, W_)
    cam = mujoco.MjvCamera(); cam.lookat = [0.20, 0, 0.06]; cam.distance = 0.85; cam.azimuth = 0; cam.elevation = -35
    opt = mujoco.MjvOption()
    cap = cv2.VideoCapture(str(ROOT / clip["relative_path"])); src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    px = d["px"]; found = d["found"]
    t0 = time.time()
    for k in range(0, N, a.stride):
        data.qpos[:] = Q[k]; mujoco.mj_forward(model, data)
        for s in range(2):
            data.mocap_pos[mocap_ids[s]] = targets[s][0][k]
            gid = model.body(f"target_{HANDS[s]}").geomadr[0]
            model.geom_rgba[gid, 3] = 0.85 if found[k, s] else 0.15
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam, opt); sim = renderer.render()[:, :, ::-1].copy()
        for s, nm in ((0, "L"), (1, "R")):
            if not found[k, s]:
                cv2.putText(sim, f"{nm}: no track (coasting)", (12, 690 - 28 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 235), 2, cv2.LINE_AA)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(times[k] * src_fps))); ok, fr = cap.read()
        if not ok: fr = np.zeros((1080, 1920, 3), np.uint8)
        for s, col in ((0, (255, 120, 40)), (1, (40, 80, 255))):
            if found[k, s]:
                P = px[k, s].astype(int)
                for i, j in HAND_CONN: cv2.line(fr, tuple(P[i]), tuple(P[j]), col, 3)
                for pnt in P: cv2.circle(fr, tuple(pnt), 5, (255, 255, 255), -1)
        vid = cv2.resize(fr, (W_, H))
        for img, label in ((vid, f"ego worker  {clip_id}  t={times[k]:.1f}s"), (sim, "SO-101 x2 in MuJoCo  (spheres = retargeted wrist targets)")):
            cv2.rectangle(img, (0, 0), (W_, 36), (0, 0, 0), -1)
            cv2.putText(img, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(outdir / f"{k:05d}.jpg"), np.hstack([vid, sim]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if k % 150 == 0: print(f"  rendered {k}/{N} {time.time()-t0:.1f}s", flush=True)
    print("frames in", outdir)


if __name__ == "__main__":
    main()
