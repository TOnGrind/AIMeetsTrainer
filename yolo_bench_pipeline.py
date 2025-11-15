import argparse, math, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import deque

# Ultralytics YOLO (pose)
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO ist nicht installiert. Bitte zuerst: pip install ultralytics")

# ------------------ Geometrie ------------------
def angle_at_point(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0: return np.nan
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def angle_to_horizontal(p, q):
    v = np.array(q, float) - np.array(p, float)
    if np.linalg.norm(v) == 0: return np.nan
    horiz = np.array([1.0, 0.0])
    cosang = np.clip(np.dot(v/np.linalg.norm(v), horiz), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def deriv(x, t, smooth_win=9, poly=2):
    if len(x) < smooth_win:
        vx = np.gradient(x, t)
        return vx, x
    xs = savgol_filter(x, smooth_win if smooth_win%2==1 else smooth_win+1, poly, mode="interp")
    vx = np.gradient(xs, t)
    return vx, xs

# ------------------ Rolling Mean Helfer ------------------
class RollingMean:
    def __init__(self, k=5):
        self.k = int(max(1, k))
        self.buf = deque(maxlen=self.k)
    def push(self, v):
        self.buf.append(float(v))
        return self.mean()
    def mean(self):
        if not self.buf: return None
        return float(np.nanmean(self.buf))

# ------------------ Phasen-/Rep-Logik ------------------
class RepStateMachine:
    def __init__(self,
                 fps,
                 top_deg=None, bottom_deg=None,
                 vel_eps=3.0,
                 hysteresis=7.0,
                 dwell_bottom_max=1.5,
                 min_state_frames=2,
                 mean_k=5,
                 backslide_v=5.0):
        self.fps = fps
        self.vel_eps = float(vel_eps)
        self.hyst = float(hysteresis)
        self.dwell_bottom_max = float(dwell_bottom_max)
        self.min_state_frames = int(max(1, min_state_frames))
        self.backslide_v = float(backslide_v)
        self.top_deg = top_deg
        self.bottom_deg = bottom_deg
        self.phase = "TOP"
        self.rep_id = 0
        self.rep_status = "ok"
        self.in_bottom_since = None
        self.awaited_bottom = False
        self.awaited_top = False
        self.rm_angle = RollingMean(k=mean_k)
        self.rm_vel   = RollingMean(k=mean_k)
        self._phase_frames = 0
        self.last_rep_time = None

    def _is_top(self, elbow_deg_m):
        return elbow_deg_m >= (self.top_deg - self.hyst)
    def _is_bottom(self, elbow_deg_m):
        return elbow_deg_m <= (self.bottom_deg + self.hyst)
    def _still(self, vel_m):
        return abs(vel_m) < self.vel_eps
    def _enough(self):
        return self._phase_frames >= self.min_state_frames

    def update(self, t, elbow_deg, elbow_vel, top_lock_s=0.8):
        event = None
        if self.top_deg is None: self.top_deg = 170.0
        if self.bottom_deg is None: self.bottom_deg = 110.0

        ang_m = self.rm_angle.push(elbow_deg)
        vel_m = self.rm_vel.push(elbow_vel if not np.isnan(elbow_vel) else 0.0)

        if np.isnan(vel_m) or self._still(vel_m): direction = 0
        elif vel_m < 0: direction = -1
        else: direction = +1
        self._phase_frames += 1

        # TOP-Lock: nach Repabschluss kurz Bewegungen ignorieren
        if self.phase == "TOP" and self.last_rep_time is not None and (t - self.last_rep_time) < top_lock_s:
            return self.phase, 0, self.rep_id, self.rep_status, None

        if self.phase == "TOP":
            if direction < 0 and not self._is_top(ang_m) and self._enough():
                self.phase = "DOWN"; self._phase_frames = 0
                self.awaited_bottom = True; self.rep_status = "ok"

        elif self.phase == "DOWN":
            if self._is_bottom(ang_m) and self._still(vel_m) and self._enough():
                self.phase = "BOTTOM"; self._phase_frames = 0
                self.in_bottom_since = t; self.awaited_bottom = False
            elif direction > 0 and self.awaited_bottom and self._enough():
                self.rep_status = "fail:no_bottom"; event = "fail:no_bottom"
                self.phase = "UP"; self._phase_frames = 0
                self.awaited_bottom = False; self.awaited_top = True

        elif self.phase == "BOTTOM":
            if self.in_bottom_since is not None and (t - self.in_bottom_since) > self.dwell_bottom_max:
                if not str(self.rep_status).startswith("fail"):
                    self.rep_status = "fail:long_bottom"; event = "fail:long_bottom"
            if direction > 0 and self._enough():
                self.phase = "UP"; self._phase_frames = 0
                self.in_bottom_since = None; self.awaited_top = True

        elif self.phase == "UP":
            if vel_m < -self.backslide_v and self._enough():
                self.rep_status = "fail:backslide_up"; event = "fail:backslide_up"
                self.phase = "DOWN"; self._phase_frames = 0
                self.awaited_top = False; self.awaited_bottom = True
            elif self._is_top(ang_m) and self._still(vel_m) and self._enough():
                self.phase = "TOP"; self._phase_frames = 0
                self.awaited_top = False
                self.rep_id += 1; event = "rep++"; self.last_rep_time = t
                self.rep_status = "ok"
            elif direction < 0 and self.awaited_top and self._enough():
                self.rep_status = "fail:no_top"; event = "fail:no_top"
                self.phase = "DOWN"; self._phase_frames = 0
                self.awaited_top = False; self.awaited_bottom = True

        return self.phase, direction, self.rep_id, self.rep_status, event

# ------------------ YOLO Keypoint-Hilfen ------------------
# COCO-17: 5=L_SH, 6=R_SH, 7=L_EL, 8=R_EL, 9=L_WR, 10=R_WR, 11=L_HIP, 12=R_HIP
KP = {
    "L_SH":5, "R_SH":6, "L_EL":7, "R_EL":8, "L_WR":9, "R_WR":10, "L_HIP":11, "R_HIP":12
}

def pick_person_with_highest_conf(result):
    # result.keypoints.xy shape: (num_persons, 17, 2); result.keypoints.conf shape: (num_persons, 17)
    if result.keypoints is None or result.keypoints.xy is None: return None
    xy = result.keypoints.xy
    conf = result.keypoints.conf
    n = xy.shape[0]
    if n == 0: return None
    # Score: Mittelwert aus Schulter/Elbow/Wrist/Hip-Confidences beider Seiten
    important = [KP[k] for k in ["L_SH","R_SH","L_EL","R_EL","L_WR","R_WR","L_HIP","R_HIP"]]
    scores = conf[:, important].mean(dim=1).cpu().numpy() if hasattr(conf, 'cpu') else conf[:, important].mean(axis=1)
    idx = int(np.argmax(scores))
    return idx


def get_side_points_yolo(xy, conf, side="auto"):
    # xy shape: (17,2); conf shape: (17,)
    L = {"SH":KP["L_SH"], "EL":KP["L_EL"], "WR":KP["L_WR"], "HP":KP["L_HIP"]}
    R = {"SH":KP["R_SH"], "EL":KP["R_EL"], "WR":KP["R_WR"], "HP":KP["R_HIP"]}

    def p(i):
        return (float(xy[i,0]), float(xy[i,1]))
    def c(i):
        return float(conf[i]) if conf is not None else 0.0

    if side == "auto":
        left_score  = np.mean([c(L["SH"]), c(L["EL"]), c(L["WR"]), c(L["HP"])])
        right_score = np.mean([c(R["SH"]), c(R["EL"]), c(R["WR"]), c(R["HP"])])
        side = "left" if left_score >= right_score else "right"

    S = L if side=="left" else R
    SH, EL, WR, HP = p(S["SH"]), p(S["EL"]), p(S["WR"]), p(S["HP"])
    return side, SH, EL, WR, HP

# ------------------ Hauptpipeline (YOLO) ------------------

def process_video_yolo(
    video_path,
    outdir="runs/bench_yolo",
    side="auto",
    model_name="yolov8n-pose.pt",
    device=None,
    vis=True,
    # Glättung/Entscheidung
    smooth_win=9, poly=2,
    mean_k=7, stride=1,
    vel_eps=5.0, hysteresis=10.0,
    dwell_bottom_max=1.5, min_state_frames=6,
    backslide_v=5.0,
    top_percentile=85.0, bottom_percentile=15.0,
    clamp_top_min=150.0, clamp_bottom_max=110.0,
):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = str(outdir / "annotated.mp4")
    writer = cv2.VideoWriter(out_vid, fourcc, fps, (W, H))

    model = YOLO(model_name)
    if device is not None:
        model.to(device)

    rows = []
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None
    pbar = tqdm(total=frame_total, desc="Processing (YOLO-Pose)")
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        do_process = (idx % max(1, stride) == 0)

        elbow_deg = np.nan; upperarm_torso_deg = np.nan; wrist_horiz_deg = np.nan
        chosen_side = side

        if do_process:
            # Inferenz
            results = model.predict(frame, verbose=False, conf=0.25, imgsz=max(640, min(W, H)))
            res = results[0]
            pid = pick_person_with_highest_conf(res)
            if pid is not None:
                # Keypoints + optional Visualisierung
                kps_xy = res.keypoints.xy[pid].detach().cpu().numpy() if hasattr(res.keypoints.xy, 'detach') else res.keypoints.xy[pid].numpy()
                kps_cf = res.keypoints.conf[pid].detach().cpu().numpy() if hasattr(res.keypoints.conf, 'detach') else res.keypoints.conf[pid].numpy()

                chosen_side, SH, EL, WR, HP = get_side_points_yolo(kps_xy, kps_cf, side)

                elbow_deg = angle_at_point(SH, EL, WR)
                upperarm_torso_deg = angle_at_point(HP, SH, EL)  # Primärsignal
                wrist_horiz_deg = angle_to_horizontal(EL, WR)

                if vis:
                    frame = res.plot()  # Ultralytics Zeichnung (Skelet + BBox)

        # Overlay Basistext
        if vis:
            y0 = 28
            cv2.putText(frame, f"Side: {chosen_side}", (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        t = idx / fps
        rows.append([idx, t, elbow_deg, upperarm_torso_deg, wrist_horiz_deg])
        writer.write(frame)
        idx += 1
        pbar.update(1)

    pbar.close(); writer.release(); cap.release()

    # --- DataFrame & Signale ---
    df = pd.DataFrame(rows, columns=["frame","time_s","elbow_deg","upperarm_torso_deg","forearm_horiz_deg"]) 

    t = df["time_s"].to_numpy()
    primary = df["upperarm_torso_deg"].to_numpy()
    vel, primary_s = deriv(primary, t, smooth_win=smooth_win, poly=poly)
    df["primary_deg_smooth"] = primary_s
    df["primary_vel_deg_s"]  = vel

    direction = np.where(np.abs(df["primary_vel_deg_s"]) < vel_eps, 0, np.sign(df["primary_vel_deg_s"]))
    df["direction_raw"] = direction.astype(float)

    # Adaptive Schwellen
    valid = ~np.isnan(primary_s)
    if valid.sum() >= 10:
        top_deg = np.nanpercentile(primary_s, float(top_percentile))
        bottom_deg = np.nanpercentile(primary_s, float(bottom_percentile))
        top_deg = max(top_deg, float(clamp_top_min))
        bottom_deg = min(bottom_deg, float(clamp_bottom_max))
    else:
        top_deg, bottom_deg = 150.0, 60.0

    rsm = RepStateMachine(
        fps=fps,
        top_deg=top_deg, bottom_deg=bottom_deg,
        vel_eps=vel_eps, hysteresis=hysteresis,
        dwell_bottom_max=dwell_bottom_max,
        min_state_frames=min_state_frames,
        mean_k=mean_k,
        backslide_v=backslide_v
    )

    phases = []; reps=[]; statuses=[]; events=[]
    for i in range(len(df)):
        ph, dirn, rep_id, rep_stat, ev = rsm.update(
            t=df.at[i,"time_s"],
            elbow_deg=df.at[i,"primary_deg_smooth"],
            elbow_vel=df.at[i,"primary_vel_deg_s"])
        phases.append(ph); reps.append(rep_id); statuses.append(rep_stat); events.append(ev)

    df["phase"], df["rep_id"], df["rep_status"], df["event"] = phases, reps, statuses, events

    # Rolling-Mean Serien (nur zur Einsicht)
    df["primary_deg_meank"] = pd.Series(df["primary_deg_smooth"]).rolling(window=int(max(1,mean_k)), min_periods=1).mean()
    df["primary_vel_meank"] = pd.Series(df["primary_vel_deg_s"]).rolling(window=int(max(1,mean_k)), min_periods=1).mean()

    outdir = Path(outdir)
    out_csv = outdir / "features.csv"
    df.to_csv(out_csv, index=False)

    # Plots
    plt.figure(figsize=(12,5))
    plt.title("Bench Press – Shoulder angle & phases (YOLO pose)")
    plt.plot(df["time_s"], df["upperarm_torso_deg"], label="Shoulder (raw)", alpha=0.35)
    plt.plot(df["time_s"], df["primary_deg_smooth"], label="Shoulder smooth", linewidth=2)
    plt.plot(df["time_s"], df["primary_deg_meank"], label=f"Shoulder mean[{mean_k}]", linewidth=1.5)
    plt.axhline(top_deg, linestyle="--", label=f"Top~{top_deg:.0f}°")
    plt.axhline(bottom_deg, linestyle="--", label=f"Bottom~{bottom_deg:.0f}°")
    plt.xlabel("Zeit (s)"); plt.ylabel("Grad")

    # Rep-Marker
    mask_rep = df["event"].fillna("").eq("rep++")
    rep_times = df.loc[mask_rep, "time_s"].to_numpy()
    rep_ids   = df.loc[mask_rep, "rep_id"].to_numpy()
    y_series = df["primary_deg_smooth"]; y_max = float(np.nanmax(y_series))
    y_label = y_max + max(2.0, 0.02 * y_max)
    for t_rep, rid in zip(rep_times, rep_ids):
        plt.axvline(t_rep, linestyle="--", linewidth=1.2, alpha=0.7)
        plt.text(t_rep, y_label, f"{int(rid)}", rotation=90, va="bottom", ha="center", fontsize=9)
    if len(rep_times) > 0:
        y_at_rep = np.interp(rep_times, df["time_s"], y_series)
        plt.scatter(rep_times, y_at_rep, s=35, zorder=3)

    plt.legend(); ymin, ymax = plt.ylim(); plt.ylim(ymin, max(ymax, y_label + 1.0))
    plt.tight_layout(); plt.savefig(outdir / "elbow_timeseries.png", dpi=200); plt.close()

    total_reps = int(df["rep_id"].max() if len(df) else 0)
    fails = df["event"].fillna("").str.startswith("fail").sum()
    print(f"\n[OK] Annotiertes Video: {outdir/'annotated.mp4'}")
    print(f"[OK] Features: {out_csv}")
    print(f"[OK] Plot: {outdir/'elbow_timeseries.png'}")
    print(f"[INFO] Reps gezählt: {total_reps} | Events fail: {fails}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Pfad zum Eingabevideo (.mp4/.mov)")
    ap.add_argument("--outdir", default="runs/bench_yolo", help="Output-Ordner")
    ap.add_argument("--side", default="auto", choices=["left","right","auto"])
    ap.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics Pose-Checkpoint")
    ap.add_argument("--device", default=None, help="z.B. 'cuda:0' oder 'cpu'")
    # Stellschrauben
    ap.add_argument("--mean_k", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vel_eps", type=float, default=5.0)
    ap.add_argument("--hysteresis", type=float, default=10.0)
    ap.add_argument("--dwell_bottom_max", type=float, default=1.5)
    ap.add_argument("--min_state_frames", type=int, default=6)
    ap.add_argument("--backslide_v", type=float, default=5.0)
    ap.add_argument("--top_percentile", type=float, default=85.0)
    ap.add_argument("--bottom_percentile", type=float, default=15.0)
    ap.add_argument("--clamp_top_min", type=float, default=150.0)
    ap.add_argument("--clamp_bottom_max", type=float, default=110.0)
    # Glättung
    ap.add_argument("--smooth_win", type=int, default=9)
    ap.add_argument("--poly", type=int, default=2)

    args = ap.parse_args()
    process_video_yolo(
        video_path=args.video,
        outdir=args.outdir,
        side=args.side,
        model_name=args.model,
        device=args.device,
        vis=True,
        smooth_win=args.smooth_win,
        poly=args.poly,
        mean_k=args.mean_k,
        stride=args.stride,
        vel_eps=args.vel_eps,
        hysteresis=args.hysteresis,
        dwell_bottom_max=args.dwell_bottom_max,
        min_state_frames=args.min_state_frames,
        backslide_v=args.backslide_v,
        top_percentile=args.top_percentile,
        bottom_percentile=args.bottom_percentile,
        clamp_top_min=args.clamp_top_min,
        clamp_bottom_max=args.clamp_bottom_max,
    )
