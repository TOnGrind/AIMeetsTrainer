import argparse, math, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ------------------ Geometrie ------------------
def angle_at_point(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0: return np.nan
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def angle_to_horizontal(p, q):
    """Winkel des Vektors p->q zur Horizontalen (0° = exakt horizontal, 90° = vertikal)."""
    v = np.array(q, float) - np.array(p, float)
    if np.linalg.norm(v) == 0: return np.nan
    horiz = np.array([1.0, 0.0])
    cosang = np.clip(np.dot(v/np.linalg.norm(v), horiz), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def deriv(x, t, smooth_win=9, poly=2):
    """Ableitung (erste) über gleichabständige Zeit t; Savitzky-Golay glättet."""
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

# ------------------ Auswahl linke/rechte Seite ------------------
def get_points(landmarks, w, h, side="auto"):
    # Indizes: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    L = {"SH":11, "EL":13, "WR":15, "HP":23}
    R = {"SH":12, "EL":14, "WR":16, "HP":24}
    conf = lambda idx: landmarks[idx].visibility if landmarks and 0<=idx<len(landmarks) else 0.0
    def xy(idx):
        l = landmarks[idx]
        return (l.x*w, l.y*h)

    if side == "auto":
        left_score  = np.mean([conf(L["SH"]), conf(L["EL"]), conf(L["WR"]), conf(L["HP"])])
        right_score = np.mean([conf(R["SH"]), conf(R["EL"]), conf(R["WR"]), conf(R["HP"])])
        side = "left" if left_score >= right_score else "right"

    S = L if side=="left" else R
    SH, EL, WR, HP = xy(S["SH"]), xy(S["EL"]), xy(S["WR"]), xy(S["HP"])
    return side, SH, EL, WR, HP

# ------------------ Phasen-/Rep-Logik (robuster) ------------------
class RepStateMachine:
    """
    FSM mit:
      - completed_bottom-Gating: rep++ nur nach echtem Bottom
      - Top-Lock: Sperrzeit nach rep++, um Ghost-Reps oben zu vermeiden
      - Zeitbasierter Mindestdauer (_enough(t)) statt Frames
    """
    def __init__(self,
                 fps,
                 top_deg=None, bottom_deg=None,
                 vel_eps=3.0,
                 hysteresis=7.0,
                 dwell_bottom_max=1.5,
                 min_state_frames=2,
                 mean_k=5,
                 backslide_v=5.0,
                 top_lock_s=0.8          # <— NEU: Sperrzeit oben nach rep++
                 ):
        self.fps = float(fps)
        self.vel_eps = float(vel_eps)
        self.hyst = float(hysteresis)
        self.dwell_bottom_max = float(dwell_bottom_max)
        self.min_state_frames = int(max(1, min_state_frames))
        self.min_state_time_s = self.min_state_frames / max(1.0, self.fps)  # <— zeitbasiert
        self.backslide_v = float(backslide_v)

        self.top_deg = top_deg
        self.bottom_deg = bottom_deg

        self.phase = "TOP"
        self.rep_id = 0
        self.rep_status = "ok"

        self.in_bottom_since = None
        self.awaited_bottom = False
        self.awaited_top = False

        # Rep-Gating & Top-Lock
        self.completed_bottom = False      # <— Nur wenn True, darf oben gezählt werden
        self.last_rep_time = None          # <— Zeitpunkt des letzten rep++
        self.top_lock_s = float(top_lock_s)

        # Rolling Means
        self.rm_angle = RollingMean(k=mean_k)
        self.rm_vel   = RollingMean(k=mean_k)

        # Zeit-/Zustandsverwaltung
        self._phase_frames = 0
        self._state_enter_time = 0.0       # <— für _enough(t)

    def _is_top(self, elbow_deg_m):
        return elbow_deg_m >= (self.top_deg - self.hyst)
    def _is_bottom(self, elbow_deg_m):
        return elbow_deg_m <= (self.bottom_deg + self.hyst)
    def _still(self, vel_m):
        return abs(vel_m) < self.vel_eps
    def _enough(self, t):
        return (t - self._state_enter_time) >= self.min_state_time_s

    def _enter(self, new_phase, t):
        self.phase = new_phase
        self._phase_frames = 0
        self._state_enter_time = float(t)

    def update(self, t, elbow_deg, elbow_vel):
        """
        t: Zeit (s), elbow_deg/vel bereits geglättet; RollingMean mittelt zusätzlich.
        Returns: (phase, direction, rep_id, rep_status, event)
        """
        event = None
        if self.top_deg is None: self.top_deg = 170.0
        if self.bottom_deg is None: self.bottom_deg = 110.0

        ang_m = self.rm_angle.push(elbow_deg)
        vel_m = self.rm_vel.push(elbow_vel if not np.isnan(elbow_vel) else 0.0)

        if np.isnan(vel_m) or self._still(vel_m): direction = 0
        elif vel_m < 0: direction = -1
        else: direction = +1

        self._phase_frames += 1

        # TOP-Lock: nach rep++ kurze Sperrzeit für neue Bewegungen/Wechsel
        if self.phase == "TOP" and self.last_rep_time is not None:
            if (t - self.last_rep_time) < self.top_lock_s:
                # Sperre: bleib in TOP, kein Zustandswechsel
                return self.phase, 0, self.rep_id, self.rep_status, None

        if self.phase == "TOP":
            # TOP -> DOWN: klare Abwärtsbewegung + Top verlassen
            if direction < 0 and not self._is_top(ang_m) and self._enough(t):
                self.awaited_bottom = True
                self.rep_status = "ok"
                self._enter("DOWN", t)

 

        elif self.phase == "DOWN":
            # 1) Klassische Bottom-Erkennung (unten + still)
            if self._is_bottom(ang_m) and self._still(vel_m) and self._enough(t):
                self.in_bottom_since = t
                self.awaited_bottom = False
                self.completed_bottom = True
                self._enter("BOTTOM", t)

            # 2) NEU: Richtungswechsel im Bottom-Bereich gilt als "echter Bottom"
            elif (direction > 0) and self._is_bottom(ang_m) and self._enough(t):
                self.completed_bottom = True          # <— entscheidend
                self.awaited_bottom = False
                self.awaited_top = True
                self._enter("UP", t)


        elif self.phase == "BOTTOM":
            # Zu lange unten?
            if self.in_bottom_since is not None and (t - self.in_bottom_since) > self.dwell_bottom_max:
                if not str(self.rep_status).startswith("fail"):
                    self.rep_status = "fail:long_bottom"; event = "fail:long_bottom"
            # Start nach oben
            if direction > 0 and self._enough(t):
                self.awaited_top = True
                self.in_bottom_since = None
                self._enter("UP", t)

        elif self.phase == "UP":
            # Backslide (während UP wieder deutlich nach unten)
            if vel_m < -self.backslide_v and self._enough(t):
                self.rep_status = "fail:backslide_up"; event = "fail:backslide_up"
                self.awaited_top = False
                self.awaited_bottom = True
                self._enter("DOWN", t)
            # UP -> TOP: oben erreicht + Stillstand
            elif self._is_top(ang_m) and self._still(vel_m) and self._enough(t):
                # WICHTIG: rep++ nur wenn vorher echter BOTTOM
                if self.completed_bottom:
                    self.rep_id += 1
                    event = "rep++"
                    self.last_rep_time = t          # <— Start Top-Lock
                else:
                    # oben ohne echten Bottom -> als „no_bottom“ markieren (optional)
                    if not str(self.rep_status).startswith("fail"):
                        self.rep_status = "fail:no_bottom"
                        event = "fail:no_bottom"
                # Reset für nächsten Zyklus
                self.completed_bottom = False
                self.awaited_top = False
                self.rep_status = "ok" if event == "rep++" else self.rep_status
                self._enter("TOP", t)
            # Richtungswechsel nach unten, bevor TOP erreicht
            elif direction < 0 and self.awaited_top and self._enough(t):
                self.rep_status = "fail:no_top"; event = "fail:no_top"
                self.awaited_top = False
                self.awaited_bottom = True
                self._enter("DOWN", t)

        return self.phase, direction, self.rep_id, self.rep_status, event


# ------------------ Hauptpipeline ------------------
def process_video(
    video_path,
    outdir="runs/bench",
    side="auto",
    vis=True,
    # --- Glättung für Kurven (Savitzky-Golay, wie gehabt)
    smooth_win=9,
    poly=2,
    # --- Neue Stellschrauben (alle entschärft und parametrisierbar)
    mean_k=5,                  # Frames für Gleitmittel (FSM-Entscheidung)
    stride=1,                  # jeden n-ten Frame auswerten
    vel_eps=3.0,               # °/s Stillstand
    hysteresis=7.0,            # ° Hysterese Top/Bottom
    dwell_bottom_max=1.5,      # s
    min_state_frames=2,        # Frames Mindestdauer je Zustand
    backslide_v=5.0,           # °/s für Backslide-Fehler in UP
    top_percentile=85.0,       # adaptive Top-Schwelle (Perzentil geglätteter Winkel)
    bottom_percentile=15.0,    # adaptive Bottom-Schwelle
    clamp_top_min=150.0,       # harte Grenzen wie zuvor
    clamp_bottom_max=110.0
):
    """
    Analyse Bankdrücken mit MediaPipe Pose.
    - side: 'left'|'right'|'auto'
    - mean_k: Fenster für gleitendes Mittel der letzten k Frames (FSM-Entscheidungen!)
    - stride: nur jeden n-ten Frame verarbeiten (Performance / Rauschen)
    - vel_eps/hysteresis/dwell_bottom_max/min_state_frames/backslide_v: Akzeptanzkriterien
    - adaptive Top/Bottom per Perzentilen; zusätzlich hart geclamped.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = str(outdir / "annotated.mp4")
    writer = cv2.VideoWriter(out_vid, fourcc, fps, (W, H))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None
    pbar = tqdm(total=frame_total, desc="Processing (MediaPipe)")
    idx = 0
    shown_rep = -1  # um Rep-Inkremente am Overlay zu flashen

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Stride: nur jeden n-ten Frame *auswerten* (aber Video/CSV weiterführen)
        do_process = (idx % max(1, stride) == 0)

        elbow_deg = np.nan; upperarm_torso_deg = np.nan; wrist_horiz_deg = np.nan
        chosen_side = side

        if do_process:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                chosen_side, SH, EL, WR, HP = get_points(lm, W, H, side)
                # Winkel:
                elbow_deg = angle_at_point(SH, EL, WR)              # 1) Ellbogenflexion
                upperarm_torso_deg = angle_at_point(HP, SH, EL)     # 2) Oberarm vs Torso
                wrist_horiz_deg = angle_to_horizontal(EL, WR)       # 3) Vorarm vs Horizontal

                if vis:
                    mp_drawing.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
        else:
            # wenn wir nicht prozessieren, zeichnen wir nichts Neues
            res = None

        # Overlay Basistext
        if vis:
            y0 = 28
            cv2.putText(frame, f"Side: {chosen_side}", (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        t = idx / fps
        rows.append([idx, t, elbow_deg, upperarm_torso_deg, wrist_horiz_deg])
        writer.write(frame)
        idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    cap.release()
    pose.close()

    # --- DataFrame & Signale ---
    df = pd.DataFrame(rows, columns=["frame","time_s","elbow_deg","upperarm_torso_deg","forearm_horiz_deg"])

    # Savitzky-Golay (wie gehabt)
    t = df["time_s"].to_numpy()
    primary = df["upperarm_torso_deg"].to_numpy()           # <— Primärsignal!
    vel, primary_s = deriv(primary, t, smooth_win=smooth_win, poly=poly)
    df["primary_deg_smooth"] = primary_s
    df["primary_vel_deg_s"]  = vel

    direction = np.where(np.abs(df["primary_vel_deg_s"]) < vel_eps, 0,
                        np.sign(df["primary_vel_deg_s"]))
    df["direction_raw"] = direction.astype(float)


    # --- Adaptive Top/Bottom (entschärft), mit Clamps ---
    valid = ~np.isnan(primary_s)
    if valid.sum() >= 10:
        top_deg = np.nanpercentile(primary_s, float(top_percentile))
        bottom_deg = np.nanpercentile(primary_s, float(bottom_percentile))
        # Clamps ggf. an neue Winkelbereiche anpassen oder großzügiger machen:
        top_deg = max(top_deg, float(clamp_top_min))
        bottom_deg = min(bottom_deg, float(clamp_bottom_max))
    else:
    # Fallback-Werte für Schulterwinkel (konservativ):
        top_deg, bottom_deg = 150.0, 60.0


    # --- FSM (mit Rolling Means & gelockerten Kriterien) ---
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
    # Wir führen der FSM die geglätteten Reihen zu – sie mittelt zusätzlich über mean_k
    for i in range(len(df)):
        ph, dirn, rep_id, rep_stat, ev = rsm.update(
        t=df.at[i,"time_s"],
        elbow_deg=df.at[i,"primary_deg_smooth"],
        elbow_vel=df.at[i,"primary_vel_deg_s"])

        phases.append(ph)
        reps.append(rep_id)
        statuses.append(rep_stat)
        events.append(ev)


    df["phase"] = phases
    df["rep_id"] = reps
    df["rep_status"] = statuses
    df["event"] = events

    # Für Transparenz: Rolling-Mean-Reihen (optional rekonstruieren via erneutem Durchlauf)
    # Hier quick&dirty: gleitende Mittel über die geglätteten Werte (nur fürs CSV)
    df["primary_deg_meank"] = pd.Series(df["primary_deg_smooth"]).rolling(window=int(max(1,mean_k)), min_periods=1).mean()
    df["primary_vel_meank"] = pd.Series(df["primary_vel_deg_s"]).rolling(window=int(max(1,mean_k)), min_periods=1).mean()

    # Export
    outdir = Path(outdir)
    out_csv = outdir / "features.csv"
    df.to_csv(out_csv, index=False)

    # Plots
    plt.figure(figsize=(12,5))
    plt.title("Bench Press – Ellbogenwinkel & Phasen (entschärft, mean_k)")

    plt.plot(df["time_s"], df["upperarm_torso_deg"], label="Shoulder (raw)", alpha=0.35)
    plt.plot(df["time_s"], df["primary_deg_smooth"], label="Shoulder smooth", linewidth=2)
    plt.plot(df["time_s"], df["primary_deg_meank"], label=f"Shoulder mean[{mean_k}]", linewidth=1.5)
    plt.axhline(top_deg, linestyle="--", label=f"Top~{top_deg:.0f}°")
    plt.axhline(bottom_deg, linestyle="--", label=f"Bottom~{bottom_deg:.0f}°")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Grad")

    # --- Rep-Marker einfügen ---
    mask_rep = df["event"].fillna("").eq("rep++")
    rep_times = df.loc[mask_rep, "time_s"].to_numpy()
    rep_ids   = df.loc[mask_rep, "rep_id"].to_numpy()

    # y-Position der Labels knapp oberhalb der Kurve bestimmen
    y_series = df["primary_deg_smooth"]
    y_max = float(np.nanmax(y_series))
    y_label = y_max + max(2.0, 0.02 * y_max)  # etwas Abstand

    for t_rep, rid in zip(rep_times, rep_ids):
        plt.axvline(t_rep, linestyle="--", linewidth=1.2, alpha=0.7)
        plt.text(t_rep, y_label, f"{int(rid)}", rotation=90, va="bottom", ha="center", fontsize=9)

    # Optional: Marker-Punkte direkt auf der Kurve
    if len(rep_times) > 0:
        y_at_rep = np.interp(rep_times, df["time_s"], y_series)
        plt.scatter(rep_times, y_at_rep, s=35, zorder=3)

    plt.legend()

    # Falls Text oben abgeschnitten wäre → etwas mehr Platz lassen
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, max(ymax, y_label + 1.0))

    plt.tight_layout()
    plt.savefig(outdir / "elbow_timeseries.png", dpi=200)
    plt.close()


    # Kurzer Report
    total_reps = int(df["rep_id"].max() if len(df) else 0)
    fails = df["event"].fillna("").str.startswith("fail").sum()
    print(f"\n[OK] Annotiertes Video: {outdir/'annotated.mp4'}")
    print(f"[OK] Features: {out_csv}")
    print(f"[OK] Plots: {outdir/'elbow_timeseries.png'}, {outdir/'phases.png'}")
    print(f"[INFO] Reps gezählt: {total_reps} | Events fail: {fails}")

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Pfad zum Eingabevideo (.mp4/.mov)")
    ap.add_argument("--outdir", default="runs/bench", help="Output-Ordner")
    ap.add_argument("--side", default="auto", choices=["left","right","auto"])
    # Stellschrauben:
    ap.add_argument("--mean_k", type=int, default=5, help="Frames für gleitendes Mittel der FSM")
    ap.add_argument("--stride", type=int, default=1, help="Nur jeden n-ten Frame verarbeiten")
    ap.add_argument("--vel_eps", type=float, default=3.0, help="°/s-Kriterium für Stillstand")
    ap.add_argument("--hysteresis", type=float, default=7.0, help="Grad-Hysterese für Top/Bottom")
    ap.add_argument("--dwell_bottom_max", type=float, default=1.5, help="Sekunden bis 'zu lange unten'")
    ap.add_argument("--min_state_frames", type=int, default=2, help="Mindestdauer je Zustand (Frames)")
    ap.add_argument("--backslide_v", type=float, default=5.0, help="°/s Backslide-Grenze in UP")
    ap.add_argument("--top_percentile", type=float, default=85.0, help="Perzentil für Top-Schwelle")
    ap.add_argument("--bottom_percentile", type=float, default=15.0, help="Perzentil für Bottom-Schwelle")
    ap.add_argument("--clamp_top_min", type=float, default=150.0, help="harte Mindestgrenze Top")
    ap.add_argument("--clamp_bottom_max", type=float, default=110.0, help="harte Maxgrenze Bottom")
    # S-G Glättung:
    ap.add_argument("--smooth_win", type=int, default=9, help="Savitzky-Golay Fenster")
    ap.add_argument("--poly", type=int, default=2, help="Savitzky-Golay Polynomgrad")

    args = ap.parse_args()
    process_video(
        video_path=args.video,
        outdir=args.outdir,
        side=args.side,
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
