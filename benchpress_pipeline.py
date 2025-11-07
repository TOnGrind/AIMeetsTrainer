import argparse, math, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
    if len(x) < smooth_win:  # Fallback
        vx = np.gradient(x, t)
        return vx, x
    xs = savgol_filter(x, smooth_win if smooth_win%2==1 else smooth_win+1, poly, mode="interp")
    vx = np.gradient(xs, t)
    return vx, xs

# ------------------ Auswahl linke/rechte Seite ------------------
# MediaPipe Pose 2D Landmarks (x,y) in Bildpixeln
def get_points(landmarks, w, h, side="auto"):
    # Indizes: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # Wir nutzen Schulter (11/12), Ellbogen (13/14), Handgelenk (15/16), Hüfte (23/24)
    L = {"SH":11, "EL":13, "WR":15, "HP":23}
    R = {"SH":12, "EL":14, "WR":16, "HP":24}
    conf = lambda idx: landmarks[idx].visibility if landmarks and 0<=idx<len(landmarks) else 0.0
    def xy(idx):
        l = landmarks[idx]
        return (l.x*w, l.y*h)

    if side == "auto":
        left_score = np.mean([conf(L["SH"]), conf(L["EL"]), conf(L["WR"]), conf(L["HP"])])
        right_score = np.mean([conf(R["SH"]), conf(R["EL"]), conf(R["WR"]), conf(R["HP"])])
        side = "left" if left_score >= right_score else "right"

    S = L if side=="left" else R
    SH, EL, WR, HP = xy(S["SH"]), xy(S["EL"]), xy(S["WR"]), xy(S["HP"])
    return side, SH, EL, WR, HP

# ------------------ Phasen-/Rep-Logik ------------------
class RepStateMachine:
    """
    Hauptsignal: Ellbogenwinkel (180° gestreckt oben, ~90° unten).
    DOWN = Winkel nimmt ab (neg. vel), UP = Winkel nimmt zu (pos. vel)
    TOP  = Winkel nahe obere Schwelle + geringe Geschwindigkeit
    BOTTOM = Winkel nahe untere Schwelle + geringe Geschwindigkeit
    """
    def __init__(self, fps, top_deg=None, bottom_deg=None,
                 vel_eps=5.0,    # °/s Grenze für Stillstand
                 dwell_bottom_max=1.5,  # s: zu lange unten = Fail
                 hysteresis=5.0): # Grad Hysterese für Top/Bottom
        self.fps = fps
        self.vel_eps = vel_eps
        self.dwell_bottom_max = dwell_bottom_max
        self.hyst = hysteresis
        self.rep_id = 0
        self.phase = "TOP"  # Start in Top (Arme gestreckt)
        self.in_bottom_since = None
        self.awaited_bottom = False  # erwartet, dass wir den unteren Totpunkt erreichen
        self.awaited_top = False     # erwartet, dass wir wieder Top erreichen
        self.last_event = None
        self.rep_status = "ok"       # ok / fail:<grund>
        self.top_deg = top_deg
        self.bottom_deg = bottom_deg

    def _is_top(self, elbow_deg):
        return elbow_deg >= (self.top_deg - self.hyst)

    def _is_bottom(self, elbow_deg):
        return elbow_deg <= (self.bottom_deg + self.hyst)

    def update(self, i, t, elbow_deg, elbow_vel):
        """
        Gibt (phase, direction, rep_id, rep_status) zurück.
        direction: -1 runter, +1 hoch, 0 still
        """
        # Richtung aus Geschwindigkeit
        if np.isnan(elbow_vel):
            direction = 0
        else:
            if abs(elbow_vel) < self.vel_eps: direction = 0
            elif elbow_vel < 0: direction = -1
            else: direction = +1

        # initiale Thresholds adaptiv bestimmen (bei erstem Call)
        if self.top_deg is None or self.bottom_deg is None:
            # Grobe Defaults, werden nach ersten Sekunden oft schon passend
            self.top_deg = 170.0 if self.top_deg is None else self.top_deg
            self.bottom_deg = 90.0 if self.bottom_deg is None else self.bottom_deg

        # Zustandslogik
        if self.phase == "TOP":
            if direction < 0:  # Bewegung abwärts startet
                self.phase = "DOWN"
                self.awaited_bottom = True
                self.rep_status = "ok"

        elif self.phase == "DOWN":
            if self._is_bottom(elbow_deg) and abs(elbow_vel) < self.vel_eps:
                self.phase = "BOTTOM"
                self.in_bottom_since = t
                self.awaited_bottom = False
            elif direction > 0:
                # Hat Richtung gewechselt bevor unten erreicht -> Abbruch
                if self.awaited_bottom:
                    self.rep_status = "fail:no_bottom"
                    self.phase = "UP"  # trotzdem hoch
                    self.awaited_bottom = False

        elif self.phase == "BOTTOM":
            if self.in_bottom_since is not None and (t - self.in_bottom_since) > self.dwell_bottom_max:
                self.rep_status = "fail:long_bottom"
            if direction > 0:
                self.phase = "UP"
                self.in_bottom_since = None
                self.awaited_top = True

        elif self.phase == "UP":
            if self._is_top(elbow_deg) and abs(elbow_vel) < self.vel_eps:
                # Rep abgeschlossen
                if self.rep_status.startswith("fail"):
                    # fehlgeschlagene Rep trotzdem zählen oder nicht?
                    # Wir schreiben sie als gleiche rep_id mit Status 'fail'
                    pass
                self.phase = "TOP"
                self.awaited_top = False
                self.rep_id += 1
                self.rep_status = "ok"
            elif direction < 0 and self.awaited_top:
                # Nach oben begonnen, aber wieder runter ohne Top zu erreichen
                self.rep_status = "fail:no_top"
                self.phase = "DOWN"
                self.awaited_top = False
                self.awaited_bottom = True

        return self.phase, direction, self.rep_id, self.rep_status

# ------------------ Hauptpipeline ------------------
def process_video(
    video_path,
    outdir="runs/bench",
    side="auto",
    vis=True,
    smooth_win=9,           # <-- einfacher, gültiger Default
    poly=2
):
    """
    Analyse Bankdrücken mit MediaPipe Pose.
    - side: 'left'|'right'|'auto' -> Seite für die Winkel (eine Seite reicht)
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None
    pbar = tqdm(total=frame_count, desc="Processing (MediaPipe)")
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        elbow_deg = np.nan; upperarm_torso_deg = np.nan; wrist_horiz_deg = np.nan
        chosen_side = side

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            chosen_side, SH, EL, WR, HP = get_points(lm, W, H, side)
            # Winkel:
            # 1) Ellbogenflexion (Schulter-ELLBOGEN-Handgelenk) -> 180 oben, kleiner unten
            elbow_deg = angle_at_point(SH, EL, WR)
            # 2) Oberarm ↔ Oberkörper (Winkel am Schultergelenk zwischen Oberkörper(HP->SH) und Oberarm(EL->SH))
            upperarm_torso_deg = angle_at_point(HP, SH, EL)
            # 3) "Handgelenk": Vorarm (EL->WR) gegen Horizontalen (Ausrichtung/Stacking)
            wrist_horiz_deg = angle_to_horizontal(EL, WR)

            if vis:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
                # Overlay Text
                y0 = 30
                cv2.putText(frame, f"Side: {chosen_side}", (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"Elbow: {elbow_deg:.1f} deg", (20,y0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"UpperArm-Torso: {upperarm_torso_deg:.1f} deg", (20,y0+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                cv2.putText(frame, f"Forearm-Horiz: {wrist_horiz_deg:.1f} deg", (20,y0+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)

        t = idx / fps
        rows.append([idx, t, elbow_deg, upperarm_torso_deg, wrist_horiz_deg])
        writer.write(frame)
        idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    cap.release()
    pose.close()

    df = pd.DataFrame(rows, columns=["frame","time_s","elbow_deg","upperarm_torso_deg","forearm_horiz_deg"])
    # Glätten & Ableitung
    t = df["time_s"].to_numpy()
    elbow = df["elbow_deg"].to_numpy()
    vel, elbow_s = deriv(elbow, t, smooth_win=9, poly=2)
    df["elbow_deg_smooth"] = elbow_s
    df["elbow_vel_deg_s"] = vel

    # Richtung: -1 / 0 / +1
    vel_eps = 5.0  # °/s
    direction = np.where(np.abs(df["elbow_vel_deg_s"])<vel_eps, 0, np.sign(df["elbow_vel_deg_s"]))
    df["direction"] = direction.astype(int)

    # Dynamische Top/Bottom-Schwellwerte (robust gegen Person/Setup)
    valid = ~np.isnan(elbow_s)
    if valid.sum() >= 10:
        top_deg = np.nanpercentile(elbow_s, 85)  # nahe Streckung
        bottom_deg = np.nanpercentile(elbow_s, 15)  # nahe Beugung
        # clamp vernünftig
        top_deg = max(top_deg, 150.0)
        bottom_deg = min(bottom_deg, 110.0)
    else:
        top_deg, bottom_deg = 170.0, 90.0

    # Zustandsmaschine
    rsm = RepStateMachine(fps=fps, top_deg=top_deg, bottom_deg=bottom_deg,
                          vel_eps=vel_eps, dwell_bottom_max=1.5, hysteresis=5.0)

    phases = []; reps=[]; statuses=[]
    for i in range(len(df)):
        ph, dirn, rep_id, rep_stat = rsm.update(
            i=i,
            t=df.at[i,"time_s"],
            elbow_deg=df.at[i,"elbow_deg_smooth"],
            elbow_vel=df.at[i,"elbow_vel_deg_s"])
        phases.append(ph); reps.append(rep_id); statuses.append(rep_stat)

    df["phase"] = phases
    df["rep_id"] = reps
    df["rep_status"] = statuses

    # Export
    out_csv = outdir / "features.csv"
    df.to_csv(out_csv, index=False)

    # Plots
    plt.figure(figsize=(12,5))
    plt.title("Bench Press – Ellbogenwinkel & Phasen")
    plt.plot(df["time_s"], df["elbow_deg"], label="Elbow raw", alpha=0.4)
    plt.plot(df["time_s"], df["elbow_deg_smooth"], label="Elbow smooth")
    plt.axhline(top_deg, linestyle="--", label=f"Top~{top_deg:.0f}°")
    plt.axhline(bottom_deg, linestyle="--", label=f"Bottom~{bottom_deg:.0f}°")
    plt.xlabel("Zeit (s)"); plt.ylabel("Grad")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "elbow_timeseries.png", dpi=200); plt.close()

    plt.figure(figsize=(12,2.6))
    plt.title("Phase (TOP=0, DOWN=1, BOTTOM=2, UP=3)")
    map_phase = {"TOP":0,"DOWN":1,"BOTTOM":2,"UP":3}
    plt.plot(df["time_s"], df["phase"].map(map_phase))
    plt.yticks([0,1,2,3], ["TOP","DOWN","BOTTOM","UP"])
    plt.xlabel("Zeit (s)")
    plt.tight_layout()
    plt.savefig(outdir / "phases.png", dpi=200); plt.close()

    print("\n[OK] Annotiertes Video:", out_vid)
    print("[OK] Features:", out_csv)
    print("[OK] Plots:", outdir/"elbow_timeseries.png", ",", outdir/"phases.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Pfad zum Eingabevideo (.mp4/.mov)")
    ap.add_argument("--outdir", default="runs/bench", help="Output-Ordner")
    ap.add_argument("--side", default="auto", choices=["left","right","auto"])
    args = ap.parse_args()
    process_video(args.video, outdir=args.outdir, side=args.side)
