import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Utility: Winkel & Geometrie
# -----------------------------
def angle_at_point(a, b, c):
    """
    Winkel (in Grad) am Punkt b zwischen Segmenten ba und bc.
    a, b, c: (x, y) als float
    """
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    # numerische Stabilität:
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang

def angle_to_vertical(p_top, p_bottom):
    """
    Winkel (in Grad) des Segments p_bottom->p_top relativ zur Vertikalen.
    0° = perfekt senkrecht; größere Werte = Vorneige/Seitneige.
    """
    v = np.asarray(p_top, float) - np.asarray(p_bottom, float)
    # Vertikal-Vektor zeigt nach oben (0, -1) in Bildkoordinaten ist "nach oben" allerdings -y.
    # Wir nehmen hier die Geometrie unabhängig von Bildachsen:
    vertical = np.array([0.0, -1.0])
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return np.nan
    cosang = np.clip(np.dot(v / norm_v, vertical), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang

# -----------------------------
# COCO-Indices (Ultralytics Pose)
# 0:nose 1:eyeL 2:eyeR 3:earL 4:earR
# 5:shoulderL 6:shoulderR 7:elbowL 8:elbowR 9:wristL 10:wristR
# 11:hipL 12:hipR 13:kneeL 14:kneeR 15:ankleL 16:ankleR
# -----------------------------
IDX = {
    "LSHO":5, "RSHO":6, "LHIP":11, "RHIP":12, "LKNE":13, "RKNE":14, "LANK":15, "RANK":16
}

def pick_side_points(kpts_xy, kpts_conf, prefer_side="left"):
    """
    Wählt Punkte (Shoulder, Hip, Knee, Ankle) links oder rechts – fallback, wenn Punkte fehlen.
    prefer_side: 'left'|'right'|'auto' (auto nimmt die Seite mit höheren Mittel-Konfidenzen)
    """
    def side_score(side):
        ids = [IDX[f"{side.upper()[0]}SHO"], IDX[f"{side.upper()[0]}HIP"], IDX[f"{side.upper()[0]}KNE"], IDX[f"{side.upper()[0]}ANK"]]
        confs = [kpts_conf[i] if i < len(kpts_conf) else 0.0 for i in ids]
        return np.nanmean(confs)

    if prefer_side == "auto":
        side = "left" if side_score("l") >= side_score("r") else "right"
    else:
        side = prefer_side

    if side == "left":
        sh, hp, kn, an = IDX["LSHO"], IDX["LHIP"], IDX["LKNE"], IDX["LANK"]
        sh_alt, hp_alt, kn_alt, an_alt = IDX["RSHO"], IDX["RHIP"], IDX["RKNE"], IDX["RANK"]
    else:
        sh, hp, kn, an = IDX["RSHO"], IDX["RHIP"], IDX["RKNE"], IDX["RANK"]
        sh_alt, hp_alt, kn_alt, an_alt = IDX["LSHO"], IDX["LHIP"], IDX["LKNE"], IDX["LANK"]

    def safe_get(i, alt_i):
        # returns (x,y) or (nan, nan) if both missing
        p = kpts_xy[i] if i < len(kpts_xy) and not np.isnan(kpts_xy[i]).any() else None
        if p is None or (kpts_conf is not None and (i >= len(kpts_conf) or kpts_conf[i] < 0.2)):
            p = kpts_xy[alt_i] if alt_i < len(kpts_xy) and not np.isnan(kpts_xy[alt_i]).any() else None
            if p is None: return (np.nan, np.nan)
        return tuple(p)

    SHO = safe_get(sh, sh_alt)
    HIP = safe_get(hp, hp_alt)
    KNE = safe_get(kn, kn_alt)
    ANK = safe_get(an, an_alt)
    return SHO, HIP, KNE, ANK

# -----------------------------
# Visualisierungshilfen
# -----------------------------
def put_text(img, text, org, scale=0.7, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_angle_label(img, pos, label, value):
    if not (np.isnan(value)):
        put_text(img, f"{label}: {value:.1f}°", pos)

# -----------------------------
# Hauptpipeline
# -----------------------------
def process_video(video_path, outdir, model_name="yolo11n-pose.pt", side="auto", conf=0.5, device=None):
    """
    video_path: Pfad zur Video-Datei
    outdir: Ordner für Outputs
    model_name: 'yolo11n-pose.pt' (neu) oder 'yolov8n-pose.pt' (fallback)
    side: 'left' | 'right' | 'auto'
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Laden YOLO-Pose
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
    except Exception as e:
        print(f"[WARN] Konnte {model_name} nicht laden ({e}). Versuche 'yolov8n-pose.pt' ...")
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid_path = str(outdir / "annotated.mp4")
    writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (w, h))

    frame_idx = 0
    rows = []

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None,
                desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose schätzen
        results = model.predict(frame, conf=conf, verbose=False, device=device)
        # Wir nehmen die Person mit der höchsten Konfidenz (falls mehrere)
        if len(results) == 0 or len(results[0].keypoints) == 0:
            # nichts erkannt
            time_s = frame_idx / fps
            rows.append([frame_idx, time_s, np.nan, np.nan, np.nan, -1])
            put_text(frame, "No person detected", (20, 40), color=(0,0,255))
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
            continue

        # Wähle beste Detektion
        kps = results[0].keypoints
        # kps.xy: [n, 17, 2], kps.conf: [n, 17]
        # wähle detektion 0 (Ultralytics sortiert i. d. R. nach Conf); alternativ: max box conf
        det_id = 0
        kpts_xy = kps.xy[det_id].cpu().numpy()
        kpts_conf = None
        try:
            kpts_conf = kps.conf[det_id].cpu().numpy()
        except Exception:
            pass

        # Punkte auf gewünschter Seite holen
        SHO, HIP, KNE, ANK = pick_side_points(kpts_xy, kpts_conf, prefer_side=side)

        # Winkel berechnen
        # Kniebeugung: Winkel am Knie zwischen Oberschenkel (Hip->Knee) und Unterschenkel (Ankle->Knee)
        knee_flex = angle_at_point(HIP, KNE, ANK)
        # Hüftbeugung: Winkel an der Hüfte zwischen Oberkörper (Shoulder->Hip) und Oberschenkel (Knee->Hip)
        hip_flex  = angle_at_point(SHO, HIP, KNE)
        # Oberkörperwinkel (Trunk) vs. Vertikal: Segment Hip->Shoulder
        trunk_deg = angle_to_vertical(SHO, HIP)

        # Overlay
        if not np.isnan(knee_flex): draw_angle_label(frame, (20, 40), "Knie", knee_flex)
        if not np.isnan(hip_flex):  draw_angle_label(frame, (20, 70), "Hüfte", hip_flex)
        if not np.isnan(trunk_deg): draw_angle_label(frame, (20,100), "Oberkörper", trunk_deg)

        # optional: Pose zeichnen (nur wenige Segmente)
        def draw_seg(p, q, color=(0,255,0), th=3):
            if not (np.isnan(p[0]) or np.isnan(q[0])):
                cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, th)

        draw_seg(SHO, HIP)
        draw_seg(HIP, KNE)
        draw_seg(KNE, ANK)

        time_s = frame_idx / fps
        rows.append([frame_idx, time_s, hip_flex, knee_flex, trunk_deg, 0])

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    cap.release()

    # DataFrame & einfache Säuberung
    df = pd.DataFrame(rows, columns=["frame","time_s","hip_deg","knee_deg","trunk_deg","cluster"])
    # Interpolation (optional) nur für kleine Lücken
    df[["hip_deg","knee_deg","trunk_deg"]] = df[["hip_deg","knee_deg","trunk_deg"]].interpolate(limit=10, limit_direction="both")

    # -----------------------------
    # Clustering (K-Means)
    # -----------------------------
    feats = df[["hip_deg","knee_deg","trunk_deg"]].to_numpy()
    # robuster Standardisieren ist hier gar nicht zwingend nötig; für Winkel ok
    km = KMeans(n_clusters=3, random_state=42, n_init="auto")
    good = ~np.isnan(feats).any(axis=1)
    labels = np.full(len(df), -1, dtype=int)
    labels[good] = km.fit_predict(feats[good])

    # Labels sinnvoll benennen: sortiere Cluster nach mittlerem Knie-Winkel
    # große Knie-Winkel ≈ Streckung (oberer Totpunkt), kleine ≈ Beugung (unterer Totpunkt)
    cluster_means = {}
    for c in range(3):
        cluster_means[c] = np.nanmean(df.loc[labels==c, "knee_deg"])
    # sortiere: kleinster -> "unten", größter -> "oben", der Rest -> "zwischen"
    order = sorted(cluster_means, key=lambda c: cluster_means[c])
    mapping = {order[0]:"unten", order[1]:"zwischen", order[2]:"oben"}
    phase_numeric = {order[0]:0, order[1]:1, order[2]:2}

    df["cluster"] = labels
    df["phase"] = df["cluster"].map(mapping).fillna("unbekannt")
    df["phase_id"] = df["cluster"].map(phase_numeric).fillna(-1).astype(int)

    out_csv = outdir / "features.csv"
    df.to_csv(out_csv, index=False)

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure(figsize=(12,5))
    plt.title("Winkel über die Zeit")
    plt.plot(df["time_s"], df["knee_deg"], label="Knie")
    plt.plot(df["time_s"], df["hip_deg"], label="Hüfte")
    plt.plot(df["time_s"], df["trunk_deg"], label="Oberkörper")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Grad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "angles_timeseries.png", dpi=200)
    plt.close()

    # Phasen-Plot (als farbiges Band)
    plt.figure(figsize=(12,2.6))
    plt.title("Cluster/Phasen")
    # farbige Hinterlegung je Frame
    # 0=unten,1=zwischen,2=oben -> wir zeichnen als Linie
    plt.plot(df["time_s"], df["phase_id"])
    plt.yticks([0,1,2], ["unten","zwischen","oben"])
    plt.xlabel("Zeit (s)")
    plt.tight_layout()
    plt.savefig(outdir / "phases.png", dpi=200)
    plt.close()

    print(f"\n[OK] Annotiertes Video: {out_vid_path}")
    print(f"[OK] Features/Labels:  {out_csv}")
    print(f"[OK] Plots:            {outdir/'angles_timeseries.png'}, {outdir/'phases.png'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Pfad zum Eingabevideo (mp4, mov, ...)")
    ap.add_argument("--outdir", default="runs/squat", help="Output-Ordner")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics Pose Gewichte (z. B. yolo11n-pose.pt oder yolov8n-pose.pt)")
    ap.add_argument("--side", default="auto", choices=["left","right","auto"], help="Bevorzugte Seite für Winkelberechnung")
    ap.add_argument("--conf", type=float, default=0.5, help="Detektionskonfidenz")
    ap.add_argument("--device", default=None, help="z. B. 'cuda:0' oder 'cpu'")
    args = ap.parse_args()

    process_video(args.video, args.outdir, model_name=args.model, side=args.side, conf=args.conf, device=args.device)