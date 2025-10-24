import cv2
from pathlib import Path

def mov2mp4(filename):
    # Aktuelles Verzeichnis
    current_dir = Path.cwd()
    input_path = current_dir / filename

    # Ausgabe-Dateiname (z.B. IMG_0318.mp4)
    output_path = current_dir / (input_path.stem + ".mp4")

    # Video öffnen
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Konnte {input_path} nicht öffnen.")
        return

    # Videoeigenschaften
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für MP4
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Konvertiere {input_path.name} → {output_path.name}")
    print(f"FPS: {fps}, Auflösung: {width}x{height}")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Konvertierung abgeschlossen.")


