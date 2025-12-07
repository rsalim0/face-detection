#!/usr/bin/env python3
import cv2, sys, json, time, argparse
from pathlib import Path
from mediapipe_utils import MPFaceDetector, MPFaceMesh

# -------- Paths --------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "trained_lbph_face_recognizer_model.yml"
LABELMAP_PATH = MODELS_DIR / "label_map.json"

# -------- CLI --------
ap = argparse.ArgumentParser(description="LBPH Predict using MediaPipe detection + movement tracking")
ap.add_argument("--threshold", type=float, default=80.0, help="LBPH distance cutoff (<= match, > Unknown)")
ap.add_argument("--min-conf", type=float, default=10.0, help="Minimum confidence percent to accept a match")
ap.add_argument("--camera", type=int, default=1, help="Camera index (default 1)")
ap.add_argument("--image", type=str, help="Run on a single image instead of webcam.")
ap.add_argument("--mp-model", type=int, default=0, help="MediaPipe model selection: 0 short-range, 1 full-range")
ap.add_argument("--mp-min-det", type=float, default=0.6, help="MediaPipe min detection confidence")
ap.add_argument("--show-mesh", action="store_true", help="Overlay MediaPipe Face Mesh during prediction")
args = ap.parse_args()

# -------- Checks --------
if not MODEL_PATH.exists():
    sys.exit(f"[Error] Trained model not found at: {MODEL_PATH}")
if not LABELMAP_PATH.exists():
    sys.exit(f"[Error] label_map.json not found at: {LABELMAP_PATH}")
if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    sys.exit("[Error] OpenCV contrib missing. Install: pip install opencv-contrib-python")

# -------- Load recognizer --------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

# -------- Load label map --------
with open(LABELMAP_PATH, "r") as f:
    name_to_id = json.load(f)
id_to_name = {int(v): k for k, v in name_to_id.items()}

# -------- Styles --------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)
FONT_THICK = 2
TAG_BG = (0, 160, 0)
TAG_H = 50
RECT_COLOR = (0, 255, 0)
RECT_THICK = 2

# -------- MP detector --------
mp_detector = MPFaceDetector(model_selection=args.mp_model, min_confidence=args.mp_min_det)
mp_mesh = MPFaceMesh(max_faces=1, refine_landmarks=True, min_detection_confidence=args.mp_min_det, min_tracking_confidence=0.5)

# -------- Tracking state --------
prev_face_center = None
prev_face_time = None
MOVEMENT_THRESHOLD = 5  # px


def recognize_in_frame(frame_bgr, threshold, min_conf, tnow):
    global prev_face_center, prev_face_time

    dets = mp_detector.detect(frame_bgr)
    accepted_count = 0

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)

    for det in dets:
        x, y, w, h = det.x, det.y, det.w, det.h
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        cv2.equalizeHist(roi, roi)

        pred_id, dist = recognizer.predict(roi)
        conf_pct = max(0.0, min(100.0, 100.0 - (dist / max(1e-6, threshold)) * 100.0))

        if dist <= threshold and conf_pct >= min_conf:
            accepted_count += 1
            name = id_to_name.get(pred_id, f"ID:{pred_id}")

            # Draw bbox and center
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), RECT_COLOR, RECT_THICK)
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 255, 255), -1)

            # Top tag with name/conf
            top = max(0, y - TAG_H - 10)
            cv2.rectangle(frame_bgr, (x - 2, top), (x + w + 2, top + TAG_H), TAG_BG, -1)
            label = f"{name}: {conf_pct:.1f}% (d={dist:.1f})"
            cv2.putText(frame_bgr, label, (x + 4, top + TAG_H - 15), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)

    return frame_bgr, accepted_count


# -------- Single-image mode --------
if args.image:
    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"[Error] Image not found: {img_path}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        sys.exit(f"[Error] Could not read image: {img_path}")

    if args.show_mesh:
        mesh_res = mp_mesh.process(frame)
        mp_mesh.draw(frame, mesh_res)
    out, nfaces = recognize_in_frame(frame, args.threshold, args.min_conf, time.time())
    cv2.putText(out, f"Accepted: {nfaces}", (10, 25), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(f"LBPH Predict MP - {img_path.name}", out)
    print(f"[info] Image: {img_path.name} | accepted faces: {nfaces} | thr={args.threshold} | minConf={args.min_conf}%")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mp_detector.close()
    mp_mesh.close()
    sys.exit(0)

# -------- Webcam mode --------
cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    mp_detector.close()
    sys.exit("[Error] Could not open camera. Check permissions.")

prev = time.time(); fps = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        print("[Warn] Failed to read frame.")
        break

    now = time.time()
    if args.show_mesh:
        mesh_res = mp_mesh.process(frame)
        mp_mesh.draw(frame, mesh_res)
    out, nfaces = recognize_in_frame(frame, args.threshold, args.min_conf, now)
    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, (now - prev))) if prev else 0.0
    prev = now
    hud = f"Accepted: {nfaces}  FPS: {fps:.1f}  thr={args.threshold:.0f}  minConf={args.min_conf:.0f}%  MP({args.mp_model},{args.mp_min_det})"
    cv2.putText(out, hud, (10, 25), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition + Movement (MediaPipe)", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_detector.close()
mp_mesh.close()
