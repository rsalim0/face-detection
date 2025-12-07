#!/usr/bin/env python3
import cv2
import time
import os
import argparse
from pathlib import Path
from mediapipe_utils import MPFaceDetector, MPFaceMesh

# ---- CLI ----
ap = argparse.ArgumentParser(description="Create face dataset using MediaPipe Face Detection")
ap.add_argument("--max-images", type=int, default=300, help="Max images to save")
ap.add_argument("--interval-ms", type=int, default=400, help="Interval between saves in ms")
ap.add_argument("--camera", type=int, default=1, help="Camera index")
ap.add_argument("--dataset-dir", type=str, default="dataset", help="Output dataset directory")
ap.add_argument("--model-selection", type=int, default=0, help="MediaPipe model selection (0 short-range, 1 full-range)")
ap.add_argument("--min-conf", type=float, default=0.6, help="Min detection confidence")
mesh_group = ap.add_mutually_exclusive_group()
mesh_group.add_argument("--mesh", dest="show_mesh", action="store_true", help="Enable MediaPipe Face Mesh overlay (preferred indicator)")
mesh_group.add_argument("--no-mesh", dest="show_mesh", action="store_false", help="Disable Face Mesh overlay")
ap.set_defaults(show_mesh=True)
args = ap.parse_args()

DATASET_DIR = Path(args.dataset_dir)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ---- Setup ----
detector = MPFaceDetector(model_selection=args.model_selection, min_confidence=args.min_conf)
face_mesh = MPFaceMesh(max_faces=1, refine_landmarks=True, min_detection_confidence=args.min_conf, min_tracking_confidence=0.5)

time.sleep(0.5)
ID = input('Enter your ID: ').strip()
print("Please get your face ready!")
time.sleep(2)

cam = cv2.VideoCapture(args.camera)
if not cam.isOpened():
    raise SystemExit("[Error] Could not open camera.")

win = "Dataset Generating (MediaPipe)"
cv2.namedWindow(win)

last_capture_ts = time.time()
image_count = 0
last_saved_flash_ts = 0.0
FLASH_MS = 200

while True:
    ok, frame = cam.read()
    if not ok:
        print("[Warn] Failed to read frame.")
        break

    # Create a copy of the original frame for saving purposes
    original_frame = frame.copy()

    detections = detector.detect(frame)

    # Optional mesh overlay
    if args.show_mesh:
        mesh_res = face_mesh.process(frame)
        face_mesh.draw(frame, mesh_res)

    # draw faces + capture
    captured_this_frame = False
    for det in detections:
        x, y, w, h = det.x, det.y, det.w, det.h
        # Prefer mesh overlay as the indicator; only draw rectangle if mesh is disabled
        if not args.show_mesh:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 2)

        elapsed_ms = (time.time() - last_capture_ts) * 1000.0
        if (elapsed_ms >= args.interval_ms) and (image_count < args.max_images) and not captured_this_frame:
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)  # Use the original frame here
            face_crop = gray[y:y + h, x:x + w]
            filename = DATASET_DIR / f"data.{ID}.{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(filename), face_crop)
            image_count += 1
            last_capture_ts = time.time()
            last_saved_flash_ts = last_capture_ts
            print(f"Saved [{image_count}/{args.max_images}]: {filename.name}")
            captured_this_frame = True
            break

    # ---- HUD overlay ----
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    hud = f"[MP] ID: {ID}  |  Saved: {image_count}/{args.max_images}  |  Interval: {args.interval_ms} ms  |  Press 'q' to quit"
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if (time.time() - last_saved_flash_ts) * 1000.0 <= FLASH_MS:
        text = "[OK]"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        pad_x, pad_y = 10, 10
        x0, y0 = 20, 80
        cv2.rectangle(frame, (x0 - pad_x, y0 - text_h - pad_y), (x0 + text_w + pad_x, y0 + baseline + pad_y), (0, 180, 0), -1)
        cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # progress bar
    bar_w = frame.shape[1] - 20
    pct = 0 if args.max_images == 0 else min(1.0, image_count / float(args.max_images))
    filled = int(bar_w * pct)
    y1 = frame.shape[0] - 20
    cv2.rectangle(frame, (10, y1), (10 + bar_w, y1 + 10), (255, 255, 255), 1)
    cv2.rectangle(frame, (10, y1), (10 + filled, y1 + 10), (0, 180, 0), -1)

    cv2.imshow(win, frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q') or image_count >= args.max_images:
        break

cam.release()
cv2.destroyAllWindows()

detector.close()
face_mesh.close()

print(f"\n√ Dataset generation complete.")
print(f"   ID: {ID}")
print(f"   Saved: {image_count} image(s) to '{DATASET_DIR}/'")
