#!/usr/bin/env python3
"""
Quick dataset review using MediaPipe face detection.
- Opens each image from dataset/ recursively (skips .trash)
- Draws MediaPipe detection boxes and confidence
- Press any key to advance, 'b' to go back, 'q' to quit
"""
from pathlib import Path
import cv2
import argparse
from mediapipe_utils import MPFaceDetector, MPFaceMesh

ap = argparse.ArgumentParser(description="Review dataset images with MediaPipe detections and optional Face Mesh overlay")
ap.add_argument("--dataset", type=str, default="dataset", help="Dataset directory")
ap.add_argument("--min-conf", type=float, default=0.5, help="Min detection confidence")
ap.add_argument("--model-selection", type=int, default=0, help="MediaPipe model selection (0 short, 1 full)")
ap.add_argument("--show-mesh", action="store_true", help="Overlay MediaPipe Face Mesh on images")
args = ap.parse_args()

DS = Path(args.dataset)
if not DS.exists():
    raise SystemExit(f"[Error] Dataset not found: {DS}")

# Collect images
exts = {".jpg", ".jpeg", ".png", ".bmp"}
imgs = [p for p in sorted(DS.rglob("*")) if p.is_file() and p.suffix.lower() in exts and ".trash" not in p.parts and not p.name.startswith(".")]
if not imgs:
    raise SystemExit("[Error] No images found in dataset.")

fd = MPFaceDetector(model_selection=args.model_selection, min_confidence=args.min_conf)
fm = MPFaceMesh(max_faces=1, refine_landmarks=True, min_detection_confidence=args.min_conf, min_tracking_confidence=0.5)

i = 0
while 0 <= i < len(imgs):
    p = imgs[i]
    im = cv2.imread(str(p))
    if im is None:
        print(f"[Warn] cannot read {p}")
        i += 1
        continue

    dets = fd.detect(im)
    for d in dets:
        x, y, w, h = d.x, d.y, d.w, d.h
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(im, f"{d.score:.2f}", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # Optional mesh overlay
    if args.show_mesh:
        res = fm.process(im)
        fm.draw(im, res)

    top = f"[{i+1}/{len(imgs)}] {p.relative_to(DS)}  |  dets={len(dets)}  |  q=quit, any=next, b=back"
    cv2.rectangle(im, (0,0), (im.shape[1], 28), (0,0,0), -1)
    cv2.putText(im, top, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Dataset Review (MediaPipe)", im)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('b'):
        i = max(0, i-1)
    else:
        i += 1

cv2.destroyAllWindows()
fd.close()
fm.close()
