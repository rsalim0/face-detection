#!/usr/bin/env python3
"""
Train LBPH using faces detected with MediaPipe.
This mirrors 03_train_model.py but swaps Haar detection for MediaPipe.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mediapipe_utils import MPFaceDetector

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATASET = ROOT / "dataset"
MODEL_YML = MODELS / "trained_lbph_face_recognizer_model.yml"
LABELMAP = MODELS / "label_map.json"

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Train LBPH using MediaPipe detection with optional validation")
ap.add_argument("--val-split", type=float, default=0.0, help="Fraction per label for validation (e.g., 0.2)")
ap.add_argument("--threshold", type=float, default=80.0, help="Distance threshold for 'Unknown' during validation")
ap.add_argument("--min-size", type=int, default=60, help="Min face size to accept (pixels)")
ap.add_argument("--unknown-csv", default="", help="If set, write Unknown validation samples to this CSV path")
ap.add_argument("--min-conf", type=float, default=0.5, help="MediaPipe min detection confidence")
ap.add_argument("--model-selection", type=int, default=0, help="MediaPipe model selection (0 short-range, 1 full-range)")
args = ap.parse_args()

# ---------- Helpers ----------
def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(folder.rglob("*")):
        if ".trash" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            return_path = p
            yield return_path


def label_from_name(p: Path):
    parent = p.parent.name
    if parent not in ('.trash', 'dataset'):
        return parent
    parts = p.name.split(".")
    if len(parts) > 2 and parts[0].lower() == "data":
        return parts[1]
    stem = p.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return None


def load_data(ds: Path, mp_min_conf: float, mp_model_sel: int, min_size: int):
    """
    Load images, detect largest face via MediaPipe (if any), equalize, resize (200x200).
    Returns faces, labels, paths
    """
    faces, labels, paths = [], [], []
    detector = MPFaceDetector(model_selection=mp_model_sel, min_confidence=mp_min_conf)
    try:
        for imgp in iter_images(ds):
            lbl = label_from_name(imgp)
            if not lbl:
                continue
            try:
                g = np.array(Image.open(imgp).convert("L"), dtype="uint8")
                g = cv2.equalizeHist(g)
                # run detection on BGR frame; convert back from grayscale for detector
                bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                dets = detector.detect(bgr)
                if dets:
                    # pick largest by area
                    x, y, w, h = max(((d.x, d.y, d.w, d.h) for d in dets), key=lambda r: r[2]*r[3])
                    if w >= min_size and h >= min_size:
                        roi = g[y:y+h, x:x+w]
                    else:
                        roi = g
                else:
                    roi = g
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                faces.append(roi)
                labels.append(lbl)
                paths.append(str(imgp))
            except Exception as e:
                print(f"[Warn] skip {imgp}: {e}")
    finally:
        detector.close()
    return faces, labels, paths


def encode_labels(names):
    uniq = sorted(set(names))
    name_to_id = {n: i for i, n in enumerate(uniq)}
    ids = np.array([name_to_id[n] for n in names], np.int32)
    return ids, name_to_id


def stratified_split(ids, frac):
    by = defaultdict(list)
    for i, l in enumerate(ids):
        by[int(l)].append(i)
    tr, va = [], []
    for l, idxs in by.items():
        idxs = idxs[:]
        random.shuffle(idxs)
        n_val = int(len(idxs) * frac)
        va += idxs[:n_val]; tr += idxs[n_val:]
    return tr, va


def train_lbph(faces, ids, idxs):
    recog = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    X = [faces[i] for i in idxs]
    y = np.array([ids[i] for i in idxs], np.int32)
    recog.train(X, y)
    return recog


def simple_eval(recog, faces, ids, paths, threshold, num_labels):
    unknown = num_labels
    cm = np.zeros((num_labels, num_labels + 1), dtype=int)
    total = len(faces); correct = 0; unk = 0
    unk_rows = []
    for img, true_id, pth in zip(faces, ids, paths):
        pred_id, dist = recog.predict(img)
        if dist > threshold:
            cm[true_id, unknown] += 1
            unk += 1
            unk_rows.append((pth, dist))
        else:
            cm[true_id, pred_id] += 1
            if pred_id == true_id:
                correct += 1
    acc = correct / total if total else 0.0
    return acc, unk, cm, unk_rows


def print_confusion(cm, id_to_name):
    names = [id_to_name[i] for i in range(len(id_to_name))]
    print("\nConfusion matrix (rows=true, cols=pred, last col=Unknown):")
    header = ["true\\pred"] + names + ["Unknown"]
    colw = max(8, max(len(h) for h in header))
    print(" ".join(h.ljust(colw) for h in header))
    for i, row in enumerate(cm):
        cells = [names[i].ljust(colw)] + [str(v).ljust(colw) for v in row]
        print(" ".join(cells))


# ---------- Main ----------
def main():
    random.seed(42)
    if not DATASET.exists():
        raise SystemExit(f"[Error] Dataset not found: {DATASET}")

    faces, names, paths = load_data(DATASET, args.min_conf, args.model_selection, args.min_size)
    print(f"[info] loaded {len(faces)} samples from {DATASET}")
    if faces:
        c = Counter(names)
        print("[info] per-label counts:", dict(c))
    else:
        raise SystemExit("[Error] No usable images.")

    ids, name_to_id = encode_labels(names)
    id_to_name = {v: k for k, v in name_to_id.items()}

    if args.val_split > 0:
        tr_idx, va_idx = stratified_split(ids, args.val_split)
        recog = train_lbph(faces, ids, tr_idx)
        Xv = [faces[i] for i in va_idx]
        yv = [int(ids[i]) for i in va_idx]
        pv = [paths[i] for i in va_idx]
        acc, unk_count, cm, unk_rows = simple_eval(recog, Xv, yv, pv, args.threshold, len(name_to_id))
        total_samples = len(faces)
        val_count = len(Xv)
        train_count = len(tr_idx)
        print("\n------------------ Validation Summary ------------------")
        print(f"Total dataset size : {total_samples} image(s)")
        print(f"Validation split   : {args.val_split*100:.0f}%  →  {val_count} image(s) used for testing")
        print(f"Training set size  : {train_count} image(s)")
        print(f"Threshold distance : {args.threshold} (faces with distance > {args.threshold} are 'Unknown')")
        print("--------------------------------------------------------")
        print(f"\n[VAL] acc={acc:.3f}  (threshold={args.threshold})   n={val_count}")
        print(f"[VAL] Unknown (rejected): {unk_count}")
        print_confusion(cm, id_to_name)
        if args.unknown_csv and unk_rows:
            out = Path(args.unknown_csv); out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["path", "dist"])
                for p, d in unk_rows:
                    w.writerow([p, f"{d:.3f}"])
            print(f"[OK] Unknown list → {out}")
        MODELS.mkdir(parents=True, exist_ok=True)
        recog.save(str(MODEL_YML))
    else:
        recog = train_lbph(faces, ids, list(range(len(faces))))
        MODELS.mkdir(parents=True, exist_ok=True)
        recog.save(str(MODEL_YML))

    with open(LABELMAP, "w") as f:
        json.dump(name_to_id, f, indent=2)

    print(f"\n[OK] Model → {MODEL_YML}")
    print(f"[OK] Label map → {LABELMAP}")
    print("     Labels:", ", ".join(f"{k}:{v}" for k, v in name_to_id.items()))


if __name__ == "__main__":
    main()
