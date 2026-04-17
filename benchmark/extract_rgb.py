"""
Extract per-frame RGB channel means from UBFC-rPPG v2 using MediaPipe.

Outputs one CSV per subject to benchmark/data/ubfc_rgb/ with columns:
  frame, r, g, b, gt_hr, fps

The TS evaluator (evaluate_ts.ts) reads these CSVs and runs the actual
shipped DSP pipeline on them.

Usage:
  cd Heart-rate-measurement-using-camera
  ./venv/bin/python benchmark/extract_rgb.py
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

DATASET_ROOT = os.path.expanduser("~/UBFC_DATASET/DATASET_2")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ubfc_rgb")

LEFT_CHEEK = [345, 352, 411, 425, 266, 371, 355]
RIGHT_CHEEK = [116, 123, 187, 205, 36, 142, 126]
FOREHEAD = [107, 66, 69, 151, 299, 296, 336, 9]


def polygon_mean_rgb(frame, landmarks, indices, w, h):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    pixels = frame[mask == 1]
    if len(pixels) == 0:
        return None
    return float(pixels[:, 2].mean()), float(pixels[:, 1].mean()), float(pixels[:, 0].mean())


def load_ground_truth(subject_dir):
    with open(os.path.join(subject_dir, "ground_truth.txt")) as f:
        lines = f.readlines()
    hr = [float(x) for x in lines[1].split()]
    return hr


_global_ts_ms = 0

def extract_subject(vid_path, subject_dir, landmarker):
    global _global_ts_ms
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    gt_hr = load_ground_truth(subject_dir)
    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_idx += 1
        _global_ts_ms += 33
        result = landmarker.detect_for_video(mp_image, _global_ts_ms)
        r_val, g_val, b_val = "", "", ""
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            vals = []
            for poly in [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]:
                v = polygon_mean_rgb(frame, lm, poly, w, h)
                if v is not None:
                    vals.append(v)
            if vals:
                r_val = round(np.mean([v[0] for v in vals]), 4)
                g_val = round(np.mean([v[1] for v in vals]), 4)
                b_val = round(np.mean([v[2] for v in vals]), 4)
        gt_idx = min(frame_idx - 1, len(gt_hr) - 1)
        rows.append({
            "frame": frame_idx,
            "r": r_val, "g": g_val, "b": b_val,
            "gt_hr": round(gt_hr[gt_idx], 2),
            "fps": round(fps, 4),
        })
    cap.release()
    return pd.DataFrame(rows)


def main():
    subjects = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("subject")
    ])
    print(f"Found {len(subjects)} subjects")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    for subj in subjects:
        out_path = os.path.join(OUTPUT_DIR, f"{subj}.csv")
        if os.path.exists(out_path):
            print(f"  {subj}: exists, skip")
            continue
        vid_path = os.path.join(DATASET_ROOT, subj, "vid.avi")
        if not os.path.exists(vid_path):
            continue
        print(f"  {subj}...", end=" ", flush=True)
        t0 = time.time()
        df = extract_subject(vid_path, os.path.join(DATASET_ROOT, subj), landmarker)
        df.to_csv(out_path, index=False)
        print(f"{len(df)} frames, {time.time()-t0:.1f}s")

    landmarker.close()
    print(f"\nDone. CSVs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
