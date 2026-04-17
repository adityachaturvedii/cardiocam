"""
Extract per-frame RGB + GT SpO2 from UBFC-rPPG DATASET_1.

Same approach as extract_rgb.py but for DATASET_1 which has SpO2 ground
truth in gtdump.xmp files (columns: timestamp_ms, HR, SpO2, PPG).

Usage:
  ./venv/bin/python benchmark/extract_rgb_ds1.py
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision

DATASET_ROOT = os.path.expanduser("~/UBFC_DATASET/DATASET_1")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ubfc_ds1_rgb")

LEFT_CHEEK = [345, 352, 411, 425, 266, 371, 355]
RIGHT_CHEEK = [116, 123, 187, 205, 36, 142, 126]
FOREHEAD = [107, 66, 69, 151, 299, 296, 336, 9]

_global_ts_ms = 0


def polygon_mean_rgb(frame, landmarks, indices, w, h):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)
    pixels = frame[mask == 1]
    if len(pixels) == 0:
        return None
    return float(pixels[:, 2].mean()), float(pixels[:, 1].mean()), float(pixels[:, 0].mean())


def load_ground_truth_ds1(subject_dir):
    """Load GT from gtdump.xmp: columns timestamp_ms, HR, SpO2, PPG."""
    gt = np.loadtxt(os.path.join(subject_dir, "gtdump.xmp"), delimiter=",")
    return gt  # shape (N, 4)


def extract_subject(vid_path, subject_dir, landmarker):
    global _global_ts_ms
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    gt = load_ground_truth_ds1(subject_dir)
    gt_ts_ms = gt[:, 0]
    gt_hr = gt[:, 1]
    gt_spo2 = gt[:, 2]
    gt_ppg = gt[:, 3]

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

        # Find closest GT sample to this frame's timestamp
        frame_time_ms = (frame_idx - 1) / fps * 1000
        gt_idx = np.argmin(np.abs(gt_ts_ms - frame_time_ms))

        rows.append({
            "frame": frame_idx,
            "r": r_val, "g": g_val, "b": b_val,
            "gt_hr": round(float(gt_hr[gt_idx]), 2),
            "gt_spo2": round(float(gt_spo2[gt_idx]), 1),
            "gt_ppg": round(float(gt_ppg[gt_idx]), 4),
            "fps": round(fps, 4),
        })
    cap.release()
    return pd.DataFrame(rows)


def main():
    subjects = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])
    print(f"Found {len(subjects)} subjects in DATASET_1")
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
            print(f"  {subj}: no vid.avi, skip")
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
