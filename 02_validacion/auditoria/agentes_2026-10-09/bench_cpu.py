"""CPU-only micro-benchmarks for the FENCINGLY_PROYECTO speed audit.
Read-only: does not import torch.cuda, does not touch the GPU, does not write
to the project tree. Writes nothing; only prints.
"""
import os
import sys
import time
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
os.environ.setdefault("HF_HOME", "D:/FENCINGLY_HF")

import rutas as RU  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

VIDEOS_DIR = pathlib.Path("D:/FENCINGLY_VIDEOS_RAW/_descargados")
clips = sorted(VIDEOS_DIR.glob("*.mp4"))[:6]
print("sample clips:", [c.name for c in clips])

# ---- 1. full decode cost (cv2.VideoCapture, CPU software H.264 decode) ----
decode_times = []
frame_counts = []
for c in clips:
    cap = cv2.VideoCapture(str(c))
    t0 = time.perf_counter()
    n = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        n += 1
    dt = time.perf_counter() - t0
    cap.release()
    decode_times.append(dt)
    frame_counts.append(n)
    print("decode %-28s frames=%4d  t=%.3fs  (%.2f ms/frame)" % (c.name, n, dt, 1000 * dt / max(n, 1)))

print("\nTOTAL decode: n_clips=%d  sum_t=%.2fs  mean_t=%.3fs  mean_frames=%.0f"
      % (len(clips), sum(decode_times), sum(decode_times) / len(decode_times),
         sum(frame_counts) / len(frame_counts)))

# ---- 2. gris computation cost (resize 160x90 + cvtColor), as in fase1 ----
cap = cv2.VideoCapture(str(clips[0]))
frames = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    frames.append(im)
cap.release()
t0 = time.perf_counter()
gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32") for f in frames]
dt = time.perf_counter() - t0
print("\ngris (resize+cvtColor) on %d frames: %.3fs (%.2f ms/frame)" % (len(frames), dt, 1000 * dt / len(frames)))

# ---- 3. HF Sapiens2ImageProcessor CPU preprocessing cost ----
try:
    from transformers import Sapiens2ImageProcessor
    snap = RU.hf_modelo("sapiens2-pose-0.4b")
    proc = Sapiens2ImageProcessor.from_pretrained(str(snap))
    proc.size = {"height": 384, "width": 288}
    print("\nloaded processor from", snap)

    H, W = frames[0].shape[:2]
    rgb = np.ascontiguousarray(frames[0][:, :, ::-1])
    box = [[0.2 * W, 0.15 * H, 0.35 * W, 0.70 * H]]  # plausible fencer crop, xywh in px

    for batch in (1, 2, 4, 8):
        imgs = [rgb] * batch
        boxes = [box] * batch
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            e = proc(images=imgs, boxes=boxes, return_tensors="pt")
        dt = (time.perf_counter() - t0) / reps
        print("processor batch=%d: %.4f s/call  (%.4f s/crop)" % (batch, dt, dt / batch))
except Exception as ex:
    print("\nprocessor benchmark failed: %r" % (ex,))
