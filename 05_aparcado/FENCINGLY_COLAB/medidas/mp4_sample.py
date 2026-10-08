"""Sample MP4 files and get duration/fps/resolution via cv2 metadata reads."""
import sys, pathlib, random, time, json
import cv2
import numpy as np

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")

# Main MP4 corpus directories
MP4_DIRS = {
    "contexto_largo/videos": BASE / "scratchpad/contexto_largo/videos",
    "height_study/mp4": BASE / "scratchpad/height_study/mp4",
    "detector/videos_faltan": BASE / "scratchpad/detector/videos_faltan",
    "detector/videos_cinco": BASE / "scratchpad/detector/videos_cinco",
    "detector/videos_llavador": BASE / "scratchpad/detector/videos_llavador",
}

TARGET_PER_DIR = 60  # sample ~60 per major dir = ~300 total

def get_mp4_info(path):
    """Read cap metadata (no frame decode)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = n_frames / fps if fps > 0 else None
    return {"fps": fps, "n_frames": n_frames, "width": w, "height": h, "duration_s": duration_s}

random.seed(42)
t0 = time.time()

print("Collecting MP4 file lists...", flush=True)
dir_samples = {}
for label, d in MP4_DIRS.items():
    if d.exists():
        files = list(d.glob("*.mp4"))
        print(f"  {label}: {len(files)} files", flush=True)
        n = min(TARGET_PER_DIR, len(files))
        dir_samples[label] = random.sample(files, n)

total_sample = sum(len(v) for v in dir_samples.values())
print(f"\nSampling {total_sample} MP4 files...", flush=True)

records = []
errors = []
for label, files in dir_samples.items():
    print(f"\n  Processing {label} ({len(files)} files)...", flush=True)
    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"    {i}/{len(files)}...", flush=True)
        try:
            info = get_mp4_info(f)
            if info:
                info["file"] = f.name
                info["dir"] = label
                info["size_bytes"] = f.stat().st_size
                records.append(info)
            else:
                errors.append({"file": str(f), "error": "could not open"})
        except Exception as e:
            errors.append({"file": str(f), "error": str(e)})

elapsed = time.time() - t0
print(f"\nLoaded {len(records)} OK, {len(errors)} errors in {elapsed:.1f}s", flush=True)

# ── Per-directory totals ────────────────────────────────────────────────────────
print("\n=== MP4 CORPUS TOTALS (from file system, not sample) ===")
print(f"{'DIRECTORY':<40} {'N_MP4':>8} {'SIZE_GB':>9} {'MEAN_MB':>9}")
print("-" * 70)
for label, d in MP4_DIRS.items():
    if d.exists():
        files = list(d.glob("*.mp4"))
        if files:
            sizes = [f.stat().st_size for f in files]
            total_gb = sum(sizes) / 1e9
            mean_mb = np.mean(sizes) / 1e6
            print(f"{label:<40} {len(files):>8} {total_gb:>9.2f} {mean_mb:>9.2f}")

# ── Duration distribution ──────────────────────────────────────────────────────
durations = [r["duration_s"] for r in records if r["duration_s"] is not None]
if durations:
    d_arr = np.array(durations)
    print(f"\n=== DURATION_S (sample n={len(d_arr)}) ===")
    print(f"  min={d_arr.min():.1f}  p05={np.percentile(d_arr,5):.1f}  median={np.median(d_arr):.1f}  "
          f"mean={d_arr.mean():.1f}  p95={np.percentile(d_arr,95):.1f}  max={d_arr.max():.1f}")

# ── FPS histogram ──────────────────────────────────────────────────────────────
fps_all = [r["fps"] for r in records if r["fps"] is not None]
fps_rounded = {}
for v in fps_all:
    k = round(v)
    fps_rounded[k] = fps_rounded.get(k, 0) + 1
print(f"\n=== FPS (rounded, sample n={len(fps_all)}) ===")
for k in sorted(fps_rounded):
    pct = 100 * fps_rounded[k] / len(fps_all)
    print(f"  fps~{k:<4}  n={fps_rounded[k]:<5}  ({pct:.1f}%)")

# ── Resolution histogram ───────────────────────────────────────────────────────
res_counts = {}
for r in records:
    k = f"{r['width']}x{r['height']}"
    res_counts[k] = res_counts.get(k, 0) + 1
print(f"\n=== RESOLUTION (sample n={len(records)}) ===")
for k in sorted(res_counts, key=lambda x: -res_counts[x]):
    pct = 100 * res_counts[k] / len(records)
    print(f"  {k:<12}  n={res_counts[k]:<5}  ({pct:.1f}%)")

# ── Size distribution ──────────────────────────────────────────────────────────
sizes = [r["size_bytes"] for r in records]
if sizes:
    s_arr = np.array(sizes, dtype=float) / 1e6  # MB
    print(f"\n=== MP4 SIZE_MB (sample n={len(s_arr)}) ===")
    print(f"  min={s_arr.min():.2f}  p05={np.percentile(s_arr,5):.2f}  median={np.median(s_arr):.2f}  "
          f"mean={s_arr.mean():.2f}  p95={np.percentile(s_arr,95):.2f}  max={s_arr.max():.2f}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = pathlib.Path(__file__).parent / "mp4_sample.json"
with open(out_path, "w") as f:
    json.dump({"records": records, "errors": errors}, f, indent=2)
print(f"\nSaved {len(records)} records to {out_path}")
