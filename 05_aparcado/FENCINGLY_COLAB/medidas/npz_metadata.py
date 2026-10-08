"""Sample NPZ files and extract metadata for Colab sizing.

Loads at least 300 NPZ (spread across directories), reads meta_json, fits
bytes ~ a + b*n_frames_window by least squares.
"""
import sys, os, json, pathlib, random, time, traceback
import numpy as np

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")

# Directories to sample from (only real fencing clip NPZ, skip venv/fixtures)
SAMPLE_DIRS = [
    # Main corpus
    BASE / "localrun/runs_full/data_20260906T234700053391Z/npz",
    # Extra runs
    BASE / "localrun/runs_extra5/data_20260907T113112234735Z/npz",
    BASE / "localrun/runs_extra6/data_20260907T113112438707Z/npz",
    BASE / "localrun/runs_extra7/data_20260907T145758345522Z/npz",
    BASE / "localrun/runs_extra8/data_20260907T145758559720Z/npz",
    BASE / "localrun/runs_extra1/data_20260907T005507157743Z/npz",
    BASE / "localrun/runs_extra2/data_20260907T005507370214Z/npz",
    BASE / "localrun/runs_extra3/data_20260907T010246490954Z/npz",
    # Contexto largo (larger windows)
    BASE / "scratchpad/contexto_largo/ws_corto/npz",
    BASE / "scratchpad/contexto_largo/ws_largo/data/npz",
    # Ventana 500
    BASE / "scratchpad/ventana500/ws_corto/data/npz",
    BASE / "scratchpad/ventana500/ws_largo/npz",
    # Detector
    BASE / "scratchpad/detector/ws/cinco_local_0/npz",
    BASE / "scratchpad/detector/ws/cinco_resto_0/npz",
    BASE / "scratchpad/detector/ws/dobles_0/npz",
    BASE / "scratchpad/detector/ws/resto_0/npz",
    BASE / "scratchpad/detector/ws/resto_1/npz",
    # Height study
    BASE / "scratchpad/height_study/npz_prod",
    # Runs (small)
    BASE / "localrun/runs/data_20260906T222812804850Z/npz",
]

TARGET_TOTAL = 500  # aim for this many

def sample_npz_files():
    """Return a list of (path, size_bytes) sampled proportionally across dirs."""
    dir_files = {}
    for d in SAMPLE_DIRS:
        if d.exists():
            files = list(d.glob("*.npz"))
            if files:
                dir_files[d] = files

    total_available = sum(len(v) for v in dir_files.values())
    print(f"Available: {total_available} NPZ files across {len(dir_files)} directories", flush=True)

    sample = []
    for d, files in dir_files.items():
        n = max(5, round(len(files) / total_available * TARGET_TOTAL))
        n = min(n, len(files))
        chosen = random.sample(files, n)
        for f in chosen:
            sample.append((f, f.stat().st_size))

    random.shuffle(sample)
    return sample

def load_meta(path):
    """Load NPZ and return (meta_dict, {key: (shape, dtype)}, size_bytes)."""
    data = np.load(str(path), allow_pickle=False)
    meta = json.loads(data["meta_json"].item())
    arrays = {k: (data[k].shape, str(data[k].dtype)) for k in data.files if k != "meta_json"}
    data.close()
    return meta, arrays

random.seed(42)
t0 = time.time()

print("Sampling NPZ files...", flush=True)
sample = sample_npz_files()
print(f"Sampled {len(sample)} files. Loading metadata...", flush=True)

records = []
errors = []
array_schema = None  # from first successful load

for i, (path, size_bytes) in enumerate(sample):
    if i % 50 == 0:
        print(f"  {i}/{len(sample)}...", flush=True)
    try:
        meta, arrays = load_meta(path)
        if array_schema is None:
            array_schema = arrays

        rec = {
            "path": str(path),
            "size_bytes": size_bytes,
            "fps": meta.get("fps"),
            "n_frames_window": meta.get("n_frames_window"),
            "pre_s_eff": meta.get("pre_s_eff"),
            "t_window_start_s": meta.get("t_window_start_s"),
            "t_window_end_s": meta.get("t_window_end_s"),
            "f_touch": meta.get("f_touch"),
            # Look for source video fields
            "n_frames": meta.get("n_frames"),
            "duration_s": meta.get("duration_s"),
            "video_frames": meta.get("video_frames"),
            "source_frames": meta.get("source_frames"),
            "t_touch_s": meta.get("t_touch_s"),
            "n_frames_total": meta.get("n_frames_total"),
        }
        records.append(rec)
    except Exception as e:
        errors.append({"path": str(path), "error": str(e)})

print(f"\nLoaded {len(records)} OK, {len(errors)} errors in {time.time()-t0:.1f}s", flush=True)

# ── Array schema ──────────────────────────────────────────────────────────────
print("\n=== ARRAY SCHEMA (from first NPZ) ===")
if array_schema:
    for k, (shape, dtype) in sorted(array_schema.items()):
        print(f"  {k:<30} shape={shape}  dtype={dtype}")
else:
    print("  (none)")

# ── FPS histogram ─────────────────────────────────────────────────────────────
fps_vals = [r["fps"] for r in records if r["fps"] is not None]
fps_counts = {}
for v in fps_vals:
    fps_counts[v] = fps_counts.get(v, 0) + 1
print("\n=== FPS HISTOGRAM ===")
for fps_val in sorted(fps_counts):
    pct = 100 * fps_counts[fps_val] / len(fps_vals)
    print(f"  fps={fps_val:<8}  n={fps_counts[fps_val]:<6}  ({pct:.1f}%)")

# ── n_frames_window distribution ──────────────────────────────────────────────
nfw = [r["n_frames_window"] for r in records if r["n_frames_window"] is not None]
if nfw:
    nfw_arr = np.array(nfw, dtype=float)
    print("\n=== N_FRAMES_WINDOW ===")
    print(f"  n={len(nfw_arr)}  min={nfw_arr.min():.0f}  p05={np.percentile(nfw_arr,5):.0f}  "
          f"median={np.median(nfw_arr):.0f}  mean={nfw_arr.mean():.1f}  "
          f"p95={np.percentile(nfw_arr,95):.0f}  max={nfw_arr.max():.0f}")

# ── pre_s_eff distribution ────────────────────────────────────────────────────
pre = [r["pre_s_eff"] for r in records if r["pre_s_eff"] is not None]
if pre:
    pre_arr = np.array(pre, dtype=float)
    frac_below4 = (pre_arr < 4.0).mean()
    print("\n=== PRE_S_EFF ===")
    print(f"  n={len(pre_arr)}  min={pre_arr.min():.2f}  median={np.median(pre_arr):.2f}  "
          f"mean={pre_arr.mean():.2f}  max={pre_arr.max():.2f}  frac<4.0={frac_below4:.3f}")

# ── bytes vs n_frames_window regression ───────────────────────────────────────
sz = [r["size_bytes"] for r in records if r["size_bytes"] is not None and r["n_frames_window"] is not None]
nfw2 = [r["n_frames_window"] for r in records if r["size_bytes"] is not None and r["n_frames_window"] is not None]
if sz and nfw2:
    sz_arr = np.array(sz, dtype=float)
    nfw_arr2 = np.array(nfw2, dtype=float)
    # OLS: bytes = a + b * n_frames
    X = np.column_stack([np.ones_like(nfw_arr2), nfw_arr2])
    result = np.linalg.lstsq(X, sz_arr, rcond=None)
    coeffs = result[0]
    a, b = coeffs[0], coeffs[1]
    pred = a + b * nfw_arr2
    ss_res = np.sum((sz_arr - pred) ** 2)
    ss_tot = np.sum((sz_arr - sz_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    residuals = sz_arr - pred
    print("\n=== BYTES vs N_FRAMES_WINDOW (OLS: bytes = a + b*T) ===")
    print(f"  a={a:.1f} bytes (intercept)")
    print(f"  b={b:.1f} bytes/frame")
    print(f"  R²={r2:.4f}")
    print(f"  residual std={residuals.std():.1f}  p05={np.percentile(residuals,5):.1f}  p95={np.percentile(residuals,95):.1f}")
    print(f"  n={len(sz_arr)}")
    # Illustrative extrapolations
    print("\n  Extrapolation examples:")
    for T in [100, 200, 300, 500, 1000, 1500]:
        print(f"    T={T:>5}  predicted_bytes={a + b*T:>10.0f}  ({(a+b*T)/1e6:.2f} MB)")

# ── Source video duration fields ───────────────────────────────────────────────
print("\n=== SOURCE VIDEO FIELDS (presence in sampled records) ===")
source_fields = ["n_frames", "duration_s", "video_frames", "source_frames", "t_touch_s", "n_frames_total"]
for f in source_fields:
    vals = [r[f] for r in records if r[f] is not None]
    pct = 100 * len(vals) / len(records) if records else 0
    if vals:
        arr = np.array(vals, dtype=float)
        print(f"  {f:<20}: {len(vals)} present ({pct:.0f}%)  min={arr.min():.0f}  median={np.median(arr):.0f}  max={arr.max():.0f}")
    else:
        print(f"  {f:<20}: absent in all samples")

# ── Scale factor: source_frames / n_frames_window ─────────────────────────────
for sf in source_fields:
    pairs = [(r[sf], r["n_frames_window"]) for r in records
             if r[sf] is not None and r["n_frames_window"] is not None and r["n_frames_window"] > 0]
    if pairs:
        ratios = np.array([s / w for s, w in pairs], dtype=float)
        print(f"\n  {sf}/n_frames_window: n={len(ratios)}  min={ratios.min():.1f}  "
              f"median={np.median(ratios):.1f}  mean={ratios.mean():.1f}  p95={np.percentile(ratios,95):.1f}  max={ratios.max():.1f}")

# ── Save results ─────────────────────────────────────────────────────────────
out_path = pathlib.Path(__file__).parent / "npz_metadata.json"
with open(out_path, "w") as f:
    json.dump({"records": records, "errors": errors, "array_schema": {k: {"shape": v[0], "dtype": v[1]} for k, v in (array_schema or {}).items()}}, f, indent=2)
print(f"\nSaved {len(records)} records to {out_path}")
if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors[:5]:
        print(f"  {e}")
