"""NPZ corpus inventory: count, bytes, index.csv status for every npz/ directory."""
import sys, os, json, csv, pathlib, time

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565")
DESKTOP = pathlib.Path(r"C:\Users\Alejandro\Desktop")

def find_npz_dirs(roots):
    """Find directories that contain *.npz files."""
    seen = set()
    dirs = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.npz"):
                d = p.parent
                if d not in seen:
                    seen.add(d)
                    dirs.append(d)
        except Exception as e:
            print(f"  Error scanning {root}: {e}", file=sys.stderr)
    return sorted(dirs)

def index_stats(npz_dir):
    """Return (has_index, n_ok_cached) for the index.csv in or next to npz_dir."""
    candidates = [
        npz_dir / "index.csv",
        npz_dir.parent / "index.csv",
    ]
    for idx in candidates:
        if idx.exists():
            try:
                ok_cached = 0
                with open(idx, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        s = (row.get("status") or "").strip().lower()
                        if s in ("ok", "cached"):
                            ok_cached += 1
                return True, ok_cached
            except Exception:
                return True, -1
    return False, 0

print("Scanning for NPZ directories...", flush=True)
t0 = time.time()
npz_dirs = find_npz_dirs([BASE, DESKTOP])
print(f"Found {len(npz_dirs)} NPZ directories in {time.time()-t0:.1f}s", flush=True)

rows = []
for d in npz_dirs:
    try:
        files = list(d.glob("*.npz"))
        n = len(files)
        total_bytes = sum(f.stat().st_size for f in files)
        mean_bytes = total_bytes // n if n > 0 else 0
        has_idx, n_ok = index_stats(d)
        # Make path relative to BASE for readability
        try:
            rel = str(d.relative_to(BASE))
        except ValueError:
            try:
                rel = str(d.relative_to(DESKTOP))
                rel = "DESKTOP/" + rel
            except ValueError:
                rel = str(d)
        rows.append({
            "dir": rel,
            "n_npz": n,
            "total_MB": round(total_bytes / 1e6, 1),
            "mean_KB": round(mean_bytes / 1e3, 1),
            "has_index": has_idx,
            "n_ok_cached": n_ok,
        })
    except Exception as e:
        print(f"  Error on {d}: {e}", file=sys.stderr)

# Print table
print(f"\n{'DIR':<70} {'N_NPZ':>7} {'TOTAL_MB':>10} {'MEAN_KB':>9} {'HAS_IDX':>8} {'N_OK':>7}")
print("-" * 115)
for r in rows:
    d = r['dir']
    if len(d) > 69:
        d = "..." + d[-66:]
    print(f"{d:<70} {r['n_npz']:>7} {r['total_MB']:>10} {r['mean_KB']:>9} {str(r['has_index']):>8} {r['n_ok_cached']:>7}")

total_n = sum(r['n_npz'] for r in rows)
total_mb = sum(r['total_MB'] for r in rows)
print(f"\nTOTAL: {len(rows)} directories, {total_n} NPZ files, {total_mb:.1f} MB")

out_path = pathlib.Path(__file__).parent / "npz_inventory.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved to {out_path}")
