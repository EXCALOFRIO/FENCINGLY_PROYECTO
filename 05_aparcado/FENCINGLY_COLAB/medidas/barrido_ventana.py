"""Barrido de ventanas de extraccion sobre los artefactos de QC ya medidos. No re-extrae nada.

Para cada candidato (pre_s, post_s) calcula, sobre los 15.451 clips con QC utilizable:
  - cuadros a decodificar por clip = min(frames_clip, f1 - f0) con la geometria real de `lights.window`
  - bytes de presupuesto de decodificacion = cuadros * W * H * 3 * DECODE_COPY_FACTOR(2)
  - cuantos clips superan el tope de `max_decoded_bytes` de cada runtime de Colab
  - las dos banderas BLOQUEANTES que dependen de la ventana (cut_in_window / replay_in_window)
  - si la ventana cubre la segunda luz y la ventana de guardia [t-4.0, t-3.5]

`use_clock=False` en todos los candidatos salvo el desplegado, porque el reloj recorta el inicio.
"""
import json
import math
import sys
from pathlib import Path

QC = Path(sys.argv[1])
DECODE_COPY_FACTOR = 2
GiB = 1024 ** 3

# (nombre, pre_s, post_s, use_clock)
CANDIDATOS = [
    ("desplegado 4.0/0.3 +reloj", 4.0, 0.3, True),
    ("4.0/0.3 sin reloj", 4.0, 0.3, False),
    ("6.0/1.0", 6.0, 1.0, False),
    ("8.0/3.0", 8.0, 3.0, False),
    ("12.0/6.0", 12.0, 6.0, False),
    ("DURACION COMPLETA 60/60", 60.0, 60.0, False),
]
TOPES = {"T4 estandar (0.35*11 GiB)": 3.85 * GiB, "GPU High-RAM (tope 6.0 GiB)": 6.0 * GiB,
         "si el tope 6.0 sube a 12 GiB": 12.0 * GiB}

filas = []
for p in sorted(QC.glob("*.json")):
    try:
        q = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    dur = q.get("duration_s")
    fps = q.get("video_fps") or q.get("fps")
    tt = q.get("t_touch_s")
    w, h = q.get("w"), q.get("h")
    if not dur or not fps or tt is None or not w or not h:
        continue
    clk = q.get("clock_start_s")
    filas.append({"dur": float(dur), "fps": float(fps), "tt": float(tt), "w": int(w), "h": int(h),
                  "clk": None if clk is None else float(clk),
                  "cuts": [float(c) for c in (q.get("cuts_t") or [])],
                  "reps": [(float(r[0]), float(r[1])) for r in (q.get("replays") or []) if len(r) >= 2],
                  "otras": [f for f in (q.get("blocking") or []) if f not in ("cut_in_window", "replay_in_window")],
                  # dt entre luces: cuanto despues del tocado llega la segunda luz
                  "dt_ms": q.get("dt_ms")})

n = len(filas)
print(f"clips con QC utilizable: {n}\n")
CLOCK_BIAS_S, MIN_PRE_S = 0.5, 1.5

print(f"{'candidato':28} {'cuadros med':>11} {'MiB med':>8} {'MiB p95':>8} "
      + " ".join(f"{k.split('(')[0].strip():>16}" for k in TOPES)
      + f" {'bloq':>7} {'2a luz':>7} {'guardia':>8}")
for nombre, pre, post, use_clock in CANDIDATOS:
    cuadros, mib, fuera = [], [], {k: 0 for k in TOPES}
    bloq = 0
    cubre_luz = 0
    cubre_guardia = 0
    for r in filas:
        fps, T = r["fps"], int(round(r["dur"] * r["fps"]))
        ft = int(round(r["tt"] * fps))
        f0 = max(0, int(ft - pre * fps))
        f1 = min(T, int(ft + post * fps) + 1)
        if use_clock and r["clk"] is not None and r["clk"] > 0.05:
            fa = int(round((r["clk"] + CLOCK_BIAS_S) * fps))
            if fa > f0:
                f0 = max(0, min(fa, int(ft - MIN_PRE_S * fps)))
        nf = max(1, f1 - f0)
        est = nf * r["w"] * r["h"] * 3 * DECODE_COPY_FACTOR
        cuadros.append(nf)
        mib.append(est / 1024 ** 2)
        for k, tope in TOPES.items():
            fuera[k] += int(est > tope)
        w0, w1 = f0 / fps, (f1 - 1) / fps
        c = any(w0 < x <= w1 for x in r["cuts"])
        rp = any(t0 <= w1 and t1 > w0 for t0, t1 in r["reps"])
        bloq += int(bool(r["otras"]) or c or rp)
        dt = r["dt_ms"]
        try:
            dt = None if dt is None else float(dt)
        except (TypeError, ValueError):
            dt = None
        if dt is None or not math.isfinite(dt):
            cubre_luz += 1                      # sin segunda luz declarada: no aplica, no penaliza
        else:
            cubre_luz += int(w1 >= r["tt"] + abs(dt) / 1000.0 - 1e-9)
        cubre_guardia += int(w0 <= r["tt"] - 4.0 + 1e-9)
    cuadros.sort()
    mib.sort()
    med = cuadros[len(cuadros) // 2]
    print(f"{nombre:28} {med:11d} {mib[len(mib)//2]:8.0f} {mib[int(.95*(len(mib)-1))]:8.0f} "
          + " ".join(f"{100.0*fuera[k]/n:15.2f}%" for k in TOPES)
          + f" {100.0*bloq/n:6.2f}% {100.0*cubre_luz/n:6.2f}% {100.0*cubre_guardia/n:7.2f}%")

print("\ncolumnas: 'bloq' = % de clips con qc_blocking no vacio (los que build_current_store descarta);")
print("'2a luz' = % de clips cuya ventana llega a t_touch + |dt_ms| (los que no declaran dt no penalizan);")
print("'guardia' = % de clips cuya ventana empieza en t_touch - 4.0 s o antes (la ventana [t-4.0, t-3.5]);")
print("las tres columnas de % de runtime son el % de clips que el planificador de decodificacion RECHAZA.")
