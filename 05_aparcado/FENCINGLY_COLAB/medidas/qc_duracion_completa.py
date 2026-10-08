"""Que le pasa al filtro de QC si la ventana pasa a ser el clip entero.

No re-extrae nada. Lee los artefactos de QC ya escritos (analysis/qc/*.json), que traen ya
medidos `cuts_t`, `replays`, `freezes`, `duration_s` y `t_touch_s`, y vuelve a evaluar las dos
banderas BLOQUEANTES que dependen de la ventana:

    cut_in_window    = any(w0 <  c <= w1 for c in cuts)
    replay_in_window = any(r.t0 <= w1 and r.t1 > w0 for r in replays)

con la ventana actual (w0, w1 tal como quedaron guardados) y con la ventana de duracion completa
(w0 = 0, w1 = duration_s - 1/fps, que es el tope que impone `limited_by_available_video`).

`colab_workflow.build_current_store` descarta toda fila con `qc_blocking` no vacio, asi que este
recuento es directamente cuantos clips entrarian o no al Store.
"""
import json
import sys
from pathlib import Path

QC = Path(sys.argv[1])
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

BLOQUEANTES_VENTANA = ("cut_in_window", "replay_in_window")

n = 0
sin_datos = 0
tot = {"cut_ahora": 0, "rep_ahora": 0, "cut_full": 0, "rep_full": 0,
       "bloq_ahora": 0, "bloq_full": 0, "bloq_otro_ahora": 0, "bloq_otro_full": 0,
       "n_cuts_pos": 0, "n_rep_pos": 0}
dur = []
w_len = []
files = sorted(QC.glob("*.json"))
if LIMIT:
    files = files[:LIMIT]
for p in files:
    try:
        q = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        sin_datos += 1
        continue
    if not q.get("qc_window_valid"):
        sin_datos += 1
        continue
    duration = q.get("duration_s")
    fps = q.get("video_fps") or q.get("fps") or 30.0
    if not duration or not fps:
        sin_datos += 1
        continue
    n += 1
    cuts = [float(c) for c in (q.get("cuts_t") or [])]
    reps = [(float(r[0]), float(r[1])) for r in (q.get("replays") or []) if isinstance(r, (list, tuple)) and len(r) >= 2]
    frz = [(float(a), float(b)) for a, b in (q.get("freezes") or [])]
    w0, w1 = float(q["qc_window_start_s"]), float(q["qc_window_end_s"])
    W0, W1 = 0.0, max(0.0, float(duration) - 1.0 / float(fps))
    dur.append(float(duration))
    w_len.append(w1 - w0)
    tot["n_cuts_pos"] += int(len(cuts) > 0)
    tot["n_rep_pos"] += int(len(reps) > 0)

    def cut_in(a, b):
        return any(a < c <= b for c in cuts)

    def rep_in(a, b):
        return any(t0 <= b and t1 > a for t0, t1 in reps)

    c_a, r_a = cut_in(w0, w1), rep_in(w0, w1)
    c_f, r_f = cut_in(W0, W1), rep_in(W0, W1)
    tot["cut_ahora"] += c_a
    tot["rep_ahora"] += r_a
    tot["cut_full"] += c_f
    tot["rep_full"] += r_f
    # bloqueo total: las banderas de ventana mas las bloqueantes que NO dependen de la ventana
    otras = [f for f in (q.get("blocking") or []) if f not in BLOQUEANTES_VENTANA]
    tot["bloq_otro_ahora"] += int(bool(otras))
    tot["bloq_otro_full"] += int(bool(otras))
    tot["bloq_ahora"] += int(bool(otras) or c_a or r_a)
    tot["bloq_full"] += int(bool(otras) or c_f or r_f)

dur.sort()
w_len.sort()


def pct(x):
    return f"{100.0 * x / n:.2f}%" if n else "n/a"


def q_(v, f):
    return v[int(f * (len(v) - 1))] if v else float("nan")


print(f"directorio                : {QC}")
print(f"artefactos QC leidos      : {len(files)}  (usables {n}, descartados {sin_datos})")
print(f"duracion del clip (s)     : min {dur[0]:.2f} | p05 {q_(dur,.05):.2f} | mediana {q_(dur,.5):.2f} | media {sum(dur)/len(dur):.2f} | p95 {q_(dur,.95):.2f} | max {dur[-1]:.2f}")
print(f"ventana guardada (s)      : min {w_len[0]:.2f} | mediana {q_(w_len,.5):.2f} | media {sum(w_len)/len(w_len):.2f} | max {w_len[-1]:.2f}")
print(f"factor duracion/ventana   : mediana {q_(dur,.5)/q_(w_len,.5):.2f}x | medio {(sum(dur)/len(dur))/(sum(w_len)/len(w_len)):.2f}x")
print()
print(f"clips con >=1 corte       : {tot['n_cuts_pos']} ({pct(tot['n_cuts_pos'])})")
print(f"clips con >=1 repeticion  : {tot['n_rep_pos']} ({pct(tot['n_rep_pos'])})")
print()
print("bandera BLOQUEANTE          ventana actual        duracion completa")
print(f"cut_in_window             : {tot['cut_ahora']:6d} ({pct(tot['cut_ahora']):>7})   {tot['cut_full']:6d} ({pct(tot['cut_full']):>7})")
print(f"replay_in_window          : {tot['rep_ahora']:6d} ({pct(tot['rep_ahora']):>7})   {tot['rep_full']:6d} ({pct(tot['rep_full']):>7})")
print(f"otras bloqueantes (igual) : {tot['bloq_otro_ahora']:6d} ({pct(tot['bloq_otro_ahora']):>7})   {tot['bloq_otro_full']:6d} ({pct(tot['bloq_otro_full']):>7})")
print(f"BLOQUEADO (cualquiera)    : {tot['bloq_ahora']:6d} ({pct(tot['bloq_ahora']):>7})   {tot['bloq_full']:6d} ({pct(tot['bloq_full']):>7})")
print()
print(f"clips que ENTRAN al Store : {n - tot['bloq_ahora']:6d} ahora   ->   {n - tot['bloq_full']:6d} con duracion completa"
      f"   (perdida {tot['bloq_full'] - tot['bloq_ahora']}, {pct(tot['bloq_full'] - tot['bloq_ahora'])})")
