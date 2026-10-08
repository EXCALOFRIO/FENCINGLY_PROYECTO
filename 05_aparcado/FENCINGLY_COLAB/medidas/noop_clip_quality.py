"""Prueba de NO-OP del arreglo de ambito de `cut_in_window` / `replay_in_window`.

    python -B noop_clip_quality.py <dir_analysis_qc> [...]

El arreglo acota las DOS banderas BLOQUEANTES a la ventana de ANALISIS en vez de a la ventana
GUARDADA:

    a0 = max(w0, t_touch - ANALYSIS_PRE_S)      ANALYSIS_PRE_S  = 4.0
    a1 = min(w1, t_touch + ANALYSIS_POST_S)     ANALYSIS_POST_S = 0.3

    cut_in_window    = any(a0 <  c <= a1 for c in cuts)
    replay_in_window = any(r.t0 <= a1 and r.t1 > a0 for r in replays)

Sobre un corpus extraido con la ventana desplegada (pre_s = 4.0, post_s = 0.3) se cumple
`a0 == w0` y `a1 == w1` salvo redondeo, asi que las dos banderas tienen que salir IDENTICAS. Este
script lo cuenta clip a clip y solo aprueba con CERO discrepancias.

Que se cambia y que NO: solo las dos bloqueantes. `n_cuts_before/after`,
`replay_before_touch`, `replay_after_touch`, `freeze_in_window` y el resto siguen describiendo el
clip COMPLETO, que es lo que se quiere para el informe de calidad.
"""
import json
import sys
from pathlib import Path

ANALYSIS_PRE_S, ANALYSIS_POST_S = 4.0, 0.3


def evalua(cuts, reps, a, b):
    corte = any(a < c <= b for c in cuts)
    repet = any(t0 <= b and t1 > a for t0, t1 in reps)
    return corte, repet


def revisa(qc_dir):
    qc_dir = Path(qc_dir)
    n = usables = 0
    dif_corte = dif_rep = 0
    ventana_identica = 0
    max_da0 = max_da1 = 0.0
    ejemplos = []
    for p in qc_dir.glob("*.json"):
        n += 1
        try:
            q = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not q.get("qc_window_valid"):
            continue
        tt = q.get("t_touch_s")
        if tt is None:
            continue
        w0, w1 = q.get("qc_window_start_s"), q.get("qc_window_end_s")
        if w0 is None or w1 is None:
            continue
        usables += 1
        tt, w0, w1 = float(tt), float(w0), float(w1)
        cuts = [float(c) for c in (q.get("cuts_t") or [])]
        reps = [(float(r[0]), float(r[1])) for r in (q.get("replays") or [])
                if isinstance(r, (list, tuple)) and len(r) >= 2]
        a0 = max(w0, tt - ANALYSIS_PRE_S)
        a1 = min(w1, tt + ANALYSIS_POST_S)
        max_da0 = max(max_da0, abs(a0 - w0))
        max_da1 = max(max_da1, abs(a1 - w1))
        if abs(a0 - w0) < 1e-9 and abs(a1 - w1) < 1e-9:
            ventana_identica += 1
        antes = evalua(cuts, reps, w0, w1)
        despues = evalua(cuts, reps, a0, a1)
        if antes[0] != despues[0]:
            dif_corte += 1
            if len(ejemplos) < 5:
                ejemplos.append((p.stem, "cut", antes[0], despues[0], round(w0, 3), round(a0, 3),
                                 round(w1, 3), round(a1, 3), [round(c, 2) for c in cuts]))
        if antes[1] != despues[1]:
            dif_rep += 1
            if len(ejemplos) < 5:
                ejemplos.append((p.stem, "replay", antes[1], despues[1], round(w0, 3), round(a0, 3),
                                 round(w1, 3), round(a1, 3), reps))
    return {"dir": str(qc_dir), "artefactos": n, "usables": usables,
            "ventana_identica": ventana_identica, "dif_cut_in_window": dif_corte,
            "dif_replay_in_window": dif_rep, "max_delta_a0": max_da0, "max_delta_a1": max_da1,
            "ejemplos": ejemplos}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    total_dif = 0
    for d in sys.argv[1:]:
        r = revisa(d)
        print("=" * 78)
        print(r["dir"])
        print(f"  artefactos QC              : {r['artefactos']}  (usables {r['usables']})")
        print(f"  ventana analisis == guardada: {r['ventana_identica']} de {r['usables']}"
              + (f"  ({100*r['ventana_identica']/r['usables']:.2f} %)" if r["usables"] else ""))
        print(f"  desviacion maxima a0 / a1  : {r['max_delta_a0']:.6f} s / {r['max_delta_a1']:.6f} s")
        print(f"  DISCREPANCIAS cut_in_window   : {r['dif_cut_in_window']}")
        print(f"  DISCREPANCIAS replay_in_window: {r['dif_replay_in_window']}")
        for e in r["ejemplos"]:
            print("   ejemplo:", e)
        total_dif += r["dif_cut_in_window"] + r["dif_replay_in_window"]
    print("=" * 78)
    if total_dif == 0:
        print("NO-OP DEMOSTRADO: cero discrepancias en las dos banderas bloqueantes.")
    else:
        print(f"NO ES NO-OP: {total_dif} discrepancias. NO aplicar el arreglo.")
        raise SystemExit(1)
