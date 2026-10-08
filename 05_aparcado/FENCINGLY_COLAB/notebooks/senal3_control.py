"""Senal 3 (`apunta_rival`): prueba de sanidad, prueba de circularidad y figuras.

TRES COSAS, en este orden y con el criterio declarado antes de mirar:

A. SANIDAD DIRECCIONAL. Si el angulo entre el eje codo-muneca y el vector muneca->torso del
   rival mide "la punta amenaza", entonces **en el tocado tiene que ser MENOR que en la
   guardia**: al tocar el brazo apunta al rival. Criterio: la mediana de la diferencia
   (guardia - tocado) tiene que ser POSITIVA, y el signo consistente en la mayoria de clips.
   Si no separa esas dos ventanas, el sustituto no vale y se dice.

B. CIRCULARIDAD CONTRA EL ATAJO DEL ORDEN DE LUCES. "Quien apuntaba primero" esta muy cerca
   de "quien encendio primero", y ese atajo es el que tiene al proyecto en el techo. Asi que
   no basta comparar contra el azar: se compara contra `first_light`. Si el acuerdo es casi
   total, la senal esta midiendo la luz y no el reglamento.
   AVISO DE POTENCIA: n = 24 clips. Esto es una SONDA, no una medida. No se da AUC ni
   intervalo utilizable, y se dice.

C. FIGURAS con la geometria dibujada, que es donde se ve si esta implementada al reves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parents[1]
WS = AQUI / "ws_completo"
FIG = AQUI / "figuras_senales"
FIG.mkdir(exist_ok=True)
for p in (str(BASE / "scratchpad/detector/codigo_congelado"), str(AQUI)):
    if p not in sys.path:
        sys.path.insert(0, p)
from barra import Barra                                              # noqa: E402
from fencingly_v2 import video as V                                  # noqa: E402

GUARDIA = (-0.60, -0.40)     # ventana de guardia, en s respecto al tocado
TOQUE = (-0.06, 0.06)        # ventana del tocado
CONF, CODO, MUNECA = 0.30, (7, 8), (9, 10)
HOMBROS, CADERAS = (5, 6), (11, 12)

geo = pd.read_csv(AQUI / "senales23_geometria.csv.gz", low_memory=False)
res = pd.read_csv(WS / "resultado.csv", low_memory=False)
res = res[res.estado == "ok"]

# ---------------------------------------------------------------- A. sanidad
print("=" * 78)
print("A. SANIDAD DIRECCIONAL: el angulo debe ser MENOR en el tocado que en la guardia")
print("=" * 78)
fil = []
for (cid, lado), g in geo.groupby(["clip_id", "lado"]):
    gu = g[(g.dt_toque_s >= GUARDIA[0]) & (g.dt_toque_s <= GUARDIA[1])]["apunta_rival_sg"]
    to = g[(g.dt_toque_s >= TOQUE[0]) & (g.dt_toque_s <= TOQUE[1])]["apunta_rival_sg"]
    if gu.notna().sum() < 2 or to.notna().sum() < 1:
        continue
    fil.append({"clip_id": cid, "lado": lado,
                "guardia_deg": round(float(gu.median()), 1),
                "toque_deg": round(float(to.median()), 1),
                "dif_deg": round(float(gu.median() - to.median()), 1)})
sa = pd.DataFrame(fil)
print(sa.to_string(index=False))
if len(sa):
    pos = int((sa["dif_deg"] > 0).sum())
    print(f"\n  n = {len(sa)} (clip x lado)")
    print(f"  mediana de (guardia - tocado) : {sa['dif_deg'].median():+.1f} grados")
    print(f"  con el signo esperado (>0)    : {pos} de {len(sa)} = {100*pos/len(sa):.1f} %")
    print(f"  guardia mediana {sa['guardia_deg'].median():.1f} deg  "
          f"tocado mediana {sa['toque_deg'].median():.1f} deg")
    ok = sa["dif_deg"].median() > 0 and pos > len(sa) / 2
    print(f"  VEREDICTO: {'PASA' if ok else 'NO PASA'} la sanidad direccional")
    sa.to_csv(AQUI / "senal3_sanidad.csv", index=False)

# ---------------------------------------------------------------- B. circularidad
print("\n" + "=" * 78)
print("B. CIRCULARIDAD: 'quien apunta antes' contra 'quien encendio antes'")
print("=" * 78)
idx = pd.read_csv(WS / "npz" / "index.csv", low_memory=False)
fl = dict(zip(idx["clip_id"].astype(str), idx["first_light"]))
nl = dict(zip(idx["clip_id"].astype(str), pd.to_numeric(idx["n_valid_lights"], errors="coerce")))
fil = []
UMBRAL = 30.0        # "apunta" = angulo por debajo de 30 grados
for cid, g in geo.groupby("clip_id"):
    prim = {}
    for lado in ("L", "R"):
        gl = g[(g.lado == lado) & (g.dt_toque_s >= -2.0) & (g.dt_toque_s <= 0.0)]
        bajo = gl[gl["apunta_rival_sg"] < UMBRAL]
        prim[lado] = float(bajo["dt_toque_s"].min()) if len(bajo) else np.nan
    quien = ("left" if prim["L"] < prim["R"] else "right") \
        if np.isfinite(prim["L"]) and np.isfinite(prim["R"]) and prim["L"] != prim["R"] else None
    fil.append({"clip_id": cid, "t_apunta_L": prim["L"], "t_apunta_R": prim["R"],
                "apunta_primero": quien, "first_light": fl.get(cid),
                "n_valid_lights": nl.get(cid)})
ci = pd.DataFrame(fil)
ci.to_csv(AQUI / "senal3_circularidad.csv", index=False)
print(ci.to_string(index=False))
ev = ci[ci.apunta_primero.notna() & ci.first_light.isin(["left", "right"])]
print(f"\n  clips evaluables (los dos definidos): {len(ev)} de {len(ci)}")
if len(ev):
    ac = int((ev.apunta_primero == ev.first_light).sum())
    print(f"  acuerdo con `first_light`: {ac} de {len(ev)} = {100*ac/len(ev):.1f} %")
    print("  LECTURA: si el acuerdo fuera casi total, la senal estaria midiendo la LUZ y no")
    print("  el reglamento, y seria el mismo atajo que ya tiene al proyecto en el techo.")
    print(f"  POTENCIA: n = {len(ev)}. Es una SONDA. No se da AUC ni intervalo: con esta n el")
    print("  intervalo de Wilson cubriria medio eje y no decidiria nada.")

# ---------------------------------------------------------------- C. figuras
print("\n" + "=" * 78)
print("C. FIGURAS con la geometria dibujada")
print("=" * 78)
sel = res.head(4)
b = Barra(total=len(sel), desc="figuras s3", unit="clip")
hechas = 0
for _, r in sel.iterrows():
    cid = str(r["clip_id"])
    p = WS / "npz" / f"{cid}.npz"
    if not p.is_file():
        b.update(1); continue
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(z["meta_json"].item())
        poses = z["poses"].astype(np.float64)
    T = len(poses); fps = meta["fps"]; ft = meta.get("f_touch") or T // 2
    clip = V.Clip(str(r["ruta"]))
    ks = [max(0, int(ft + d * fps)) for d in (-0.8, -0.5, -0.2, 0.0)]
    ks = [k for k in ks if k < T]
    _i, frames = clip.read_all(min(ks), max(ks) + 1)
    base = min(ks)
    celdas = []
    for k in ks:
        if k - base >= len(frames):
            continue
        fr = frames[k - base].copy()
        H, W = fr.shape[:2]
        for s, col in ((0, (60, 220, 60)), (1, (60, 160, 255))):
            o = 1 - s
            cf = poses[k, s, :, 2]
            cm = np.nanmean(poses[:, s, list(MUNECA), 2], axis=0)
            kk = int(np.nanargmax(cm)) if np.isfinite(cm).any() else 0
            codo, mun = CODO[kk], MUNECA[kk]
            if cf[codo] < CONF or cf[mun] < CONF:
                continue
            pc = (poses[k, s, codo, :2] * [W, H]).astype(int)
            pm = (poses[k, s, mun, :2] * [W, H]).astype(int)
            ids = [i for i in HOMBROS + CADERAS if poses[k, o, i, 2] >= CONF]
            if not ids:
                continue
            pt = (np.nanmean([poses[k, o, i, :2] for i in ids], 0) * [W, H]).astype(int)
            # eje del antebrazo, prolongado
            d = pm - pc
            n = np.linalg.norm(d)
            if n > 1:
                fin = (pm + d / n * 0.35 * W).astype(int)
                cv2.line(fr, tuple(pm), tuple(fin), col, 2, cv2.LINE_AA)
            cv2.line(fr, tuple(pc), tuple(pm), col, 4, cv2.LINE_AA)      # antebrazo
            cv2.line(fr, tuple(pm), tuple(pt), (240, 240, 240), 2, cv2.LINE_AA)  # al rival
            cv2.circle(fr, tuple(pt), 7, (240, 240, 240), 2, cv2.LINE_AA)
            gl = geo[(geo.clip_id == cid) & (geo.lado == "LR"[s])]
            fila = gl[gl.frame_idx == k]
            a = float(fila["apunta_rival_sg"].iloc[0]) if len(fila) else float("nan")
            cv2.putText(fr, f"{'LR'[s]} apunta={a:.0f}deg", (10, 26 + 26 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        cv2.putText(fr, f"f={k}  dt={(k-ft)/fps:+.2f}s", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        celdas.append(cv2.resize(fr, (600, int(600 * H / W)), interpolation=cv2.INTER_AREA))
    del frames
    if celdas:
        hh = max(c.shape[0] for c in celdas)
        celdas = [cv2.copyMakeBorder(c, 0, hh - c.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0)) for c in celdas]
        rej = np.hstack(celdas[:2]) if len(celdas) < 4 else np.vstack(
            [np.hstack(celdas[:2]), np.hstack(celdas[2:4])])
        cab = np.zeros((34, rej.shape[1], 3), np.uint8)
        cv2.putText(cab, f"{cid}  eje antebrazo (color) vs vector al torso rival (blanco)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(FIG / f"s3__{cid}.jpg"), np.vstack([cab, rej]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        hechas += 1
    b.update(1)
b.close()
print(f"  {hechas} figuras en {FIG}")
