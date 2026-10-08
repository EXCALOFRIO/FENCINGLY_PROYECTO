"""Senales 2 y 3 CON LA MUNECA ARMADA CORRECTA, y remedida de la redundancia.

QUE CAMBIA respecto de `senales_nuevas.py`
------------------------------------------
Antes elegia la muneca armada por **confianza media mas alta**, que es un proxy y era la parte
mas debil de la senal: `fa_ang_sg` correlacionaba solo 0,468 con `forearm_ang`, y eso no era
"otra medida" sino **medir el brazo equivocado en algunos clips**.

Ahora se usa la API que ya existe:

    clip_stats.dominant_hand(P, side, end, fps) -> dict

con `P` de forma (T, 17, 3) en x de pista, y devuelve `armed_wrist` (9 o 10), `hand`
('diestro'/'zurdo') y `hand_conf` en [0,5, 1].

TRES AVISOS, aplicados
----------------------
1. **La confianza viaja con la columna**, y cuando `dominant_hand` no resuelve (`w == 0`,
   devuelve `armed_wrist=None`) la fila sale **INDEFINIDA (NaN), no con una eleccion por
   defecto**. Un empate resuelto a la fuerza es peor que un NaN.
2. **`end` es el indice del TOCADO, no el final del array.** `dominant_hand` mira
   `[end - 0,6 s, end + 2]`, asi que con clips de duracion completa pasar `len(P)` mediria la
   cola del clip en vez de la accion. Se pasa `touch_frame_raw` (= f_touch - f0, y con f0 = 0
   es f_touch). Es la misma clase de fallo que `f0 = 0` con `allez_scale`.
3. **`dominant_hand` ya aplica `FACING[side]` dentro.** No se aplica dos veces.

Nota medida: `pan` es irrelevante para esta funcion. Sus dos indicios son diferencias DENTRO
del mismo fotograma (`x_R - x_L`, y normas `muneca - hombro`), y un desplazamiento por
fotograma se cancela en las dos. Se pasa la x de pista igualmente, por fidelidad al contrato.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parents[1]
WS = AQUI / "ws_completo"
for p in (str(BASE / "scratchpad/detector/codigo_congelado"), str(AQUI)):
    if p not in sys.path:
        sys.path.insert(0, p)
from barra import Barra                                              # noqa: E402
from senales_nuevas import suaviza, deriva, ang_norm, desenrolla      # noqa: E402
from fencingly_v2 import clip_stats as CS, features as F             # noqa: E402

CONF = 0.30
HOMBROS, CADERAS = (5, 6), (11, 12)
CODO_DE = {9: 7, 10: 8}          # COCO: muneca 9 -> codo 7 (izq); 10 -> 8 (der)
GUARDIA, TOQUE = (-0.60, -0.40), (-0.06, 0.06)

res = pd.read_csv(WS / "resultado.csv", low_memory=False)
res = res[res.estado == "ok"]

filas, manos = [], []
b = Barra(total=len(res), desc="muneca+geometria", unit="clip")
for _, r in res.iterrows():
    cid = str(r["clip_id"])
    p = WS / "npz" / f"{cid}.npz"
    if not p.is_file():
        b.update(1); continue
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(z["meta_json"].item())
        poses = z["poses"].astype(np.float64)
        pan = z["pan"].astype(np.float64)
    T, fps = len(poses), meta["fps"]
    # `end` = indice del TOCADO dentro del array. Con f0 = 0 es f_touch.
    end = meta.get("touch_frame_raw")
    if end is None:
        end = (meta.get("f_touch") or T // 2) - (meta.get("f0") or 0)
    end = int(np.clip(end, 0, T - 1))
    asp = meta.get("size") or [16, 9]
    asp = float(asp[0]) / float(asp[1]) if asp and asp[1] else 16 / 9

    for s, lado in ((0, "L"), (1, "R")):
        o = 1 - s
        # --- x de PISTA para dominant_hand, sin corregir aspecto (su contrato) ---
        P = poses[:, s].copy()
        P[..., 0] = P[..., 0] + pan[:, None]
        dh = CS.dominant_hand(P, lado, end, fps)
        manos.append({"clip_id": cid, "lado": lado, "end": end, "T": T, **dh})
        mun = dh.get("armed_wrist")
        # --- geometria en unidades isotropas ---
        xy = poses[:, :, :, :2].copy()
        cf = poses[:, :, :, 2]
        xy[cf < CONF] = np.nan
        xy[..., 0] *= asp
        if mun is None:
            # INDEFINIDO: no se elige muneca por defecto
            filas.append(pd.DataFrame({
                "clip_id": cid, "lado": lado, "frame_idx": np.arange(T),
                "dt_toque_s": (np.arange(T) - end) / fps, "fps": fps,
                "muneca_armada": np.nan, "mano": None, "mano_conf": np.nan,
                "fa_ang": np.nan, "fa_ang_sg": np.nan, "fa_ang_vel": np.nan,
                "fa_ang_vel_sg": np.nan, "apunta_rival": np.nan,
                "apunta_rival_sg": np.nan, "apunta_vel_sg": np.nan}))
            continue
        codo = CODO_DE[mun]
        v = xy[:, s, mun] - xy[:, s, codo]
        ang = ang_norm(np.degrees(np.arctan2(-v[:, 1], v[:, 0])))
        desen = desenrolla(ang)
        ang_s = ang_norm(suaviza(desen))
        tor = np.nanmean(np.stack([xy[:, o, i] for i in HOMBROS + CADERAS], 1), axis=1)
        w = tor - xy[:, s, mun]
        nv = np.linalg.norm(v, axis=1); nw = np.linalg.norm(w, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.einsum("ij,ij->i", v, w) / (nv * nw)
        apunta = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        apunta_s = suaviza(apunta)
        filas.append(pd.DataFrame({
            "clip_id": cid, "lado": lado, "frame_idx": np.arange(T),
            "dt_toque_s": (np.arange(T) - end) / fps, "fps": fps,
            "muneca_armada": mun, "mano": dh.get("hand"), "mano_conf": dh.get("hand_conf"),
            "fa_ang": ang, "fa_ang_sg": ang_s,
            "fa_ang_vel": deriva(desen, fps), "fa_ang_vel_sg": deriva(suaviza(desen), fps),
            "apunta_rival": apunta, "apunta_rival_sg": apunta_s,
            "apunta_vel_sg": deriva(apunta_s, fps),
            "largo_antebrazo": nv, "dist_mun_torso": nw}))
    b.update(1)
b.close()

geo = pd.concat(filas, ignore_index=True)
mn = pd.DataFrame(manos)
geo.to_csv(AQUI / "senales_v2_geometria.csv.gz", index=False, compression="gzip")
mn.to_csv(AQUI / "senales_v2_manos.csv", index=False)

print("=" * 78); print("A. LA MUNECA ARMADA, por dominant_hand"); print("=" * 78)
print(mn[["clip_id", "lado", "end", "T", "hand", "armed_wrist", "hand_conf",
          "hand_n_frames"]].to_string(index=False))
n = len(mn); res_ok = int(mn["armed_wrist"].notna().sum())
print(f"\n  resueltos: {res_ok} de {n} = {100*res_ok/n:.1f} %")
print(f"  INDEFINIDOS (armed_wrist None -> NaN, no eleccion por defecto): {n - res_ok}")
if res_ok:
    print(f"  confianza: mediana {mn['hand_conf'].median():.3f}  "
          f"min {mn['hand_conf'].min():.3f}  "
          f"por debajo de 0,60: {int((mn['hand_conf'] < 0.60).sum())}")
    print(f"  mano: {dict(mn['hand'].value_counts())}")
# cuanto cambia respecto del proxy anterior
try:
    v1 = pd.read_csv(AQUI / "senales23_geometria.csv.gz", low_memory=False)
    a1 = v1.groupby(["clip_id", "lado"])["fa_ang"].median().rename("v1")
    a2 = geo.groupby(["clip_id", "lado"])["fa_ang"].median().rename("v2")
    cmp = pd.concat([a1, a2], axis=1).dropna()
    dif = (cmp.v1 - cmp.v2).abs()
    print(f"\n  cambio del angulo mediano frente al proxy de confianza (n={len(cmp)}):")
    print(f"    mediana {dif.median():.1f} grados   > 5 grados en {int((dif > 5).sum())} de {len(cmp)}")
    print("    (> 5 grados = el proxy elegia OTRA muneca en ese clip-lado)")
except Exception as exc:
    print(f"  comparacion con la v1 no disponible: {exc!r}")

print("\n" + "=" * 78); print("B. COBERTURA"); print("=" * 78)
for c in ("fa_ang", "fa_ang_sg", "fa_ang_vel", "fa_ang_vel_sg",
          "apunta_rival", "apunta_rival_sg", "apunta_vel_sg"):
    print(f"    {c:16s} {float(np.isfinite(geo[c]).mean()):.3f}")

print("\n" + "=" * 78); print("C. REDUNDANCIA, con media de |r|"); print("=" * 78)
fil = []
b = Barra(total=len(res), desc="redundancia", unit="clip")
for _, r in res.iterrows():
    cid = str(r["clip_id"])
    p = WS / "npz" / f"{cid}.npz"
    if not p.is_file():
        b.update(1); continue
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(z["meta_json"].item())
        poses = z["poses"].astype(np.float32); luces = z["lights"].astype(np.float32)
    asp = meta.get("size") or [16, 9]
    asp = float(asp[0]) / float(asp[1]) if asp and asp[1] else 16 / 9
    try:
        X, _v, nombres = F.pair_features(poses, luces, meta["fps"], asp)
    except Exception:
        b.update(1); continue
    g = geo[geo.clip_id == cid]
    for lado, pre in (("L", "L_"), ("R", "R_")):
        gl = g[g.lado == lado]
        if len(gl) != len(X):
            continue
        for nueva in ("fa_ang_sg", "fa_ang_vel_sg", "apunta_rival_sg", "apunta_vel_sg"):
            y = gl[nueva].to_numpy(np.float64)
            for j, nm in enumerate(nombres):
                if not nm.startswith(pre):
                    continue
                x = np.asarray(X[:, j], np.float64)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 20 or np.nanstd(x[ok]) < 1e-9 or np.nanstd(y[ok]) < 1e-9:
                    continue
                fil.append({"clip_id": cid, "lado": lado, "nueva": nueva,
                            "existente": nm[2:],
                            "abs_r": abs(float(np.corrcoef(x[ok], y[ok])[0, 1])),
                            "n": int(ok.sum())})
    b.update(1)
b.close()
rd = pd.DataFrame(fil)
rd.to_csv(AQUI / "senales_v2_redundancia.csv", index=False)
print("  umbral declarado: media|r| >= 0,95 = REDUNDANTE\n")
for nueva in sorted(rd.nueva.unique()):
    g = (rd[rd.nueva == nueva].groupby("existente")["abs_r"]
         .agg(media="mean", mediana="median", n="size").sort_values("media", ascending=False))
    print(f"  --- {nueva} ---")
    print(g.head(4).round(4).to_string())
    top = float(g.iloc[0]["media"])
    print(f"      max media|r| = {top:.4f} con {g.index[0]}  -> "
          f"{'REDUNDANTE' if top >= 0.95 else 'NO redundante'}")
    if "forearm_ang" in g.index:
        print(f"      contra forearm_ang: media|r| = {float(g.loc['forearm_ang','media']):.4f} "
              "(antes 0,4678 con el proxy de confianza)")
    print()

print("=" * 78); print("D. SANIDAD DE LA SENAL 3, rehecha con la muneca correcta"); print("=" * 78)
fil = []
for (cid, lado), g in geo.groupby(["clip_id", "lado"]):
    gu = g[(g.dt_toque_s >= GUARDIA[0]) & (g.dt_toque_s <= GUARDIA[1])]["apunta_rival_sg"]
    to = g[(g.dt_toque_s >= TOQUE[0]) & (g.dt_toque_s <= TOQUE[1])]["apunta_rival_sg"]
    if gu.notna().sum() < 2 or to.notna().sum() < 1:
        continue
    fil.append({"clip_id": cid, "lado": lado, "guardia": round(float(gu.median()), 1),
                "toque": round(float(to.median()), 1),
                "dif": round(float(gu.median() - to.median()), 1)})
sa = pd.DataFrame(fil)
if len(sa):
    pos = int((sa["dif"] > 0).sum())
    print(f"  n = {len(sa)} (clip x lado)")
    print(f"  mediana de (guardia - tocado): {sa['dif'].median():+.1f} grados  "
          f"(con el proxy: -15,0)")
    print(f"  con el signo esperado (>0)   : {pos} de {len(sa)} = {100*pos/len(sa):.1f} %  "
          "(con el proxy: 33,3 %)")
    print(f"  guardia mediana {sa['guardia'].median():.1f}  tocado mediana {sa['toque'].median():.1f}")
    ok = sa["dif"].median() > 0 and pos > len(sa) / 2
    print(f"  VEREDICTO: {'PASA' if ok else 'NO PASA'}")
    print("\n  Y sigue en pie lo declarado: el control se evalua en LOS DOS tiradores.")
    print("  Restringirlo al que puntua condicionaria por el RESULTADO (AGENTS.md 36) y")
    print("  no se hace, pase o no pase.")
    sa.to_csv(AQUI / "senales_v2_sanidad.csv", index=False)
