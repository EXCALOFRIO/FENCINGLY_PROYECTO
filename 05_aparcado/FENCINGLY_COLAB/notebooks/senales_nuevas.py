"""Las TRES senales nuevas, medidas sobre los 24 NPZ de duracion completa.

1. TIEMPO DE DELIBERACION DEL ARBITRO: de las luces a que el reloj se reanuda.
2. VELOCIDAD ANGULAR DEL ANTEBRAZO: derivada de `forearm_ang`, que YA existe.
3. EJE DEL ANTEBRAZO CONTRA EL RIVAL: sustituto de "punta en linea".

QUE SE PUEDE MEDIR CON n = 24 Y QUE NO
--------------------------------------
SE PUEDE: cobertura con su n, y **redundancia contra las columnas que ya existen**, que es
lo que mato a la distancia entre munecas (r = 0,98). Es la prueba mas discriminante y la mas
barata, asi que es la que se hace.
NO SE PUEDE: AUC de prioridad con intervalo. 24 clips no dan potencia para eso, y el suelo de
ruido del proyecto es 0,8 pp. Cualquier AUC de aqui seria un numero sin intervalo utilizable,
o sea la clase de cifra que este proyecto ya se ha comido dos veces. Se declara y no se da.

SUAVIZADO, y su justificacion
-----------------------------
Savitzky-Golay de ventana 7 y orden 2 ANTES de derivar. La ventana no es un gusto: el telecine
25->30 afecta a **8.503 clips (75,8 % de la banda de 30)** con `dup_period` de **6,0 fotogramas
exactos**, o sea un duplicado cada seis. Un duplicado da velocidad cero en un paso y doble en
el siguiente. Una ventana de 7 (> 6) promedia sobre el periodo entero del artefacto; una de 5
no lo cubre. Se guardan las DOS versiones, cruda y suavizada.

NULO: el circular NO vale para derivadas — las infla hasta 7,79x porque `np.roll` deja una
costura que se convierte en una derivada enorme. Si se evalua, el nulo es el de signo de lado.
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
FIG = AQUI / "figuras_senales"
FIG.mkdir(exist_ok=True)
for p in (str(BASE / "scratchpad/detector/codigo_congelado"), str(AQUI),
          str(BASE / "scratchpad/reloj/trabajo")):
    if p not in sys.path:
        sys.path.insert(0, p)
from barra import Barra                                              # noqa: E402
from scipy.signal import savgol_filter                               # noqa: E402

CONF = 0.30
CODO, MUNECA = (7, 8), (9, 10)          # COCO: izq/der
HOMBROS, CADERAS = (5, 6), (11, 12)
VENTANA_SG, ORDEN_SG = 7, 2


def suaviza(x: np.ndarray) -> np.ndarray:
    """SG(7,2) por columna, respetando NaN por tramos contiguos."""
    y = np.array(x, np.float64, copy=True)
    if y.ndim == 1:
        y = y[:, None]
    out = np.full_like(y, np.nan)
    for j in range(y.shape[1]):
        col = y[:, j]
        ok = np.isfinite(col)
        if ok.sum() < VENTANA_SG:
            out[:, j] = col
            continue
        idx = np.flatnonzero(ok)
        cortes = np.flatnonzero(np.diff(idx) > 1)
        for a, b in zip(np.r_[0, cortes + 1], np.r_[cortes + 1, len(idx)]):
            tr = idx[a:b]
            if len(tr) >= VENTANA_SG:
                out[tr, j] = savgol_filter(col[tr], VENTANA_SG, ORDEN_SG)
            else:
                out[tr, j] = col[tr]
    return out[:, 0] if x.ndim == 1 else out


def deriva(x: np.ndarray, fps: float) -> np.ndarray:
    d = np.full_like(x, np.nan, dtype=np.float64)
    d[1:-1] = (x[2:] - x[:-2]) * fps / 2.0
    return d


def ang_norm(a: np.ndarray) -> np.ndarray:
    """A [-180, 180) para que la derivada no salte 360."""
    return (a + 180.0) % 360.0 - 180.0


def desenrolla(a: np.ndarray) -> np.ndarray:
    """`np.unwrap` POR TRAMOS CONTIGUOS de valores finitos.

    `np.unwrap` es acumulativo, asi que un solo NaN envenena todo lo que viene despues.
    Aplicado global bajaba la cobertura de la velocidad angular de 0,846 a 0,577 sin que
    faltara ningun dato: era el propio operador el que la borraba.
    """
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    cortes = np.flatnonzero(np.diff(idx) > 1)
    for x, y in zip(np.r_[0, cortes + 1], np.r_[cortes + 1, len(idx)]):
        tr = idx[x:y]
        out[tr] = np.degrees(np.unwrap(np.radians(a[tr])))
    return out


# ============================================================ senal 1: deliberacion
def deliberacion(res: pd.DataFrame) -> pd.DataFrame:
    """Tiempo de las luces a la reanudacion del reloj, leido del video completo."""
    import reloj as R
    import ancla_region as AR
    filas = []
    b = Barra(total=len(res), desc="s1 deliberacion", unit="clip")
    for _, r in res.iterrows():
        cid = str(r["clip_id"])
        npzp = WS / "npz" / f"{cid}.npz"
        if not npzp.is_file():
            b.update(1); continue
        with np.load(npzp, allow_pickle=False) as z:
            meta = json.loads(z["meta_json"].item())
        sede = r.get("tournament_id")
        fila = {"clip_id": cid, "sede": sede, "caso": r["_caso"],
                "t_touch_s": meta.get("t_touch_s"), "fps": meta["fps"],
                "dur_s": round(meta["n_frames_window"] / meta["fps"], 2),
                "perfil": AR.perfil_de(sede, None)}
        try:
            out = R.clock_region(str(r["ruta"]), sede=sede)
            diag = out.get("reloj_diag") or {}
            tics = diag.get("tics_s") or []
            fila["n_tics"] = diag.get("n_tics")
            fila["frac_corre"] = diag.get("frac_corre")
            tt = fila["t_touch_s"]
            # reanudacion = primer tic tras el tocado que INICIA un par regular de 1 Hz
            reanuda = None
            if tt is not None and len(tics) >= 2:
                post = [t for t in tics if t > tt + 0.30]
                for i in range(len(post) - 1):
                    if 0.7 < (post[i + 1] - post[i]) < 1.4:
                        reanuda = post[i]
                        break
            fila["t_reanuda_s"] = reanuda
            fila["deliberacion_s"] = round(reanuda - tt, 2) if (reanuda and tt) else None
            fila["cola_tras_toque_s"] = round(fila["dur_s"] - tt, 2) if tt else None
            fila["definido"] = reanuda is not None
        except Exception as exc:
            fila["error"] = repr(exc)
            fila["definido"] = False
        filas.append(fila)
        b.update(1)
    b.close()
    return pd.DataFrame(filas)


# ============================================================ senales 2 y 3
def geometria(res: pd.DataFrame):
    """Por fotograma y tirador: `forearm_ang`, su velocidad, y el angulo al torso rival."""
    filas = []
    b = Barra(total=len(res), desc="s2+s3 geometria", unit="clip")
    for _, r in res.iterrows():
        cid = str(r["clip_id"])
        p = WS / "npz" / f"{cid}.npz"
        if not p.is_file():
            b.update(1); continue
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta_json"].item())
            poses = z["poses"].astype(np.float64)      # (T,2,17,3) normalizado
        T = len(poses)
        fps = meta["fps"]
        ft = meta.get("f_touch")
        asp = (meta.get("size") or [16, 9])
        asp = float(asp[0]) / float(asp[1]) if asp and asp[1] else 16 / 9
        xy = poses[..., :2].copy()
        cf = poses[..., 2]
        xy[cf < CONF] = np.nan
        xy[..., 0] *= asp                              # a unidades isotropas
        for s in (0, 1):
            o = 1 - s
            # muneca armada = la de mayor confianza media (proxy; la dominante esta en
            # clip_stats.dominant_hand y aqui no se reimplementa)
            cm = np.nanmean(cf[:, s, list(MUNECA)], axis=0)
            k = int(np.nanargmax(cm)) if np.isfinite(cm).any() else 0
            codo, mun = CODO[k], MUNECA[k]
            v = xy[:, s, mun] - xy[:, s, codo]         # (T,2) eje codo->muneca
            ang = ang_norm(np.degrees(np.arctan2(-v[:, 1], v[:, 0])))
            desen = desenrolla(ang)
            ang_s = ang_norm(suaviza(desen))
            vel = deriva(desen, fps)
            vel_s = deriva(suaviza(desen), fps)
            # torso del rival = medio de hombros y caderas
            tor = np.nanmean(np.stack([xy[:, o, i] for i in HOMBROS + CADERAS], 1), axis=1)
            w = tor - xy[:, s, mun]                    # muneca -> torso rival
            nv = np.linalg.norm(v, axis=1); nw = np.linalg.norm(w, axis=1)
            cos = np.einsum("ij,ij->i", v, w) / (nv * nw)
            apunta = np.degrees(np.arccos(np.clip(cos, -1, 1)))    # 0 = apunta al rival
            apunta_s = suaviza(apunta)
            filas.append(pd.DataFrame({
                "clip_id": cid, "lado": "LR"[s], "frame_idx": np.arange(T),
                "dt_toque_s": (np.arange(T) - (ft if ft is not None else 0)) / fps,
                "fps": fps, "src_fps": meta.get("src_fps") or np.nan,
                "fa_ang": ang, "fa_ang_sg": ang_s,
                "fa_ang_vel": vel, "fa_ang_vel_sg": vel_s,
                "apunta_rival": apunta, "apunta_rival_sg": apunta_s,
                "apunta_vel_sg": deriva(apunta_s, fps),
                "dist_mun_torso": nw,
            }))
        b.update(1)
    b.close()
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


# ============================================================ redundancia
def redundancia(geo: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Correlacion de las nuevas columnas contra las de `features.pair_features`.

    Es la prueba que mato a la distancia entre munecas (r = 0,98 con columnas existentes).
    Umbral declarado: **|r| >= 0,95 = redundante**.
    """
    from fencingly_v2 import features as F
    fil = []
    b = Barra(total=len(res), desc="redundancia", unit="clip")
    for _, r in res.iterrows():
        cid = str(r["clip_id"])
        p = WS / "npz" / f"{cid}.npz"
        if not p.is_file():
            b.update(1); continue
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(z["meta_json"].item())
            poses = z["poses"].astype(np.float32)
            luces = z["lights"].astype(np.float32)
        asp = meta.get("size") or [16, 9]
        asp = float(asp[0]) / float(asp[1]) if asp and asp[1] else 16 / 9
        try:
            X, _valid, nombres = F.pair_features(poses, luces, meta["fps"], asp)
        except Exception as exc:
            print(f"  {cid}: pair_features fallo {exc!r}"); b.update(1); continue
        g = geo[geo.clip_id == cid]
        for lado, pre in (("L", "L_"), ("R", "R_")):
            gl = g[g.lado == lado]
            if len(gl) != len(X):
                continue
            for nueva in ("fa_ang_vel_sg", "apunta_rival_sg", "apunta_vel_sg", "fa_ang_sg"):
                y = gl[nueva].to_numpy(np.float64)
                for j, nm in enumerate(nombres):
                    if not nm.startswith(pre):
                        continue
                    x = np.asarray(X[:, j], np.float64)
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 20 or np.nanstd(x[ok]) < 1e-9 or np.nanstd(y[ok]) < 1e-9:
                        continue
                    fil.append({"clip_id": cid, "lado": lado, "nueva": nueva,
                                "existente": nm[2:], "r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                                "n": int(ok.sum())})
        b.update(1)
    b.close()
    return pd.DataFrame(fil)


if __name__ == "__main__":
    res = pd.read_csv(WS / "resultado.csv", low_memory=False)
    res = res[res.estado == "ok"]

    print("=" * 78); print("SENAL 1 - TIEMPO DE DELIBERACION DEL ARBITRO"); print("=" * 78)
    d1 = deliberacion(res)
    d1.to_csv(AQUI / "senal1_deliberacion.csv", index=False)
    print(d1[["clip_id", "sede", "perfil", "t_touch_s", "dur_s", "cola_tras_toque_s",
              "n_tics", "t_reanuda_s", "deliberacion_s", "definido"]].to_string(index=False))
    n = len(d1); k = int(d1["definido"].sum())
    print(f"\n  COBERTURA: {k} de {n} = {100*k/max(n,1):.1f} %")
    if k:
        v = d1.loc[d1["definido"], "deliberacion_s"]
        print(f"  deliberacion: mediana {v.median():.2f} s  rango {v.min():.2f} a {v.max():.2f} s")
    print(f"  cola tras el tocado: mediana {d1['cola_tras_toque_s'].median():.2f} s "
          f"(con POST_S=0,3 s no habria nada que medir)")
    print("  PRECISION: SIN REFERENCIA. No hay verdad de cuando el arbitro decidio.")
    print("  AUC contra 'el modelo falla': NO MEDIDO. n=24 no da potencia; hace falta la")
    print("  etiqueta y el corpus entero, y hay que medirlo DENTRO DE SEDE (la deliberacion")
    print("  larga puede correlacionar con la sede y la epoca, no con la ambiguedad).")

    print("\n" + "=" * 78); print("SENALES 2 y 3 - GEOMETRIA DEL ANTEBRAZO"); print("=" * 78)
    geo = geometria(res)
    geo.to_csv(AQUI / "senales23_geometria.csv.gz", index=False, compression="gzip")
    print(f"  filas {len(geo)}  ({geo.clip_id.nunique()} clips x 2 lados)")
    for c in ("fa_ang", "fa_ang_sg", "fa_ang_vel", "fa_ang_vel_sg",
              "apunta_rival", "apunta_rival_sg", "apunta_vel_sg"):
        cob = float(np.isfinite(geo[c]).mean())
        print(f"    {c:16s} cobertura {cob:.3f}"
              + ("   <- NaN = INDEFINIDO, no 0" if cob < 1 else ""))
    print("\n  efecto del suavizado en la velocidad angular (el telecine golpea aqui):")
    for nom, sub in (("banda 30 (telecine)", geo[(geo.fps > 29) & (geo.fps < 31)]),
                     ("banda 50 (sin duplicados)", geo[(geo.fps > 49) & (geo.fps < 51)])):
        if not len(sub):
            continue
        cr = np.nanpercentile(np.abs(sub["fa_ang_vel"].dropna()), 99)
        sv = np.nanpercentile(np.abs(sub["fa_ang_vel_sg"].dropna()), 99)
        print(f"    {nom:26s} |vel| p99 cruda {cr:8.1f} grados/s  suavizada {sv:8.1f}  "
              f"factor {cr/max(sv,1e-9):.2f}x")

    print("\n" + "=" * 78); print("REDUNDANCIA CONTRA LAS COLUMNAS QUE YA EXISTEN"); print("=" * 78)
    rd = redundancia(geo, res)
    if len(rd):
        rd.to_csv(AQUI / "senales23_redundancia.csv", index=False)
        ag = (rd.groupby(["nueva", "existente"])["r"]
                .agg(r_medio="mean", r_abs_max=lambda s: s.abs().max(), n="size")
                .reset_index())
        print("  umbral declarado: |r| >= 0,95 = REDUNDANTE\n")
        for nueva in sorted(ag.nueva.unique()):
            g = ag[ag.nueva == nueva].reindex(
                ag[ag.nueva == nueva]["r_medio"].abs().sort_values(ascending=False).index)
            print(f"  --- {nueva} ---")
            print(g.head(6)[["existente", "r_medio", "r_abs_max", "n"]].to_string(index=False))
            peor = g["r_medio"].abs().max()
            print(f"      max |r medio| = {peor:.3f}  -> "
                  f"{'REDUNDANTE' if peor >= 0.95 else 'NO redundante'}\n")
