"""La sonda de ZONA sobre V-JEPA 2 (solo `media`): ¿sabe el resumen de video DONDE en la pista
ocurre el tocado?

El protocolo, el criterio y sus cambios estan en
`04_documentacion/informes/VJEPA_ZONA_2026-10-09.md`, escritos ANTES de entrenar nada y ANTES de
mirar ningun acierto en PRUEBA. Aqui no se decide nada que no este alli.

QUE SE MIDE: una sonda LINEAL sobre `media` (2,1024; vista real + su espejo horizontal), impar por
CONSTRUCCION en la zona (5 clases ordenadas `CA.ZONAS`: el espejo intercambia fotl<->fotr y deja
`middle`), igual de espiritu que `sonda_lineal`/`ajusta_fusion` de
`02_validacion/herramienta/sonda_vjepa.py` (REUTILIZADAS aqui, copiadas, no importadas, para no
depender de ese fichero mientras corre la otra medida), en tres usos:

  (1) RESPALDO   en los PRUEBA que la produccion NO contesta: sonda contra `middle`.
  (2) FUSION     en los PRUEBA que SI contestan: [onehot(zona_final), logits sonda] contra `zona_final` sola.
  (3) INFORMATIVO la sonda sola en TODO PRUEBA, contra `middle`.

RECURSOS: solo CPU, 4 hilos (`OMP_NUM_THREADS`, `torch.set_num_threads`), SOLO la clave `media` de
cada npz (nunca `tok`: serian 11,6 GB para los 5.524 clips). RAM esperada: decenas de MB.

    python sonda_vjepa_zona.py [--rapido]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

SEMILLA = 20261009
B_BOOT = 2000
VJEPA_DIR = pathlib.Path(os.environ.get("FENCINGLY_VJEPA_DIR", "D:/FENCINGLY_SCRATCH/vjepa"))
AJUSTE_TXT = RU.VAL / "auditoria" / "zona_cobertura_2026-10-09" / "torneos_ajuste.txt"
SALIDA = RU.VAL / "auditoria" / "vjepa_zona_2026-10-09"

GRID_L2 = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
GRID_PESOS = (False, True)
GRID_C_FUSION = (0.01, 0.1, 1.0, 10.0, 100.0)

IDX = {z: i for i, z in enumerate(CA.ZONAS)}


# ---------------------------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------------------------

def construye_dataset():
    """Clips con `strip_location` en `CA.ZONAS`, `zona_<clip>.csv` y embedding V-JEPA. QUE
    DEVUELVE: lista de dicts por clip (clip, torneo, weapon, y, zona, zona_final, via_zona,
    motivo, npz)."""
    df = CA.manifiesto()
    df = df[df["strip_location"].isin(CA.ZONAS)]
    filas = []
    for _, m in df.iterrows():
        clip = m["clip_id"]
        zp = RU.TRABAJO / ("zona_%s.csv" % clip)
        vp = VJEPA_DIR / ("vjepa_%s.npz" % clip)
        if not zp.is_file() or not vp.is_file():
            continue
        with open(zp, encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None) or {}
        filas.append({
            "clip": clip, "torneo": m["torneo"], "weapon": m["weapon"],
            "y": m["strip_location"],
            "zona": r.get("zona") or None,
            "zona_final": r.get("zona_final") or None,
            "via_zona": r.get("via_zona") or None,
            "motivo": r.get("motivo"),
            "npz": vp,
        })
    return filas


def carga_media(filas):
    """QUE DEVUELVE: M (n,2,1024) fp32, CJ (n,4) fp32 (caja en pixeles), via_ancla (n,). SOLO lee
    `media`, `caja`, `via_ancla`: NUNCA `tok` (11,6 GB)."""
    n = len(filas)
    M = np.empty((n, 2, 1024), dtype="float32")
    CJ = np.empty((n, 4), dtype="float32")
    via = []
    for i, f in enumerate(filas):
        with np.load(f["npz"]) as z:
            M[i] = z["media"]
            CJ[i] = z["caja"]
            via.append(str(z["via_ancla"]))
    return M, CJ, np.array(via)


def ancho_marco(clips):
    """QUE DEVUELVE: dict clip -> W (ancho del video en pixeles), de `poses_<clip>.npz` (NUNCA del
    video ni de `tok`); None si falta el fichero o la clave."""
    out = {}
    for c in clips:
        p = RU.TRABAJO / ("poses_%s.npz" % c)
        if not p.is_file():
            out[c] = None
            continue
        try:
            with np.load(p) as z:
                out[c] = int(z["W"]) if "W" in z.files else None
        except Exception:
            out[c] = None
    return out


def rasgo_caja(caja, w):
    """Posicion horizontal del centro del recorte, normalizada e IMPAR al espejar: 0 en el centro
    del fotograma, +-1 en los bordes. `None` -> 0.0 (neutro, no aporta ni resta)."""
    if w is None or w <= 0:
        return 0.0
    cx = 0.5 * (float(caja[0]) + float(caja[2]))
    return float((cx / w - 0.5) * 2.0)


# ---------------------------------------------------------------------------------------------
# LA SONDA, impar por construccion
# ---------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def sonda_zona(con_caja=False):
    torch = _torch()
    nn = torch.nn

    class Sonda(nn.Module):
        """5 logits de zona a partir de `u` (posicion lateral IMPAR) con centros y sesgos
        simetricos: `logit_k(espejo) = logit_{4-k}(original)` para CUALQUIER valor de los
        parametros (estructural, no aprendido)."""

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1024, dtype=torch.float64))
            self.wc = nn.Parameter(torch.zeros((), dtype=torch.float64)) if con_caja else None
            self.p1 = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.p2 = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.p_tau = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.b_m = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.b_h = nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.b_w = nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, m0, m1, cx=None):
            """m0, m1: (n,1024) float64, las medias de las vistas real/espejo. `cx`: (n,) float64
            o None, el rasgo de caja de la vista 0 (para la vista 1 se usa su opuesto, como
            correspondre a espejar TODO el fotograma)."""
            a = m0 @ self.w
            b = m1 @ self.w
            if self.wc is not None and cx is not None:
                a = a + self.wc * cx
                b = b - self.wc * cx
            u = 0.5 * (a - b)
            c1 = nn.functional.softplus(self.p1)
            c2 = c1 + nn.functional.softplus(self.p2)
            tau = nn.functional.softplus(self.p_tau) + 1e-3
            centros = torch.stack([-c2, -c1, torch.zeros_like(c1), c1, c2])
            sesgos = torch.stack([self.b_w, self.b_h, self.b_m, self.b_h, self.b_w])
            d2 = (u[:, None] - centros[None, :]) ** 2
            return -d2 / (2 * tau * tau) + sesgos[None, :]

    return Sonda()


def entrena_sonda_zona(M, idx, y, l2, pesos, con_caja, cx, semilla=SEMILLA):
    torch = _torch()
    torch.manual_seed(semilla)
    mod = sonda_zona(con_caja=con_caja)
    m0 = torch.as_tensor(M[idx, 0], dtype=torch.float64)
    m1 = torch.as_tensor(M[idx, 1], dtype=torch.float64)
    cxb = torch.as_tensor(cx[idx], dtype=torch.float64) if con_caja else None
    yt = torch.as_tensor(y[idx], dtype=torch.long)
    w_clase = None
    if pesos:
        cuentas = torch.bincount(yt, minlength=5).double().clamp(min=1.0)
        w_clase = cuentas.sum() / (5.0 * cuentas)
    opt = torch.optim.LBFGS(mod.parameters(), max_iter=300, line_search_fn="strong_wolfe")

    def cierre():
        opt.zero_grad()
        lg = mod(m0, m1, cxb)
        p = torch.nn.functional.cross_entropy(lg, yt, weight=w_clase) + l2 * mod.w.pow(2).sum()
        if con_caja:
            p = p + l2 * mod.wc.pow(2)
        p.backward()
        return p

    opt.step(cierre)
    mod.eval()
    return mod


def logits_sonda_zona(mod, M, idx, cx, con_caja, espejo=False):
    torch = _torch()
    with torch.no_grad():
        m0 = torch.as_tensor(M[idx, 0], dtype=torch.float64)
        m1 = torch.as_tensor(M[idx, 1], dtype=torch.float64)
        cxb = torch.as_tensor(cx[idx], dtype=torch.float64) if con_caja else None
        if espejo:
            m0, m1 = m1, m0
            if cxb is not None:
                cxb = -cxb
        return mod(m0, m1, cxb).numpy()


def decide(L):
    return np.asarray(L).argmax(1)


def control_espejo(mod, M, idx, cx, con_caja):
    L = logits_sonda_zona(mod, M, idx, cx, con_caja, espejo=False)
    Le = logits_sonda_zona(mod, M, idx, cx, con_caja, espejo=True)
    pred, pred_e = decide(L), decide(Le)
    return float(((4 - pred) == pred_e).mean())


# ---------------------------------------------------------------------------------------------
# ESTADISTICA (igual estilo que `mide_zona.py` / `sonda_vjepa.py`)
# ---------------------------------------------------------------------------------------------

def boot_clip(ok_a, ok_b, n=B_BOOT, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    d = ok_b.astype("float64") - ok_a.astype("float64")
    N = len(d)
    if N == 0:
        return None, None
    ds = [d[rng.integers(0, N, N)].mean() for _ in range(n)]
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def boot_torneo(ok_a, ok_b, torneos, n=B_BOOT, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    ts = np.unique(torneos)
    if len(ts) == 0:
        return None, None
    idx = {t: np.nonzero(torneos == t)[0] for t in ts}
    d = ok_b.astype("float64") - ok_a.astype("float64")
    ds = []
    for _ in range(n):
        sel = np.concatenate([idx[t] for t in rng.choice(ts, len(ts), replace=True)])
        ds.append(d[sel].mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


# ---------------------------------------------------------------------------------------------
# ELECCION DE HIPERPARAMETROS, SOLO dentro de AJUSTE
# ---------------------------------------------------------------------------------------------

def divide_interno(torneos_aj, frac_train=0.75, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    ts = np.array(sorted(torneos_aj))
    perm = rng.permutation(len(ts))
    n_tr = int(round(frac_train * len(ts)))
    tr_t = set(ts[perm[:n_tr]].tolist())
    va_t = set(ts[perm[n_tr:]].tolist())
    return tr_t, va_t


def elige_sonda(M, cx, y, torneos, tr_idx, va_idx, con_caja, rapido=False, verboso=True):
    rejilla = [(l2, p) for l2 in GRID_L2 for p in GRID_PESOS]
    if rapido:
        rejilla = rejilla[:2]
    mejor = None
    for l2, pesos in rejilla:
        mod = entrena_sonda_zona(M, tr_idx, y, l2, pesos, con_caja, cx)
        L = logits_sonda_zona(mod, M, va_idx, cx, con_caja)
        acc = float((decide(L) == y[va_idx]).mean())
        ll = _nll(L, y[va_idx])
        if verboso:
            print("   interna %-8s L2 %.0e pesos %-3s  acierto %.3f  nll %.4f"
                  % ("caja" if con_caja else "sin_caja", l2, "si" if pesos else "no", acc, ll),
                  flush=True)
        if mejor is None or (acc, -ll) > (mejor[2], -mejor[3]):
            mejor = (l2, pesos, acc, ll)
    return mejor


def _nll(L, y):
    L = L - L.max(1, keepdims=True)
    return float(-(L[np.arange(len(y)), y] - np.log(np.exp(L).sum(1))).mean())


# ---------------------------------------------------------------------------------------------
# FUSION (uso 2): logistica estandar sobre [onehot(zona_final), logits de la sonda]
# ---------------------------------------------------------------------------------------------

def onehot_zona(valores):
    out = np.zeros((len(valores), 5), dtype="float64")
    for i, v in enumerate(valores):
        if v in IDX:
            out[i, IDX[v]] = 1.0
    return out


def rasgos_fusion(zona_final, logits):
    return np.concatenate([onehot_zona(zona_final), logits.astype("float64")], axis=1)


def elige_fusion(X, y, torneos, tr_t, va_t, rapido=False, verboso=True):
    from sklearn.linear_model import LogisticRegression
    tr = np.array([t in tr_t for t in torneos])
    va = np.array([t in va_t for t in torneos])
    rejilla = GRID_C_FUSION[:1] if rapido else GRID_C_FUSION
    mejor = None
    for c in rejilla:
        clf = LogisticRegression(solver="lbfgs", C=c, max_iter=2000)
        clf.fit(X[tr], y[tr])
        acc = float(clf.score(X[va], y[va]))
        if verboso:
            print("   interna fusion C %-6s  acierto %.3f" % (c, acc), flush=True)
        if mejor is None or acc > mejor[1]:
            mejor = (c, acc)
    return mejor[0]


# ---------------------------------------------------------------------------------------------
# INFORME (confusion, a-una-zona-o-menos, por arma/via)
# ---------------------------------------------------------------------------------------------

def matriz_confusion(y, pred):
    M = np.zeros((5, 5), dtype=int)
    for a, b in zip(y, pred):
        M[a, b] += 1
    return M


def a_una_zona(y, pred):
    return float((np.abs(y - pred) <= 1).mean())


def por_grupo(nombre, grupo, y, pred, y_base_ok=None, etiqueta="sonda", etiqueta_base="middle"):
    out = []
    for g in sorted(set(str(x) for x in grupo)):
        m = np.array([str(x) == g for x in grupo])
        n = int(m.sum())
        if n == 0:
            continue
        acc = float((pred[m] == y[m]).mean())
        linea = "   %-10s %-24s n=%5d  %s %.3f" % (nombre, g, n, etiqueta, acc)
        if y_base_ok is not None:
            linea += "  %s %.3f  dif %+.3f" % (etiqueta_base, y_base_ok[m].mean(),
                                               acc - y_base_ok[m].mean())
        out.append(linea)
    return out


# ---------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="rejilla minima: SOLO para probar el script")
    a = ap.parse_args()

    torch = _torch()
    torch.set_num_threads(4)

    print("SONDA DE ZONA SOBRE V-JEPA 2 (protocolo VJEPA_ZONA_2026-10-09)")
    print("=" * 88)

    filas = construye_dataset()
    print("clips con strip_location + zona_ + embedding V-JEPA: %d" % len(filas))
    t0 = time.time()
    M, CJ, via_ancla = carga_media(filas)
    print("media cargada: %.1f MB en %.1f s" % (M.nbytes / 1e6, time.time() - t0), flush=True)
    Wmap = ancho_marco([f["clip"] for f in filas])
    n_sin_w = sum(v is None for v in Wmap.values())
    cx = np.array([rasgo_caja(CJ[i], Wmap[f["clip"]]) for i, f in enumerate(filas)], dtype="float64")
    print("ancho de marco (W) que falta (cx=0.0 neutro): %d de %d" % (n_sin_w, len(filas)))

    y = np.array([IDX[f["y"]] for f in filas])
    torneos = np.array([f["torneo"] for f in filas])
    armas = np.array([f["weapon"] for f in filas])
    zona_h = np.array([f["zona"] for f in filas], dtype=object)
    zona_final = np.array([f["zona_final"] for f in filas], dtype=object)
    via_zona = np.array([f["via_zona"] or "sin_zona" for f in filas], dtype=object)
    contesta = np.array([z in CA.ZONAS for z in zona_h])

    ajuste_torneos = set(AJUSTE_TXT.read_text(encoding="utf-8").split())
    es_aj = np.array([t in ajuste_torneos for t in torneos])
    aj = np.nonzero(es_aj)[0]
    pr = np.nonzero(~es_aj)[0]
    print("AJUSTE: %d clips, %d torneos; PRUEBA: %d clips, %d torneos"
          % (len(aj), len(set(torneos[aj])), len(pr), len(set(torneos[pr]))))

    tr_t, va_t = divide_interno(set(torneos[aj].tolist()))
    tr_idx = np.array([i for i in aj if torneos[i] in tr_t])
    va_idx = np.array([i for i in aj if torneos[i] in va_t])
    print("particion interna de AJUSTE: entrenamiento %d clips (%d torneos), validacion %d clips (%d torneos)"
          % (len(tr_idx), len(tr_t), len(va_idx), len(va_t)), flush=True)

    print()
    print("ELIGIENDO HIPERPARAMETROS (sin mirar PRUEBA)")
    mejor_sin = elige_sonda(M, cx, y, torneos, tr_idx, va_idx, con_caja=False, rapido=a.rapido)
    print("elegida SIN caja: L2 %.0e pesos %s (interna %.3f)" % (mejor_sin[0], mejor_sin[1], mejor_sin[2]))
    mejor_con = elige_sonda(M, cx, y, torneos, tr_idx, va_idx, con_caja=True, rapido=a.rapido)
    print("elegida CON caja: L2 %.0e pesos %s (interna %.3f)" % (mejor_con[0], mejor_con[1], mejor_con[2]))

    print()
    print("REENTRENANDO EN TODO AJUSTE con lo elegido")
    mod_sin = entrena_sonda_zona(M, aj, y, mejor_sin[0], mejor_sin[1], con_caja=False, cx=cx)
    mod_con = entrena_sonda_zona(M, aj, y, mejor_con[0], mejor_con[1], con_caja=True, cx=cx)

    esp_sin_aj = control_espejo(mod_sin, M, aj, cx, con_caja=False)
    esp_con_aj = control_espejo(mod_con, M, aj, cx, con_caja=True)
    print("control del espejo (AJUSTE) sin_caja %.4f  con_caja %.4f" % (esp_sin_aj, esp_con_aj))

    # ---- FUSION: elegir C en la particion interna (solo clips que contestan) ----
    print()
    print("ELIGIENDO C DE LA FUSION (solo AJUSTE que contesta, sin mirar PRUEBA)")
    L_sin_aj = logits_sonda_zona(mod_sin, M, aj, cx, con_caja=False)
    L_sin_aj_by_i = {i: L_sin_aj[j] for j, i in enumerate(aj)}
    con_aj = aj[contesta[aj]]
    Xf_aj = rasgos_fusion(zona_final[con_aj], np.stack([L_sin_aj_by_i[i] for i in con_aj]))
    yf_aj = y[con_aj]
    tf_aj = torneos[con_aj]
    c_fusion = elige_fusion(Xf_aj, yf_aj, tf_aj, tr_t, va_t, rapido=a.rapido)
    print("elegida C fusion: %s" % c_fusion)
    from sklearn.linear_model import LogisticRegression
    clf_fusion = LogisticRegression(solver="lbfgs", C=c_fusion, max_iter=2000)
    clf_fusion.fit(Xf_aj, yf_aj)

    print()
    print("=" * 88)
    print("MIDIENDO UNA VEZ EN PRUEBA")
    print("=" * 88)

    L_sin_pr = logits_sonda_zona(mod_sin, M, pr, cx, con_caja=False)
    L_con_pr = logits_sonda_zona(mod_con, M, pr, cx, con_caja=True)
    pred_sin_pr = decide(L_sin_pr)
    pred_con_pr = decide(L_con_pr)
    esp_sin_pr = control_espejo(mod_sin, M, pr, cx, con_caja=False)
    esp_con_pr = control_espejo(mod_con, M, pr, cx, con_caja=True)
    print("control del espejo (PRUEBA) sin_caja %.4f  con_caja %.4f" % (esp_sin_pr, esp_con_pr))

    y_pr = y[pr]
    torneos_pr = torneos[pr]
    armas_pr = armas[pr]
    via_zona_pr = via_zona[pr]
    contesta_pr = contesta[pr]
    ok_middle_pr = (y_pr == IDX["middle"])

    # ---------------- USO 1: RESPALDO (no contesta) ----------------
    no_c = np.nonzero(~contesta_pr)[0]
    print()
    print("USO 1 (RESPALDO) -- PRUEBA que NO contesta la produccion: n=%d" % len(no_c))
    ok_sonda_1 = (pred_sin_pr[no_c] == y_pr[no_c]).astype("float64")
    ok_middle_1 = ok_middle_pr[no_c].astype("float64")
    acc_s1, acc_m1 = ok_sonda_1.mean(), ok_middle_1.mean()
    lo1, hi1 = boot_clip(ok_middle_1, ok_sonda_1)
    lo1t, hi1t = boot_torneo(ok_middle_1, ok_sonda_1, torneos_pr[no_c])
    pasa1 = (acc_s1 - acc_m1 >= 0.05) and lo1 is not None and lo1 > 0
    print("   sonda(sin caja) %.3f  middle %.3f  dif %+.3f" % (acc_s1, acc_m1, acc_s1 - acc_m1))
    print("   IC95 por clip   [%+.3f, %+.3f]" % (lo1, hi1))
    print("   IC95 por torneo [%+.3f, %+.3f]" % (lo1t, hi1t))
    print("   CRITERIO (dif >= +0,05 e IC95 por clip > 0): %s" % ("PASA" if pasa1 else "NO PASA"))
    ok_sonda_1c = (pred_con_pr[no_c] == y_pr[no_c]).astype("float64")
    lo1c, hi1c = boot_clip(ok_middle_1, ok_sonda_1c)
    print("   (informativo, CON caja) sonda %.3f  dif %+.3f  IC95 clip [%+.3f, %+.3f]"
          % (ok_sonda_1c.mean(), ok_sonda_1c.mean() - acc_m1, lo1c, hi1c))
    print("   por arma:")
    for l in por_grupo("arma", armas_pr[no_c], y_pr[no_c], pred_sin_pr[no_c], ok_middle_1):
        print(l)

    # ---------------- USO 2: FUSION (si contesta) ----------------
    si_c = np.nonzero(contesta_pr)[0]
    print()
    print("USO 2 (FUSION) -- PRUEBA que SI contesta la produccion: n=%d" % len(si_c))
    Xf_pr = rasgos_fusion(zona_final[pr][si_c], L_sin_pr[si_c])
    pred_fusion = clf_fusion.predict(Xf_pr)
    y2 = y_pr[si_c]
    zf_idx = np.array([IDX.get(z, -1) for z in zona_final[pr][si_c]])
    ok_zf = (zf_idx == y2).astype("float64")
    ok_fusion = (pred_fusion == y2).astype("float64")
    acc_zf, acc_fus = ok_zf.mean(), ok_fusion.mean()
    lo2, hi2 = boot_clip(ok_zf, ok_fusion)
    lo2t, hi2t = boot_torneo(ok_zf, ok_fusion, torneos_pr[si_c])
    pasa2 = (acc_fus - acc_zf >= 0.02) and lo2t is not None and lo2t > 0
    print("   zona_final %.3f  fusion %.3f  dif %+.3f" % (acc_zf, acc_fus, acc_fus - acc_zf))
    print("   IC95 por clip   [%+.3f, %+.3f]" % (lo2, hi2))
    print("   IC95 por torneo [%+.3f, %+.3f]" % (lo2t, hi2t))
    print("   CRITERIO (dif >= +0,02 e IC95 por TORNEO > 0): %s" % ("PASA" if pasa2 else "NO PASA"))
    print("   por arma:")
    for l in por_grupo("arma", armas_pr[si_c], y2, pred_fusion, ok_zf,
                       etiqueta="fusion", etiqueta_base="zona_final"):
        print(l)
    print("   por via_zona:")
    for l in por_grupo("via_zona", via_zona_pr[si_c], y2, pred_fusion, ok_zf,
                       etiqueta="fusion", etiqueta_base="zona_final"):
        print(l)

    # ---------------- USO 3: INFORMATIVO (todo PRUEBA) ----------------
    print()
    print("USO 3 (INFORMATIVO) -- TODO PRUEBA: n=%d" % len(pr))
    ok_sonda_3 = (pred_sin_pr == y_pr).astype("float64")
    ok_middle_3 = ok_middle_pr.astype("float64")
    acc_s3, acc_m3 = ok_sonda_3.mean(), ok_middle_3.mean()
    lo3, hi3 = boot_clip(ok_middle_3, ok_sonda_3)
    lo3t, hi3t = boot_torneo(ok_middle_3, ok_sonda_3, torneos_pr)
    print("   sonda(sin caja) %.3f  middle %.3f  dif %+.3f" % (acc_s3, acc_m3, acc_s3 - acc_m3))
    print("   IC95 por clip   [%+.3f, %+.3f]" % (lo3, hi3))
    print("   IC95 por torneo [%+.3f, %+.3f]" % (lo3t, hi3t))
    ok_sonda_3c = (pred_con_pr == y_pr).astype("float64")
    lo3c, hi3c = boot_clip(ok_middle_3, ok_sonda_3c)
    print("   (informativo, CON caja) sonda %.3f  dif %+.3f  IC95 clip [%+.3f, %+.3f]"
          % (ok_sonda_3c.mean(), ok_sonda_3c.mean() - acc_m3, lo3c, hi3c))
    print("   a una zona o menos: sin_caja %.3f  con_caja %.3f"
          % (a_una_zona(y_pr, pred_sin_pr), a_una_zona(y_pr, pred_con_pr)))
    print("   por arma:")
    for l in por_grupo("arma", armas_pr, y_pr, pred_sin_pr, ok_middle_3):
        print(l)
    print("   por via_zona:")
    for l in por_grupo("via_zona", via_zona_pr, y_pr, pred_sin_pr, ok_middle_3):
        print(l)
    print("   matriz de confusion (filas=humano, columnas=sonda), orden %s:" % CA.ZONAS)
    Mc = matriz_confusion(y_pr, pred_sin_pr)
    for i, fila in enumerate(Mc):
        print("      %-18s %s" % (CA.ZONAS[i], " ".join("%4d" % v for v in fila)))

    # ---------------- guarda salidas ----------------
    SALIDA.mkdir(parents=True, exist_ok=True)
    resumen = {
        "n_total": len(filas), "n_sin_ancho_marco": n_sin_w,
        "ajuste": {"n": int(len(aj)), "torneos": int(len(set(torneos[aj])))},
        "prueba": {"n": int(len(pr)), "torneos": int(len(set(torneos[pr])))},
        "hiperparametros": {"sin_caja": {"l2": mejor_sin[0], "pesos": mejor_sin[1], "interna": mejor_sin[2]},
                            "con_caja": {"l2": mejor_con[0], "pesos": mejor_con[1], "interna": mejor_con[2]},
                            "fusion_C": c_fusion},
        "control_espejo": {"ajuste_sin_caja": esp_sin_aj, "ajuste_con_caja": esp_con_aj,
                            "prueba_sin_caja": esp_sin_pr, "prueba_con_caja": esp_con_pr},
        "uso1_respaldo": {"n": int(len(no_c)), "sonda": acc_s1, "middle": acc_m1,
                          "dif": acc_s1 - acc_m1, "ic95_clip": [lo1, hi1], "ic95_torneo": [lo1t, hi1t],
                          "pasa": bool(pasa1), "con_caja": {"sonda": float(ok_sonda_1c.mean())}},
        "uso2_fusion": {"n": int(len(si_c)), "zona_final": acc_zf, "fusion": acc_fus,
                        "dif": acc_fus - acc_zf, "ic95_clip": [lo2, hi2], "ic95_torneo": [lo2t, hi2t],
                        "pasa": bool(pasa2)},
        "uso3_informativo": {"n": int(len(pr)), "sonda": acc_s3, "middle": acc_m3,
                             "dif": acc_s3 - acc_m3, "ic95_clip": [lo3, hi3], "ic95_torneo": [lo3t, hi3t],
                             "a_una_zona": a_una_zona(y_pr, pred_sin_pr),
                             "con_caja": {"sonda": float(ok_sonda_3c.mean())}},
        "confusion_prueba": Mc.tolist(),
    }
    (SALIDA / "resumen_prueba.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    with open(SALIDA / "predicciones_prueba.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "torneo", "weapon", "y", "zona_h", "zona_final", "via_zona", "contesta",
                    "pred_sonda_sin_caja", "pred_sonda_con_caja", "pred_fusion"])
        pf_by_i = {int(si_c[k]): pred_fusion[k] for k in range(len(si_c))}
        for k, i in enumerate(pr):
            f = filas[i]
            w.writerow([f["clip"], f["torneo"], f["weapon"], CA.ZONAS[y_pr[k]], f["zona"],
                        f["zona_final"], f["via_zona"], bool(contesta_pr[k]),
                        CA.ZONAS[pred_sin_pr[k]], CA.ZONAS[pred_con_pr[k]],
                        CA.ZONAS[pf_by_i[k]] if k in pf_by_i else ""])
    print()
    print("guardado: %s" % (SALIDA / "resumen_prueba.json"))
    print("guardado: %s" % (SALIDA / "predicciones_prueba.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
