"""La medida del 3D levantado (MotionAGFormer-B) contra el etiquetador en `initiated`.

El protocolo y el criterio estan en `04_documentacion/informes/3D_LEVANTADO_2026-10-09.md`, escritos
antes de este fichero. Aqui no se decide nada que no este alli.

    python verifica_3d.py [--salida D:/FENCINGLY_SCRATCH/agente_3d] [--semillas 5] [--max N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import etiquetador as ET  # noqa: E402
import levanta3d as L3  # noqa: E402
import rutas as RU  # noqa: E402


DEV = "cuda"
"""El etiquetador se entrena en la GPU solo para que 10 entrenamientos quepan en una tarde: en la CPU
cargada de hoy tardaba ~78 s por semilla con 109 clips. Mismo modelo y misma receta."""


def entrena_fijo(filas, tr, semilla, epocas=60, lote=32, lr=3e-3):
    """`etiquetador.entrena` con el reparto YA dado y la semilla de torch como parametro: mismo
    modelo, misma perdida con el espejo como aumento, mismas epocas, la ultima epoca."""
    import torch
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    mod = ET.construye().to(DEV)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    P, _A, S, y = (t.to(DEV) for t in ET._tensores(filas, tr, "initiated", torch))
    n = len(tr)
    for _ in range(epocas):
        mod.train()
        per = torch.randperm(n, device=DEV)
        for i in range(0, n, lote):
            j = per[i:i + lote]
            opt.zero_grad()
            p = torch.nn.functional.cross_entropy(mod.inicia(P[j], S[j]), y[j])
            ye = torch.where(y[j] == 0, torch.ones_like(y[j]),
                             torch.where(y[j] == 1, torch.zeros_like(y[j]), y[j]))
            p = p + torch.nn.functional.cross_entropy(mod.inicia(P[j][:, [1, 0]], S[j]), ye)
            p.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 2.0)
            opt.step()
    mod.eval()
    return mod.cpu()


def logits(mod, filas, idx, espejo=False):
    import torch
    P, _A, S, _y = ET._tensores(filas, idx, "initiated", torch)
    if espejo:
        P = P[:, [1, 0]]
    with torch.no_grad():
        return mod.inicia(P, S).numpy().astype("float64")


def boot_torneo(y, pa, pb, torneos, n=2000, semilla=ET.SEMILLA):
    rng = np.random.default_rng(semilla)
    ts = np.unique(torneos)
    idx = {t: np.nonzero(torneos == t)[0] for t in ts}
    a, b = (pa == y).astype(float), (pb == y).astype(float)
    ds = []
    for _ in range(n):
        sel = np.concatenate([idx[t] for t in rng.choice(ts, len(ts), replace=True)])
        ds.append(b[sel].mean() - a[sel].mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def brazo(filas, tr, re, semillas, n_prop):
    """Entrena `semillas` modelos con el reparto fijo; devuelve logits medios en `re` y su espejo."""
    viejo = ET.N_PROP
    ET.N_PROP = n_prop
    """`construye` lee `N_PROP` del modulo al llamarla: con el bloque propio ampliado hay que
    decirselo. Se restaura siempre."""
    try:
        Ls, Le, t0 = [], [], time.time()
        for k in range(semillas):
            m = entrena_fijo(filas, tr, ET.SEMILLA + k)
            Ls.append(logits(m, filas, re))
            Le.append(logits(m, filas, re, espejo=True))
            print("      semilla %d  acierto %.3f  (%.0f s)"
                  % (ET.SEMILLA + k, (Ls[-1].argmax(1) == np.array([filas[i]["y_initiated"] for i in re])).mean(),
                     time.time() - t0), flush=True)
        return np.mean(Ls, 0), np.mean(Le, 0), Ls
    finally:
        ET.N_PROP = viejo


def espeja_npz64(clip, destino):
    """`verifica_etiquetador.espeja_npz` con los puntos en float64. Aquel guarda `W - x` en float32,
    que REDONDEA en uno de los dos tiradores (el de la izquierda: `W - x` sube de magnitud), y el
    levantador -- que elige vecinos temporales por similitud, una seleccion discontinua -- convierte
    ese ultimo bit en diferencias de 1e-3. La prueba es de los RASGOS, asi que la entrada tiene que
    ser el espejo exacto (cambio 2 del protocolo)."""
    import verifica_etiquetador as VE
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    with np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    W = float(d["W"])
    KP = d["KP"].astype("float64")
    SC = d["SC"].copy()
    KP[:, :, :, 0] = W - KP[:, :, :, 0]
    perm = np.arange(KP.shape[2])
    for a, b in VE.PARES:
        if a < len(perm) and b < len(perm):
            perm[a], perm[b] = b, a
    d["KP"] = KP[:, :, perm][:, [1, 0]]
    d["SC"] = SC[:, :, perm][:, [1, 0]]
    for k in ("B", "Be"):
        b = d[k].copy()
        x1, x2 = 1.0 - b[:, :, 2], 1.0 - b[:, :, 0]
        b[:, :, 0], b[:, :, 2] = x1, x2
        d[k] = b[:, [1, 0]]
    for k in ("pres", "origen", "hueco"):
        if k in d:
            d[k] = d[k][:, [1, 0]]
    np.savez(destino / ("poses_%s_ESPEJO.npz" % clip), **d)
    return "%s_ESPEJO" % clip


def prueba_espejo_3d(mod, clips, destino, tol=1e-3):
    peor, mal, n = 0.0, [], 0
    for c in clips:
        r = L3.levanta_npz(mod, RU.TRABAJO / ("poses_%s.npz" % c))
        f = L3.rasgos_clip3d(r)
        e = espeja_npz64(c, destino)
        p = destino / ("poses_%s.npz" % e)
        fe = L3.rasgos_clip3d(L3.levanta_npz(mod, p))
        p.unlink(missing_ok=True)
        if f is None or fe is None:
            continue
        n += 1
        d = np.abs(f - fe[[1, 0]])
        esc = np.maximum(np.abs(f).max(axis=(0, 1)), 1e-6)
        rel = d.max(axis=(0, 1)) / esc
        if rel.max() > peor:
            peor = float(rel.max())
        if rel.max() > tol:
            mal.append((c, L3.RASGOS_3D[int(rel.argmax())], float(rel.max())))
    return n, mal, peor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="D:/FENCINGLY_SCRATCH/agente_3d")
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--max", type=int, default=0, help="solo N clips: para probar la maquinaria")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    sal = pathlib.Path(a.salida)
    (sal / "salidas").mkdir(parents=True, exist_ok=True)
    print("3D LEVANTADO CONTRA EL ETIQUETADOR (protocolo 3D_LEVANTADO_2026-10-09)")
    print("=" * 80)
    mod = L3.carga_modelo()
    if a.max:
        import catalogo as CA
        df = CA.manifiesto()
        ids = sorted(df.loc[df["initiated"].isin(CA.INICIA), "clip_id"])[:a.max]
        filas = ET.reune(clips=ids, cabeza="initiated")
    else:
        filas = ET.reune(cabeza="initiated")
    itr, ire, diag = ET.particion_por_torneo(filas)
    print("reparto: %d torneos / %d clips entrenamiento, %d / %d reservado"
          % (diag["torneos_tr"], diag["n_tr"], diag["torneos_re"], diag["n_re"]), flush=True)

    print("\n1. LEVANTAR Y RASGOS")
    t0 = time.time()
    tg, plaus, malos = [], [], 0
    f3 = {}
    import torch as _t
    for i, f in enumerate(filas):
        c = f["clip"]
        try:
            t1 = time.time()
            r = L3.levanta_npz(mod, RU.TRABAJO / ("poses_%s.npz" % c))
            _t.cuda.synchronize()
            tg.append(time.time() - t1)
            x = L3.rasgos_clip3d(r)
            if x is None:
                malos += 1
                x = np.zeros((2, ET.LARGO, len(L3.RASGOS_3D)), dtype="float32")
            f3[c] = x
            plaus.append(L3.plausibilidad(r))
            np.savez_compressed(sal / "salidas" / ("l3d_%s.npz" % c), P3=r["P3"].astype("float16"))
        except Exception as ex:
            malos += 1
            f3[c] = np.zeros((2, ET.LARGO, len(L3.RASGOS_3D)), dtype="float32")
            print("   FALLA %s %s: %s" % (c, type(ex).__name__, str(ex)[:120]), flush=True)
        if (i + 1) % 500 == 0:
            print("   %d / %d  %.0f s" % (i + 1, len(filas), time.time() - t0), flush=True)
    print("   clips %d, sin rasgos 3D %d; levantar+leer %.3f s/clip (mediana), total %.0f s"
          % (len(filas), malos, float(np.median(tg)), time.time() - t0))
    cvm = np.array([p[k]["cv_mediana"] for p in plaus for k in (0, 1)])
    cvx = np.array([p[k]["cv_max"] for p in plaus for k in (0, 1)])
    asi = np.array([p[k]["asim_media"] for p in plaus for k in (0, 1)])
    print("   plausibilidad (n=%d tiradores): CV de hueso mediana %.3f (p90 %.3f); peor hueso mediana"
          " %.3f (p90 %.3f); asimetria I/D media %.3f (p90 %.3f)"
          % (len(cvm), np.median(cvm), np.percentile(cvm, 90), np.median(cvx),
             np.percentile(cvx, 90), np.median(asi), np.percentile(asi, 90)))

    print("\n2. ESPEJO DE LOS RASGOS 3D (npz espejado de verdad)")
    n, mal, peor = prueba_espejo_3d(mod, [f["clip"] for f in filas[:12]], sal)
    print("   clips %d, con discrepancia > 1e-3: %d, peor %.2e" % (n, len(mal), peor))
    for m in mal[:5]:
        print("   MAL", m)

    print("\n3. A CONTRA A+3D, %d semillas, reparto fijo" % a.semillas)
    filas3 = []
    for f in filas:
        g = dict(f)
        g["prop"] = np.concatenate([f["prop"], f3[f["clip"]]], axis=2)
        filas3.append(g)
    y = np.array([filas[i]["y_initiated"] for i in ire])
    tor = np.array([filas[i]["torneo"] for i in ire])
    arm = np.array([filas[i]["weapon"] for i in ire])
    print("   A:")
    La, La_e, Las = brazo(filas, itr, ire, a.semillas, ET.N_PROP)
    print("   A+3D:")
    Lb, Lb_e, Lbs = brazo(filas3, itr, ire, a.semillas, ET.N_PROP + len(L3.RASGOS_3D))
    pa, pb = La.argmax(1), Lb.argmax(1)
    acc_a, acc_b = float((pa == y).mean()), float((pb == y).mean())
    lo, hi = boot_torneo(y, pa, pb, tor)

    def esp(L, Le):
        p, q = L.argmax(1), Le.argmax(1)
        return float((np.where(p == 0, 1, np.where(p == 1, 0, 2)) == q).mean())

    e_a, e_b = esp(La, La_e), esp(Lb, Lb_e)
    s1a, s1b = float((Las[0].argmax(1) == y).mean()), float((Lbs[0].argmax(1) == y).mean())
    print()
    print("RESERVADO: %d clips, %d torneos" % (len(y), len(np.unique(tor))))
    print("   A      %.3f   (semilla unica 20260922: %.3f)" % (acc_a, s1a))
    print("   A+3D   %.3f   (semilla unica 20260922: %.3f)" % (acc_b, s1b))
    print("   diferencia %+.3f   IC95 por torneo [%+.3f, %+.3f]" % (acc_b - acc_a, lo, hi))
    print("   control del espejo: A %.4f   A+3D %.4f" % (e_a, e_b))
    for w in np.unique(arm):
        m = arm == w
        print("   %-6s n=%4d   A %.3f   A+3D %.3f" % (w, m.sum(), (pa[m] == y[m]).mean(),
                                                    (pb[m] == y[m]).mean()))
    pasa = (acc_b - acc_a >= 0.02) and lo > 0 and e_a == 1.0 and e_b == 1.0 and not mal and n >= 12
    print("\nCRITERIO: diferencia >= +0,02, IC95 sin cruzar cero, espejos 1,0000, espejo de rasgos"
          " <= 1e-3 en >= 12 clips  ->  %s" % ("PASA" if pasa else "NO PASA"))
    res = {"n_re": int(len(y)), "torneos_re": int(len(np.unique(tor))), "acc_a": acc_a,
           "acc_3d": acc_b, "dif": acc_b - acc_a, "ic95": [lo, hi], "espejo_a": e_a,
           "espejo_3d": e_b, "espejo_rasgos_n": n, "espejo_rasgos_mal": len(mal),
           "espejo_rasgos_peor": peor, "s1_a": s1a, "s1_3d": s1b, "pasa": bool(pasa),
           "s_por_clip_mediana": float(np.median(tg)), "sin_3d": malos,
           "plaus_cv_mediana": float(np.median(cvm)), "plaus_asim_mediana": float(np.median(asi))}
    (sal / "resultado_3d.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
