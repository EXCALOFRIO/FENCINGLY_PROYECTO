"""La mano armada de la etapa 7 con y sin la cascada completa, contra la FICHA FEDERATIVA.

Criterio en `04_documentacion/informes/ETAPA7_PUERTAS_2026-10-09.md` §2, escrito antes de correr
esto. Las tres variantes llaman a la MISMA `cono_lib.mano_armada` con los MISMOS `KP[:T]`, `SC[:T]`
y `r["B"][:T]` que `posicion._brazos`, asi que lo que se mide es lo que correria:

  V0   (hoy)        mano_armada(KP, SC, B, W, H, lado)
  V0m  (atribuir)   + org=origen, mask
  V1   (candidato)  + org=origen, mask, accion = mascara de los eventos `accion` de la etapa 7

`accion` se lee de `eventos_<clip>.csv` (ya en disco): los eventos `accion` no dependen de la mano,
asi que son los mismos que calcularia el codigo nuevo. Solo CPU; `--procesos` en paralelo.

    python verifica_mano_accion.py [--conjunto anotados] [--procesos 6]
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys
from math import comb

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "01_extraccion"))
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))

SAL = pathlib.Path("D:/FENCINGLY_SCRATCH/agente_etapa7")
VARIANTES = ("V0", "V0m", "V1")


def _accion(clip, T):
    import rutas as RU
    p = RU.PISTA / ("eventos_%s.csv" % clip)
    if not p.is_file():
        return None, "sin_eventos"
    tramos = []
    with open(p, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f.get("tipo") != "accion":
                continue
            try:
                tramos.append((int(float(f["inicio"])), int(float(f["fin"]))))
            except (TypeError, ValueError, KeyError):
                continue
    if not tramos:
        return None, "sin_accion"
    m = np.zeros(int(T), bool)
    for a, b in tramos:
        m[max(0, a):min(int(T), b + 1)] = True
    return m, "con_accion"


def uno(clip):
    import cono_lib as CL
    import pista as PI
    import posicion as PO
    try:
        r = PI.pareja(clip)
        poses = PO.carga_poses(clip)
    except Exception as ex:
        return [{"clip": clip, "error": type(ex).__name__}]
    if r is None or poses is None:
        return [{"clip": clip, "error": "sin_det_o_poses"}]
    KP, SC = poses["KP"], poses["SC"]
    T = min(KP.shape[0], r["T"])
    org = poses.get("origen")
    org = org[:T] if org is not None else None
    mk = np.asarray(r["mask"][:T], bool)
    acc, via = _accion(clip, T)
    out = []
    for lado in (0, 1):
        f = {"clip": clip, "lado": lado, "via_accion": via}
        for v, kw in (("V0", {}), ("V0m", {"org": org, "mask": mk}),
                      ("V1", {"org": org, "mask": mk, "accion": acc})):
            try:
                m, d = CL.mano_armada(KP[:T], SC[:T], r["B"][:T], r["W"], r["H"], lado, **kw)
            except Exception as ex:
                m, d = None, {"por": "EXC_" + type(ex).__name__}
            f[v] = m
            f[v + "_por"] = (d or {}).get("por")
        out.append(f)
    return out


def mcnemar(a_ok, b_ok):
    gana = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    pierde = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = gana + pierde
    p = None
    if n:
        k = min(gana, pierde)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return gana, pierde, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--procesos", type=int, default=6)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    import catalogo as CA
    import fie as FIE
    import lotes as LO
    import rutas as RU
    ids = [c for c in LO.conjunto(a.conjunto)
           if (RU.TRABAJO / ("poses_%s.npz" % c)).is_file()
           and (RU.TRABAJO / ("det_%s.pkl" % c)).is_file()]
    if a.max:
        ids = ids[:a.max]
    print("clips con poses_ y det_: %d" % len(ids), flush=True)
    filas = []
    from multiprocessing import Pool
    with Pool(a.procesos) as pool:
        for i, res in enumerate(pool.imap_unordered(uno, ids, chunksize=8)):
            filas += res
            if (i + 1) % 500 == 0:
                print("   %d / %d" % (i + 1, len(ids)), flush=True)
    SAL.mkdir(parents=True, exist_ok=True)
    cols = ["clip", "lado", "via_accion", "error"] + [x for v in VARIANTES for x in (v, v + "_por")]
    with open(SAL / "mano_accion_por_clip.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    df = CA.manifiesto().set_index("clip_id")
    mf = FIE.manos()
    base, nb = FIE.base_diestros()
    err = collections.Counter(f["error"] for f in filas if f.get("error"))
    dec = []
    for f in filas:
        if f.get("error") or f["clip"] not in df.index:
            continue
        nom = FIE.norm(df.loc[f["clip"], "fotl_name" if f["lado"] == 0 else "fotr_name"])
        h = mf.get(nom)
        if h not in ("L", "R"):
            continue
        f["nombre"], f["verdad"] = nom, ("A" if h == "R" else "B")
        f["arma"] = df.loc[f["clip"], "weapon"]
        dec.append(f)
    print()
    print("MANO ARMADA DE LA ETAPA 7 CONTRA LA FICHA FIE")
    print("=" * 84)
    print("errores por clip: %s" % dict(err))
    print("decisiones (clip, lado) con ficha: %d   tiradores: %d   azar informado (siempre R) %.3f (n=%d)"
          % (len(dec), len({f["nombre"] for f in dec}), base, nb))

    def tabla(sub, titulo):
        print()
        print(titulo)
        print("%-6s %8s %10s %10s %7s   | por tirador: %8s %10s %10s %6s"
              % ("", "acierto", "cobertura", "acierto*", "n", "acierto", "cobertura", "acierto*", "n"))
        res = {}
        for v in VARIANTES:
            resp = [f for f in sub if f[v] in ("A", "B")]
            ok = sum(1 for f in resp if f[v] == f["verdad"])
            votos = collections.defaultdict(list)
            verdad = {}
            for f in sub:
                verdad[f["nombre"]] = f["verdad"]
                if f[v] in ("A", "B"):
                    votos[f["nombre"]].append(f[v])
            tok = tresp = 0
            dec_t = {}
            for nom, vs in votos.items():
                cnt = collections.Counter(vs).most_common()
                if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
                    continue
                tresp += 1
                dec_t[nom] = cnt[0][0]
                tok += int(cnt[0][0] == verdad[nom])
            nt = len(verdad)
            res[v] = {"dec_t": dec_t, "verdad": verdad, "clip_ok": [f[v] == f["verdad"] for f in sub]}
            print("%-6s %8.3f %10.3f %10.3f %7d   |              %8.3f %10.3f %10.3f %6d"
                  % (v, ok / max(len(resp), 1), len(resp) / max(len(sub), 1), ok / max(len(sub), 1),
                     len(sub), tok / max(tresp, 1), tresp / max(nt, 1), tok / max(nt, 1), nt))
        return res

    res = tabla(dec, "TODAS LAS ARMAS")
    for arma in ("foil", "epee", "sabre"):
        tabla([f for f in dec if f["arma"] == arma], "arma = %s" % arma)
    for via in ("con_accion", "sin_accion", "sin_eventos"):
        sub = [f for f in dec if f["via_accion"] == via]
        if sub:
            tabla(sub, "via_accion = %s" % via)

    print()
    print("PAREADO POR TIRADOR (los decididos por las dos variantes)")
    for a_, b_ in (("V1", "V0"), ("V0m", "V0"), ("V1", "V0m")):
        ra, rb = res[a_], res[b_]
        noms = [n for n in ra["verdad"] if n in ra["dec_t"] and n in rb["dec_t"]]
        aok = [ra["dec_t"][n] == ra["verdad"][n] for n in noms]
        bok = [rb["dec_t"][n] == rb["verdad"][n] for n in noms]
        g, p_, pv = mcnemar(aok, bok)
        print("   %-4s contra %-4s  gana %4d  pierde %4d  p=%s   (n=%d)"
              % (a_, b_, g, p_, "%.2g" % pv if pv is not None else "-", len(noms)))
    g, p_, pv = mcnemar(res["V1"]["clip_ok"], res["V0"]["clip_ok"])
    print("   por clip V1 contra V0: gana %d pierde %d p=%s" % (g, p_, "%.2g" % pv if pv else "-"))
    por = collections.Counter(f["V1_por"] for f in dec)
    por0 = collections.Counter(f["V0_por"] for f in dec)
    print()
    print("peldano que decide  V0: %s" % dict(por0.most_common()))
    print("                    V1: %s" % dict(por.most_common()))
    cambian = sum(1 for f in dec if f["V1"] != f["V0"])
    print("decisiones que cambian de mano V0 -> V1: %d de %d" % (cambian, len(dec)))

    vm = RAIZ / "02_validacion" / "muestras" / "mano_dominante" / "verdad_mano.csv"
    if vm.is_file():
        with open(vm, encoding="utf-8") as fh:
            ojo = list(csv.DictReader(l for l in fh if not l.startswith("#")))
        nom_ojo = set()
        for o in ojo:
            if o["clip"] in df.index:
                nom_ojo.add(FIE.norm(df.loc[o["clip"], "fotl_name" if o["lado"] == "0" else "fotr_name"]))
        print("tiradores anotados a ojo (con los que se eligio la cascada) entre los de la ficha: %d"
              % len(nom_ojo & {f["nombre"] for f in dec}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
