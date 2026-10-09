"""Evalua las reglas R0..R4 declaradas en LUCES_POR_FUENTE_2026-10-09.md sobre el recalculo de mide.py.

    python evalua.py                -> tablas por mitad (AJUSTE / PRUEBA), arma, via y torneo
"""
import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA  # noqa: E402

D = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_luces")
AUD = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\auditoria"
                   r"\agentes_2026-10-09\nuevas_senales\audio_cobertura_global.csv")
SEMILLA = 20261009
REGLAS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def lee(p):
    out = {}
    if p.is_file():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out.setdefault(r["clip"], {}).update(r)
    return out


def de_vigente(v):
    if not v or "error" in v:
        return "sin", None, None
    col = sorted({l["canal"] for l in v["luces"] if l["canal"] in ("roja", "verde")})
    pred = {("roja",): "left", ("verde",): "right", ("roja", "verde"): "dos"}.get(tuple(col), "sin")
    c = min((l["cuadro_encendido"] for l in v["luces"]), default=None)
    return pred, c, v.get("via")


def de_viejo(v):
    if not v or "error" in v and v.get("source") is None:
        return "sin", None, None
    cu = v.get("cuadros") or {}
    a, b = cu.get("red_L") is not None, cu.get("green_R") is not None
    pred = "dos" if a and b else "left" if a else "right" if b else "sin"
    vals = [x for x in cu.values() if x is not None]
    return pred, (min(vals) if vals else None), "v2_" + str(v.get("source"))


def tabla_clips():
    med = lee(D / "med_anotados.jsonl")
    r3 = lee(D / "med_r3.jsonl")
    df = CA.manifiesto().set_index("clip_id")
    aud = pd.read_csv(AUD).set_index("clip")["tiene_audio"].to_dict() if AUD.is_file() else {}
    filas = []
    for c, r in med.items():
        if "vigente" not in r:
            continue
        f = dict(clip=c, torneo=r["torneo"], arma=r["arma"], touch=df.loc[c, "touch"],
                 audio=aud.get(c), s_vig=r.get("s_vigente"), s_vie=r.get("s_viejo"))
        vj = r.get("viejo") or {}
        f["v2_fuente"] = vj.get("source")
        f["v2_cajas"] = vj.get("box_method")
        f["H"] = (vj.get("size") or [None, None])[1]
        f["fps"] = vj.get("fps")
        p0, c0, via0 = de_vigente(r.get("vigente"))
        p1, c1, via1 = de_viejo(vj)
        f.update(R0=p0, R0_c=c0, R0_via=via0, R1=p1, R1_c=c1, R1_via=via1)
        hay_v = vj.get("f_touch") is not None
        f.update(R2=p1 if hay_v else p0, R2_c=c1 if hay_v else c0,
                 R2_via=via1 if hay_v else "vig_tras_v2:" + str(via0))
        es_fie = vj.get("source") == "fie"
        tiene3 = c in r3 and "r3" in r3[c]
        if tiene3:
            p3, c3, via3 = de_vigente(r3[c]["r3"])
            f.update(R3=p3, R3_c=c3, R3_via=via3)
        # R4 y R6 solo necesitan R3 donde el viejo no contesta / no es FIE: R3 se calcula ahi siempre
        if hay_v:
            f.update(R4=p1, R4_c=c1, R4_via=via1)
        elif tiene3:
            f.update(R4=p3, R4_c=c3, R4_via="r3_tras_v2:" + str(via3))
        if es_fie:
            f.update(R5=p1, R5_c=c1, R5_via=via1)
        elif c0 is not None:
            f.update(R5=p0, R5_c=c0, R5_via="vig_no_fie:" + str(via0))
        else:
            f.update(R5=p1, R5_c=c1, R5_via=via1)
        if es_fie:
            f.update(R6=p1, R6_c=c1, R6_via=via1)
        elif tiene3 and c3 is not None:
            f.update(R6=p3, R6_c=c3, R6_via="r3_no_fie:" + str(via3))
        elif tiene3:
            f.update(R6=p1, R6_c=c1, R6_via=via1)
        filas.append(f)
    return pd.DataFrame(filas)


def mitades(d):
    tor = sorted(d["torneo"].unique())
    random.Random(SEMILLA).shuffle(tor)
    aj = set(tor[0::2])
    return d["torneo"].map(lambda t: "AJUSTE" if t in aj else "PRUEBA")


def metricas(g, R):
    if R not in g or g[R].isna().all():
        return {}
    lr = g["touch"].isin(["left", "right"])
    tod = g["touch"].isin(["left", "right", "double"])
    p = g[R]
    una = p.isin(["left", "right"]) & lr
    con_c = g[R + "_c"].notna()
    return {"n": int(len(g)), "n_A1": int(una.sum()),
            "A1": float((p[una] == g.loc[una, "touch"]).mean()) if una.any() else np.nan,
            "COB": float((p[tod] != "sin").mean()) if tod.any() else np.nan,
            "DOS": float((p[tod] == "dos").mean()) if tod.any() else np.nan,
            "ACx": float(((p == g["touch"]) & lr).sum() / max(1, lr.sum())),
            "C0": float((g.loc[con_c, R + "_c"] == 0).mean()) if con_c.any() else np.nan}


def boot_dif(g, Ra, Rb, campo="A1", B=2000, semilla=1):
    """IC95 de metrica(Rb) - metrica(Ra) remuestreando CLIPS (pareado: los mismos clips para las dos)."""
    rng = np.random.default_rng(semilla)
    lr = g["touch"].isin(["left", "right"]).to_numpy()
    t = g["touch"].to_numpy()
    pa, pb = g[Ra].to_numpy(), g[Rb].to_numpy()

    def m(p, idx):
        tt, pp, ll = t[idx], p[idx], lr[idx]
        if campo == "A1":
            u = np.isin(pp, ["left", "right"]) & ll
            return (pp[u] == tt[u]).mean() if u.any() else np.nan
        return ((pp == tt) & ll).sum() / max(1, ll.sum())
    n = len(g)
    base = m(pb, np.arange(n)) - m(pa, np.arange(n))
    ds = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ds.append(m(pb, idx) - m(pa, idx))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return base, lo, hi


def fmt(m):
    if not m:
        return "-"
    return ("n=%4d A1=%.3f(n=%4d) COB=%.3f DOS=%.3f ACx=%.3f C0=%.3f"
            % (m["n"], m["A1"], m["n_A1"], m["COB"], m["DOS"], m["ACx"], m["C0"]))


def main():
    d = tabla_clips()
    d["mitad"] = mitades(d)
    d.to_csv(D / "evalua_clips.csv", index=False)
    reglas = [R for R in REGLAS if R in d and d[R].notna().any()]
    print("clips evaluados: %d  (%s)" % (len(d), d["arma"].value_counts().to_dict()))
    print("torneos AJUSTE %d / PRUEBA %d; clips %s"
          % (d.loc[d.mitad == "AJUSTE", "torneo"].nunique(), d.loc[d.mitad == "PRUEBA", "torneo"].nunique(),
             d["mitad"].value_counts().to_dict()))
    for mitad in ("AJUSTE", "PRUEBA", "TODO"):
        g0 = d if mitad == "TODO" else d[d.mitad == mitad]
        print("\n==== %s ====" % mitad)
        for arma in ("foil", "epee", "sabre"):
            g = g0[g0.arma == arma]
            if not len(g):
                continue
            print("  %s" % arma)
            for R in reglas:
                gg = g[g[R].notna()]
                print("    %s  %s" % (R, fmt(metricas(gg, R))))
            for R in reglas[1:]:
                gg = g[g[R].notna() & g["R0"].notna()]
                b, lo, hi = boot_dif(gg, "R0", R)
                b2, lo2, hi2 = boot_dif(gg, "R0", R, campo="ACx")
                print("    %s-R0  A1 %+.3f [%+.3f, %+.3f]   ACx %+.3f [%+.3f, %+.3f]"
                      % (R, b, lo, hi, b2, lo2, hi2))
    return d


if __name__ == "__main__":
    main()
