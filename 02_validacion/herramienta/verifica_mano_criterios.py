"""Reabre los criterios de la mano armada que se retiraron con muestras de 27, ahora con la FIE.

## POR QUE HAY QUE REABRIRLOS

Las decisiones vigentes de `cono_lib` se tomaron contra **27 tiradores anotados a ojo**, y algunas
con frases que con ese n no significan nada:

| criterio | lo que se midio entonces | n | que se hizo |
|---|---|---:|---|
| `a_voto` (voto por cuadro) | 0,815 -- "+2 casos, dentro del ruido" | 27 | no se adopto |
| `a_dist2d` (distancia euclidea) | declarado, sin cifra firme | 27 | no se adopto |
| `nariz_dir` | 0,932 sobre 59 | 59 | `USA_NARIZ_SI_PIE_FLOJO = False` |
| `mano_hombro` | -- | -- | `USA_HOMBRO_SI_PIE_FLOJO = False` |
| guante blanco / cazoleta | 0,444 / 0,500 | 27 | retirados |

"Dentro del ruido" con n=27 y "dentro del ruido" con n=967 **no son la misma frase**. Alguno de
estos puede estar bien y haberse retirado por falta de muestra. Con la ficha federativa hay con
que decidirlo.

## LAS TRES DISCIPLINAS QUE ESTO RESPETA

**1. Acierto y cobertura en columnas distintas.** Un criterio que decide 12 de 19 sin fallar y otro
que decide 19 de 19 con 0,895 no se comparan por el primer numero. Se da tambien el `acierto*`,
que cuenta los mudos como fallo, porque es el unico con el que se pueden ordenar.

**2. El azar informado al lado.** El 74,3 % de los tiradores son diestros, asi que contestar
siempre "la diestra" acierta 0,743 sin mirar la imagen. Un criterio por debajo de eso no aporta,
por bonito que sea su porcentaje sobre los casos que responde.

**3. Se agrega por TIRADOR, no por clip.** Dos clips del mismo tirador no son independientes: la
mano es de la persona. Se dan las dos cifras y la de tirador es la que manda.

## Y LO QUE ESTO **NO** HACE

No adopta nada. Mide y ordena. Adoptar un criterio exige ademas un reservado **por tirador** que
no haya participado en la comparacion, y eso se hace cuando haya corpus para partirlo -- hoy
serian unas decenas de tiradores en el reservado y cualquier diferencia caeria dentro del ruido,
que es exactamente el error que se viene a corregir.

    python verifica_mano_criterios.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import fie as FIE  # noqa: E402
import rutas as RU  # noqa: E402

CRITERIOS = [
    ("cascada (lo vigente)", lambda m, d: m),
    ("geometria a (x mas cerca)", lambda m, d: d.get("mano_a")),
    ("a_voto (voto por cuadro)", lambda m, d: d.get("a_voto")),
    ("a_dist2d (euclidea)", lambda m, d: d.get("a_dist2d")),
    ("mayoria_3", lambda m, d: d.get("mayoria_3")),
    ("pie (mano_pie)", lambda m, d: d.get("_pie")),
    ("nariz (mano_nariz)", lambda m, d: d.get("_nariz")),
    ("hombro (mano_hombro)", lambda m, d: d.get("_hombro")),
]
"""Cada criterio como una funcion del dict que `mano_armada` ya devuelve. **No se reimplementa
ninguno**: se leen los que el modulo calcula de todas formas, asi que lo que se mide es lo que
correria de verdad si se adoptase, y no una copia que puede divergir."""


def recoge(trabajo=None, verboso=True):
    """QUE DEVUELVE: [(nombre_tirador, verdad_AB, {criterio: AB o None})]."""
    import cono_lib as CL
    trabajo = pathlib.Path(trabajo or RU.TRABAJO)
    df = CA.manifiesto().set_index("clip_id")
    mf = FIE.manos()
    out, roto = [], 0
    clips = sorted(os.path.basename(p)[6:-4]
                   for p in glob.glob(str(trabajo / "poses_*.npz")))
    for c in clips:
        if c not in df.index:
            continue
        try:
            with np.load(trabajo / ("poses_%s.npz" % c), allow_pickle=True) as z:
                if "KP" not in z.files:
                    roto += 1
                    continue
                KP, SC, B = z["KP"], z["SC"], z["B"]
                W, H = int(z["W"]), int(z["H"])
        except Exception:
            roto += 1
            continue
        r = df.loc[c]
        for lado in (0, 1):
            nom = FIE.norm(r["fotl_name" if lado == 0 else "fotr_name"])
            h = mf.get(nom)
            if not h:
                continue
            try:
                mano, d = CL.mano_armada(KP, SC, B, W, H, lado)
                d = dict(d)
                d["_pie"] = CL.mano_pie(KP, SC, lado)[0]
                d["_nariz"] = CL.mano_nariz(KP, SC, lado)[0]
                d["_hombro"] = CL.mano_hombro(KP, SC, lado)[0]
            except Exception:
                continue
            verdad = "A" if h == "R" else "B"
            out.append((nom, verdad, mano, d))
    if verboso:
        print("clips %d (npz ilegibles %d) -> %d decisiones con ficha"
              % (len(clips), roto, len(out)))
    return out


def evalua(filas, por_tirador=False):
    """QUE DEVUELVE: {criterio: (acierto, cobertura, acierto_contando_mudos, n)}."""
    res = {}
    for nombre, fn in CRITERIOS:
        if por_tirador:
            votos = collections.defaultdict(list)
            verdad = {}
            for nom, v, mano, d in filas:
                verdad[nom] = v
                x = fn(mano, d)
                if x in ("A", "B"):
                    votos[nom].append(x)
            ok = resp = 0
            tot = len(verdad)
            for nom, vs in votos.items():
                cnt = collections.Counter(vs)
                g, ng = cnt.most_common(1)[0]
                if len(cnt) > 1 and cnt.most_common(2)[1][1] == ng:
                    continue
                resp += 1
                ok += int(g == verdad[nom])
        else:
            ok = resp = 0
            tot = len(filas)
            for nom, v, mano, d in filas:
                x = fn(mano, d)
                if x not in ("A", "B"):
                    continue
                resp += 1
                ok += int(x == v)
        res[nombre] = (ok / max(resp, 1), resp / max(tot, 1), ok / max(tot, 1), resp)
    return res


def tabla(res, titulo, base):
    print()
    print(titulo)
    print("%-30s %8s %10s %10s %7s" % ("criterio", "acierto", "cobertura", "acierto*", "n"))
    print("-" * 70)
    for nombre, _fn in CRITERIOS:
        a, cob, ac, n = res[nombre]
        marca = ""
        if ac > base:
            marca = "  +%.3f sobre el azar informado" % (ac - base)
        elif cob > 0.9:
            marca = "  POR DEBAJO del azar informado"
        print("%-30s %8.3f %10.3f %10.3f %7d%s" % (nombre, a, cob, ac, n, marca))
    print("-" * 70)
    print("%-30s %8.3f %10.3f %10.3f %7s" % ("azar informado (siempre R)", base, 1.0, base, "-"))


def pareado(filas, a, b):
    """QUE HACE: compara DOS criterios sobre los MISMOS tiradores, contando en que discrepan.

    POR QUE PAREADO Y NO RESTANDO DOS PORCENTAJES. La pregunta "es mejor `a_voto` que la cascada"
    no se contesta restando 0,897 - 0,889: eso da +0,8 pp, que sobre 253 tiradores son **dos
    personas**, y dos personas no distinguen nada. Lo que la contesta es en cuantos tiradores uno
    acierta y el otro falla, en cada direccion -- que es la prueba de McNemar, y ademas dice
    *quienes* son, que es lo que se puede ir a mirar.
    """
    fa = dict(CRITERIOS)[a]
    fb = dict(CRITERIOS)[b]
    verdad, va, vb = {}, collections.defaultdict(list), collections.defaultdict(list)
    for nom, v, mano, d in filas:
        verdad[nom] = v
        for f, acc in ((fa, va), (fb, vb)):
            x = f(mano, d)
            if x in ("A", "B"):
                acc[nom].append(x)

    def decide(acc, nom):
        vs = acc.get(nom, [])
        if not vs:
            return None
        c = collections.Counter(vs)
        g, ng = c.most_common(1)[0]
        if len(c) > 1 and c.most_common(2)[1][1] == ng:
            return None
        return g

    solo_a = solo_b = ambos_ok = ambos_mal = 0
    for nom, v in verdad.items():
        xa, xb = decide(va, nom), decide(vb, nom)
        if xa is None or xb is None:
            continue
        oa, ob = xa == v, xb == v
        if oa and ob:
            ambos_ok += 1
        elif oa:
            solo_a += 1
        elif ob:
            solo_b += 1
        else:
            ambos_mal += 1
    n_dis = solo_a + solo_b
    p = None
    if n_dis:
        from math import comb
        k = min(solo_a, solo_b)
        p = min(1.0, 2 * sum(comb(n_dis, i) for i in range(k + 1)) / (2.0 ** n_dis))
    return {"solo_a": solo_a, "solo_b": solo_b, "ambos_ok": ambos_ok,
            "ambos_mal": ambos_mal, "discrepan": n_dis, "p": p}


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    print("LOS CRITERIOS DE LA MANO ARMADA, REABIERTOS CONTRA LA FICHA FEDERATIVA")
    print("=" * 84)
    mf = FIE.manos()
    if not mf:
        print("falta mano_fie.csv: python 01_extraccion/codigo/fie.py --manos")
        return 2
    base, n = FIE.base_diestros()
    print("azar informado: %.3f  (de %d tiradores con mano declarada)" % (base, n))
    filas = recoge()
    if len(filas) < 30:
        print("menos de 30 decisiones: no se da ninguna cifra, falta extraccion.")
        return 0
    tabla(evalua(filas, False), "POR CLIP  (cada decision cuenta)", base)
    tabla(evalua(filas, True), "POR TIRADOR  (mayoria de sus clips; la que manda)", base)
    print()
    print("* 'acierto' es sobre los casos que el criterio RESPONDE. 'acierto*' cuenta los mudos")
    print("  como fallo, y es el unico con el que se pueden ORDENAR criterios de cobertura")
    print("  distinta -- comparar por el primero premia al que casi nunca contesta.")
    print()
    print("EL CAREO PAREADO: los candidatos contra LO VIGENTE, sobre los mismos tiradores")
    print("%-30s %7s %7s %9s %8s" % ("candidato", "gana", "pierde", "discrepan", "p"))
    print("-" * 66)
    for cand in ("a_voto (voto por cuadro)", "a_dist2d (euclidea)", "mayoria_3",
                 "nariz (mano_nariz)", "geometria a (x mas cerca)"):
        r = pareado(filas, cand, "cascada (lo vigente)")
        print("%-30s %7d %7d %9d %8s"
              % (cand, r["solo_a"], r["solo_b"], r["discrepan"],
                 "%.3f" % r["p"] if r["p"] is not None else "-"))
    print("-" * 66)
    print("'gana' = tiradores donde el candidato acierta y la cascada falla; 'pierde' al reves.")
    print("La p es de McNemar de dos colas. Con p alta, la diferencia de porcentajes es RUIDO por")
    print("mucho que el candidato salga arriba en la tabla anterior.")
    print()
    print("NO SE ADOPTA NADA AQUI. Adoptar exige un reservado POR TIRADOR que no haya participado")
    print("en esta comparacion; con el corpus de hoy serian unas decenas y cualquier diferencia")
    print("caeria dentro del ruido, que es justo el error que esto viene a corregir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
