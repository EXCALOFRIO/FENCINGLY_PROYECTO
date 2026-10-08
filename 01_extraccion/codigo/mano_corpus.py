"""La mano armada decidida POR TIRADOR y no por clip, que es lo que de verdad es.

## LA IDEA, Y POR QUE ES GRATIS

**Un tirador no cambia de mano entre un asalto y el siguiente.** La mano armada es una propiedad
de la persona, no del clip. Pero la tuberia la decide clip a clip y cada clip vuelve a empezar de
cero, tirando la unica informacion que hace el problema facil: que el mismo tirador aparece muchas
veces.

Agrupar por tirador y decidir por mayoria es ganancia **por construccion**: si el acierto por clip
es `p > 0,5`, la mayoria sobre `n` clips independientes tiende a 1 cuando `n` crece. No hace falta
modelo nuevo, ni GPU, ni anotar nada. Solo el catalogo, que ya da los dos nombres de cada clip.

## LO QUE HAY QUE RESPETAR, Y NO ES OBVIO

**Los clips del mismo tirador NO son independientes.** Si el detector falla por una razon
sistematica -- una sede con mala camara, un tirador que se pone de perfil -- falla igual en todos
sus clips y la mayoria no arregla nada: la consagra. Por eso esto se MIDE contra la ficha de la
FIE en vez de darse por bueno, y por eso se publica tambien **la fraccion de votos**: un tirador
decidido 7-a-0 no vale lo mismo que uno decidido 4-a-3, y quien consuma esto tiene que poder
distinguirlos.

**Y la ficha manda donde existe.** En el 73 % de los clips anotados los dos nombres resuelven
contra la FIE. Para esos, la mano armada **no hay que detectarla: se consulta**. Eso no es hacer
trampa, es que una ficha federativa es mejor dato que una inferencia de pixeles. Lo que hay que
respetar es la disciplina del proyecto para las etiquetas: la columna **`origen_mano`** con `fie`,
`modelo` o `voto`, **nunca fusionadas**, por la misma razon que `origen_etiqueta` -- el dia que se
entrene el modelo final hay que poder pesarlas distinto o quedarse solo con las de ficha.

    python mano_corpus.py            construye 03_datos/mano_por_tirador.csv
    python mano_corpus.py --tabla    lo imprime
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import catalogo as CA  # noqa: E402
import fie as FIE  # noqa: E402
import rutas as RU  # noqa: E402

SALIDA = RU.DATOS / "mano_por_tirador.csv"

COLUMNAS = ["nombre", "mano", "origen_mano", "n_clips", "n_decididos", "frac_voto",
            "mano_fie", "mano_voto", "coincide", "peldanos"]

MIN_VOTOS = 2
"""Cuantos clips decididos hacen falta para dar una mano por voto. Con uno solo, "la mayoria" es
el propio clip y la columna `frac_voto` valdria 1,000 sin que eso signifique nada."""


def por_clip(clips=None, trabajo=None, verboso=True):
    """QUE DEVUELVE: [(clip, lado, nombre, mano, peldano)] para los clips con poses extraidas.

    Se salta en silencio lo que no se puede leer y lo CUENTA, porque un npz corrupto no es un
    fallo del criterio de la mano y mezclarlos daria un denominador equivocado.
    """
    import cono_lib as CL
    trabajo = pathlib.Path(trabajo or RU.TRABAJO)
    df = CA.manifiesto().set_index("clip_id")
    if clips is None:
        clips = sorted(os.path.basename(p)[6:-4]
                       for p in glob.glob(str(trabajo / "poses_*.npz")))
    out, roto, sin_cat = [], 0, 0
    for c in clips:
        if c not in df.index:
            sin_cat += 1
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
            if not nom:
                continue
            try:
                mano, d = CL.mano_armada(KP, SC, B, W, H, lado)
            except Exception:
                continue
            if mano in ("A", "B"):
                out.append((c, lado, nom, mano, d.get("por", "?")))
    if verboso:
        print("clips leidos %d, npz ilegibles %d, no estan en el catalogo %d -> %d decisiones"
              % (len(clips), roto, sin_cat, len(out)))
    return out


def agrupa(filas, a_es_diestra=True):
    """QUE DEVUELVE: dict nombre -> {mano_voto, n, frac, peldanos}.

    `a_es_diestra` es el sentido del mapeo A/B contra R/L, que **se mide** en
    `verifica_mano_fie.py` y no se supone: alli salio A = diestra con 0,920 de acuerdo contra
    0,080 de la hipotesis contraria.
    """
    por = collections.defaultdict(list)
    pel = collections.defaultdict(collections.Counter)
    for _c, _l, nom, mano, peldano in filas:
        por[nom].append(mano)
        pel[nom][peldano] += 1
    out = {}
    for nom, ms in por.items():
        cnt = collections.Counter(ms)
        gana, n_gana = cnt.most_common(1)[0]
        empate = len(cnt) > 1 and cnt.most_common(2)[1][1] == n_gana
        out[nom] = {
            "mano_voto": "" if empate else gana,
            "n": len(ms),
            "frac": n_gana / len(ms),
            "peldanos": ",".join("%s:%d" % kv for kv in pel[nom].most_common(3)),
            "mano_rl": "" if empate else ("R" if (gana == "A") == a_es_diestra else "L"),
        }
    return out


def construye(salida=None, clips=None, a_es_diestra=True, verboso=True):
    """QUE HACE: escribe `mano_por_tirador.csv` juntando el voto del modelo y la ficha de la FIE.

    LA FICHA MANDA DONDE EXISTE, y queda escrito en `origen_mano`. Donde no existe se usa el voto
    si tiene al menos `MIN_VOTOS` clips decididos; si tampoco, la fila sale con `mano` vacia --
    **vacia y no adivinada**, que es la diferencia entre "no lo se" y "he elegido una".
    """
    filas = por_clip(clips, verboso=verboso)
    votos = agrupa(filas, a_es_diestra)
    mf = FIE.manos()
    salida = pathlib.Path(salida or SALIDA)
    salida.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for nom in sorted(set(votos) | set(mf)):
        v = votos.get(nom, {})
        h = mf.get(nom, "")
        voto_rl = v.get("mano_rl", "")
        if h:
            mano, origen = h, "fie"
        elif voto_rl and v.get("n", 0) >= MIN_VOTOS:
            mano, origen = voto_rl, "voto"
        else:
            mano, origen = "", ""
        out.append({
            "nombre": nom, "mano": mano, "origen_mano": origen,
            "n_clips": v.get("n", 0), "n_decididos": v.get("n", 0),
            "frac_voto": round(v.get("frac", 0.0), 3) if v else "",
            "mano_fie": h, "mano_voto": voto_rl,
            "coincide": ("" if not (h and voto_rl) else ("1" if h == voto_rl else "0")),
            "peldanos": v.get("peldanos", ""),
        })
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(out)
    if verboso:
        con = [f for f in out if f["coincide"] != ""]
        ok = sum(1 for f in con if f["coincide"] == "1")
        print("%s: %d tiradores (%d con ficha, %d con voto)"
              % (salida.name, len(out), sum(1 for f in out if f["mano_fie"]),
                 sum(1 for f in out if f["mano_voto"])))
        if con:
            print("   con las DOS: %d; el voto coincide con la ficha en %d (%.3f)"
                  % (len(con), ok, ok / len(con)))
    return out


def compara_clip_vs_tirador(a_es_diestra=True, verboso=True):
    """QUE MIDE: el acierto contra la ficha, decidiendo POR CLIP y decidiendo POR TIRADOR.

    Es la comparacion que decide si la mejora existe. Las dos se miden sobre **los mismos
    tiradores** -- los que tienen ficha y al menos un clip decidido -- porque comparar dos cifras
    con denominadores distintos es exactamente el error que este proyecto arrastra.
    """
    filas = por_clip(verboso=verboso)
    mf = FIE.manos()
    esp = {}
    for nom, h in mf.items():
        esp[nom] = "A" if (h == "R") == a_es_diestra else "B"
    utiles = [f for f in filas if f[2] in esp]
    nombres = {f[2] for f in utiles}
    ok_clip = sum(1 for f in utiles if f[3] == esp[f[2]])
    votos = agrupa(utiles, a_es_diestra)
    ok_tir = sum(1 for n in nombres
                 if votos[n]["mano_voto"] and votos[n]["mano_voto"] == esp[n])
    n_dec = sum(1 for n in nombres if votos[n]["mano_voto"])
    por_n = collections.defaultdict(lambda: [0, 0])
    for n in nombres:
        v = votos[n]
        if not v["mano_voto"]:
            continue
        k = min(v["n"], 5)
        por_n[k][0] += int(v["mano_voto"] == esp[n])
        por_n[k][1] += 1
    return {
        "decisiones_clip": len(utiles), "acierto_clip": ok_clip / max(len(utiles), 1),
        "tiradores": len(nombres), "decididos": n_dec,
        "acierto_tirador": ok_tir / max(n_dec, 1),
        "por_n_clips": {k: (v[0] / max(v[1], 1), v[1]) for k, v in sorted(por_n.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", action="store_true")
    a = ap.parse_args()
    if a.tabla and SALIDA.is_file():
        with open(SALIDA, encoding="utf-8") as fh:
            for i, f in enumerate(csv.DictReader(fh)):
                if i < 25:
                    print("%-34s %-4s %-8s n=%-3s voto=%-6s fie=%-4s %s"
                          % (f["nombre"][:34], f["mano"], f["origen_mano"], f["n_clips"],
                             f["frac_voto"], f["mano_fie"], f["peldanos"][:28]))
        return 0
    construye()
    return 0


if __name__ == "__main__":
    sys.exit(main())
