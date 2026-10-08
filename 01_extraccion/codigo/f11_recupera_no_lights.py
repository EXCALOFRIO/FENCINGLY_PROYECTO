"""La prueba correcta del detector nuevo: los clips que el VIEJO pierde, ¿los recupera?

POR QUE LA MUESTRA B NO ERA LA PRUEBA BUENA. La muestra B se sorteo entre clips **que tienen NPZ**,
o sea clips en los que el detector viejo YA habia encontrado luz. Medir ahi mi detector responde a
"¿soy mejor que el viejo donde el viejo funciona?", que no es la pregunta. La pregunta que decide si
el trabajo sirve es la contraria: **en los 2.038 clips donde el viejo se queda a cero, encuentro
luz?** Si la respuesta es si, los dos juntos cubren mas que cualquiera de los dos, y el arreglo
dirigido que pide el coordinador tiene un numero.

LA MUESTRA. Los clips con `status = "no_lights"` salen de los `index.jsonl`, que es una lectura de
indice y no cuesta CPU. De ahi se sortean, con SEMILLA 20260913:

  25 clips de `tokyofoil2022`        familia C, la sede ROTA (36,78 % perdido, ~527 clips)
  25 clips de `parisfoil2022`        familia E (13,35 %)
  25 clips de `stpetersburgfoil2019` familia B (12,71 %)
  25 clips de la familia A entera    el CONTROL

EL CONTROL ES LO QUE HACE QUE ESTO MIDA ALGO. Si mi detector recupera el 60 % en la familia C y
tambien el 60 % en la familia A, entonces no estoy arreglando una familia: estoy recuperando clips
que se pierden por otra razon (repeticion, corte de plano, clip mal cortado) y el marco de
"familias" seria falso. Sin el control, un 60 % en la familia C no se puede interpretar.

CRITERIO DECLARADO ANTES DE EJECUTAR. Un clip cuenta como RECUPERADO si el detector da al menos una
luz de COLOR (roja o verde) en el grupo del tocado. No se exige acierto, porque en estos clips no hay
verdad de referencia de ninguna clase: lo que se mide es DETECCION, que es exactamente la magnitud
que el viejo puso a cero.

  El arreglo de una familia SIRVE si recupera >= 40 % de sus clips perdidos
  Y al menos 15 puntos por encima del control de la familia A.

El 40 % no se elige por conveniencia: recuperar 40 % de los 527 clips perdidos de `tokyofoil2022`
son 211 clips, que es del orden del corpus entero de `tbilisifoil2023` (222). Por debajo de eso el
arreglo no cambia el tamano de nada.

ECONOMIA. 100 clips, un paso de decodificacion por clip, sin GPU y sin modelos. El usuario tiene el
PC cargado: no se toca nada de los 25.139.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import random
import sys

import pandas as pd

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
SEMILLA = 20260913
N_POR_SEDE = 25
DIANAS = ["tokyofoil2022", "parisfoil2022", "stpetersburgfoil2019"]
FAMILIA_A = [
    "anaheimfoil2018", "anaheimfoil2019", "belgradefoil2022", "bonnfoil2019", "bonnfoil2022",
    "eurochamps2022", "eurosfoil2023", "incheonfoil2022", "katowicefoil2019", "qatarfoil2021",
    "serbiafoil2022", "tauberbischofsheimfoil2022", "tbilisifoil2023", "turinfoil2020",
    "turinfoil2023", "worldchamps2022",
]


def lee_no_lights() -> dict[str, list[str]]:
    """sede -> clips con `status = "no_lights"`, deduplicados y sin los que otra corrida saco bien."""
    por_clip: dict[str, list[str]] = {}
    sede_de: dict[str, str] = {}
    for ruta in sorted(glob.glob(os.path.join(
            F4.BASE_VIEJA if hasattr(F4, "BASE_VIEJA") else
            r"C:\Users\Alejandro\AppData\Local\Temp\claude"
            r"\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565",
            "localrun", "runs*", "*", "index.jsonl"))):
        with open(ruta, encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if not linea:
                    continue
                o = json.loads(linea)
                cid = o.get("clip_id")
                if cid is None:
                    continue
                por_clip.setdefault(cid, []).append(o.get("status"))
                if o.get("tournament_id"):
                    sede_de[cid] = o["tournament_id"]
    fuera: dict[str, list[str]] = collections.defaultdict(list)
    for cid, estados in por_clip.items():
        if "ok" in estados:
            continue
        if "no_lights" in estados and cid in sede_de:
            fuera[sede_de[cid]].append(cid)
    return {k: sorted(v) for k, v in fuera.items()}


def indice_mp4() -> dict[str, str]:
    mapa = {}
    with open(os.path.join(F4.SCRATCH, "indice_mp4.tsv"), encoding="utf-8-sig") as fh:
        for linea in fh:
            linea = linea.rstrip("\r\n")
            if linea:
                stem, camino = linea.split("\t", 1)
                mapa.setdefault(stem, camino)
    return mapa


def corre(clips: list[tuple[str, str]], mapa: dict[str, str], etiqueta: str) -> pd.DataFrame:
    filas = []
    for sede, clip in clips:
        ruta = mapa.get(clip)
        if not ruta:
            filas.append(dict(grupo=etiqueta, sede=sede, clip=clip, estado="sin_mp4",
                           luces="", n_color=0, recuperado=False))
            continue
        try:
            res = LU.detecta(ruta)
            lz = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]),
                                                     res["fps"]))
        except Exception as exc:
            filas.append(dict(grupo=etiqueta, sede=sede, clip=clip, estado=f"error:{exc}"[:40],
                              luces="", n_color=0, recuperado=False))
            continue
        color = [d for d in lz if d["canal"] != "blanca"]
        filas.append(dict(
            grupo=etiqueta, sede=sede, clip=clip, estado="ok",
            luces="+".join(f"{d['canal'][0].upper()}{d['lado']}" for d in lz) or "-",
            n_color=len(color), recuperado=len(color) > 0))
    return pd.DataFrame(filas)


def main() -> int:
    nl = lee_no_lights()
    mapa = indice_mp4()
    print("clips no_lights por sede (los que el detector viejo perdio):")
    for s in sorted(nl, key=lambda k: -len(nl[k])):
        con = sum(1 for c in nl[s] if c in mapa)
        print(f"  {s:28s} {len(nl[s]):4d}   con mp4 localizado: {con}")

    rnd = random.Random(SEMILLA)
    tandas = []
    for sede in DIANAS:
        cands = [c for c in nl.get(sede, []) if c in mapa]
        elegidos = rnd.sample(cands, min(N_POR_SEDE, len(cands)))
        tandas.append(([(sede, c) for c in elegidos], sede))
    pool_a = [(s, c) for s in FAMILIA_A for c in nl.get(s, []) if c in mapa]
    tandas.append((rnd.sample(pool_a, min(N_POR_SEDE, len(pool_a))), "CONTROL_familia_A"))

    trozos = []
    for clips, etiqueta in tandas:
        print(f"\n>>> {etiqueta}: {len(clips)} clips")
        d = corre(clips, mapa, etiqueta)
        trozos.append(d)
        print(f"    recuperados: {d['recuperado'].sum()}/{len(d)} = "
              f"{100.0 * d['recuperado'].mean():.1f} %")
        print(f"    reparto de lo detectado: "
              f"{dict(collections.Counter(d['luces']).most_common(6))}")

    d = pd.concat(trozos, ignore_index=True)
    print("\n--- RESUMEN, criterio declarado: >= 40 % y >= 15 puntos sobre el control ---")
    control = d[d["grupo"] == "CONTROL_familia_A"]["recuperado"].mean() * 100.0
    print(f"  CONTROL familia A: {control:.1f} %")
    for sede in DIANAS:
        s = d[d["grupo"] == sede]
        pct = 100.0 * s["recuperado"].mean()
        pasa = pct >= 40.0 and (pct - control) >= 15.0
        print(f"  {sede:28s} {s['recuperado'].sum():2d}/{len(s):2d} = {pct:5.1f} %   "
              f"margen sobre el control {pct - control:+5.1f}   {'PASA' if pasa else 'NO PASA'}")
    d.to_csv(os.path.join(SALIDA, "recupera_no_lights.csv"), index=False)
    print(f"\ncsv: {os.path.join(SALIDA, 'recupera_no_lights.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
