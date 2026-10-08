"""MUESTRA B: un SEGUNDO clip por sede, independiente, para ver si las cifras de la muestra A aguantan.

POR QUE EXISTE. La muestra A son 21 clips, uno por sede, y sobre ella se ajustaron cinco cosas del
detector. Un acierto medido en la misma muestra donde se ajusto no es una medida de generalizacion,
asi que hace falta una muestra que no haya visto nadie. Esta se sortea con SEMILLA 20260911 entre los
clips de cada sede que **tienen NPZ viejo** y **no son el clip de la muestra A**.

LO QUE SE PUEDE MEDIR AQUI Y LO QUE NO. En la muestra B **no hay anotacion a ojo**, asi que NO se
puede medir acierto. Lo que se mide es:

  DETECCION      en cuantos clips el detector encuentra al menos una luz de COLOR. El corpus es todo
                 de clips de tocado, asi que un clip sin ninguna luz de color es un fallo de
                 deteccion o un clip mal cortado. Es el mismo tipo de cifra que el `no_lights` del
                 detector viejo, y por tanto comparable con el.
  ACUERDO        en cuantos clips el conjunto (canal, lado) coincide con el del NPZ viejo. El acuerdo
                 NO es acierto: los dos pueden estar mal. Pero un DESACUERDO localiza donde mirar, y
                 en la muestra A el desacuerdo aparecio exactamente en los dos clips donde el viejo
                 estaba mal.
  DESFASE        distribucion de |cuadro_mio - cuadro_viejo| en los clips donde coinciden en canal y
                 lado. Con los dos detectores de acuerdo en QUE luz es, la diferencia de instante
                 acota el error de los dos juntos.

CRITERIO DECLARADO ANTES DE EJECUTAR, y el que decide si el `RING_STD_MAX` movido de 12,0 a 22,0 en
la muestra A fue un ajuste legitimo o un apano:

  deteccion de al menos una luz de color en >= 90 % de los 21 clips de la muestra B,
  y ninguna sede a 0.

Si no pasa, se informa y el umbral no se vuelve a mover.
"""
from __future__ import annotations

import os
import random
import sys

import pandas as pd

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
SEMILLA = 20260911


def main() -> int:
    viejo = pd.read_csv(os.path.join(F4.SCRATCH, "luces_npz_viejo.csv"))
    viejo = viejo[viejo["sede"] != "SIN_SEDE"]
    ya = set(pd.read_csv(os.path.join(SALIDA, "verdad_luces_ojo.csv"))["clip"])
    indice = {}
    with open(os.path.join(F4.SCRATCH, "indice_mp4.tsv"), encoding="utf-8-sig") as fh:
        for linea in fh:
            linea = linea.rstrip("\r\n")
            if linea:
                stem, camino = linea.split("\t", 1)
                indice.setdefault(stem, camino)

    rnd = random.Random(SEMILLA)
    elegidos = []
    for sede, g in viejo.groupby("sede"):
        cands = sorted(c for c in g["clip"] if c not in ya and c in indice)
        if not cands:
            print(f"AVISO: {sede} sin candidato")
            continue
        elegidos.append((sede, rnd.choice(cands)))
    print(f"muestra B: {len(elegidos)} clips, semilla {SEMILLA}")

    vj = viejo.set_index("clip")
    filas = []
    for sede, clip in elegidos:
        res = LU.detecta(indice[clip])
        lz = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]), res["fps"]))
        mio = {(d["canal"], d["lado"]): d for d in lz}
        f = vj.loc[clip]
        vieja = {}
        for col, par in (("t_red_L", ("roja", "L")), ("t_green_R", ("verde", "R")),
                         ("t_white_L", ("blanca", "L")), ("t_white_R", ("blanca", "R"))):
            if pd.notna(f.get(col)):
                vieja[par] = float(f[col]) * res["fps"]
        col_mio = {k for k in mio if k[0] != "blanca"}
        col_vjo = {k for k in vieja if k[0] != "blanca"}
        comunes = col_mio & col_vjo
        desfase = (min(abs(mio[k]["cuadro_encendido"] - vieja[k]) for k in comunes)
                   if comunes else None)
        filas.append(dict(
            sede=sede, clip=clip,
            mio="+".join(sorted(f"{a[0].upper()}{b}" for a, b in mio)) or "-",
            viejo="+".join(sorted(f"{a[0].upper()}{b}" for a, b in vieja)) or "-",
            n_color_mio=len(col_mio), n_color_viejo=len(col_vjo),
            detecta=len(col_mio) > 0,
            acuerdo_color=col_mio == col_vjo,
            desfase_cuadros=round(desfase, 1) if desfase is not None else None,
        ))

    d = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print("\n--- MUESTRA B, por sede ---")
    print(d.to_string(index=False))
    print(f"\ndetecta al menos una luz de color: {d['detecta'].sum()}/{len(d)} "
          f"= {100.0 * d['detecta'].mean():.1f} %   (criterio declarado: >= 90 %)")
    print(f"sedes a CERO: {(~d['detecta']).sum()}   "
          f"{', '.join(d.loc[~d['detecta'], 'sede'].tolist())}")
    print(f"acuerdo con el detector viejo en los canales de color: {d['acuerdo_color'].sum()}/{len(d)} "
          f"= {100.0 * d['acuerdo_color'].mean():.1f} %")
    des = d["desfase_cuadros"].dropna()
    if len(des):
        print(f"desfase |mio - viejo| en cuadros, n={len(des)}: "
              f"p50={des.median():.1f}  p90={des.quantile(0.9):.1f}  max={des.max():.1f}  "
              f"<=4 cuadros en {100.0 * (des <= 4).mean():.1f} %")
    pasa = d["detecta"].mean() >= 0.90 and (~d["detecta"]).sum() == 0
    print(f"\nVEREDICTO de la muestra B: {'PASA' if pasa else 'NO PASA'}")
    d.to_csv(os.path.join(SALIDA, "muestra_b.csv"), index=False)
    print(f"csv: {os.path.join(SALIDA, 'muestra_b.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
