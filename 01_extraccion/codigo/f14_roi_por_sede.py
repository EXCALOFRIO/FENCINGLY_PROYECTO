"""EL ARREGLO QUE SI FUNCIONA: una ROI aprendida por sede, probada en la muestra B que me tumbo.

DE DONDE SALE. `f13_roi_aprendida.py` midio que una ROI aprendida de los propios clips de una sede,
leida de la forma mas tonta posible (maximo temporal del canal medio en la ROI menos su percentil
20), enciende en el 80-90 % de clips `ok` que no se usaron para aprenderla, con `salto_roi` mediano
de 131 a 196 niveles. Mi detector general, que busca la luz en toda la banda inferior, se quedaba en
el 71,4 % de la muestra B. Asi que la ROI aprendida no es un apano: es una hipotesis con una medida
a favor, y aqui se prueba en la muestra que ya demostro tumbar cifras.

TRES COSAS DE UNA VEZ, y por eso este script existe:

  1. Aprende la ROI de las 21 sedes, cada una de sus propios clips, EXCLUYENDO los clips de la
     muestra A y de la muestra B para que la prueba no este contaminada.
  2. Prueba la ROI en los 21 clips de la MUESTRA B, que son clips `ok` (tienen luz por construccion,
     segun el detector viejo) y que ningun ajuste mio ha visto.
  3. Contesta la pregunta del coordinador sobre el "acierto por la razon equivocada": compara la
     `fy` aprendida de cada sede de la familia A con la FRANJA que mira el detector FIE viejo,
     `[y1, y1 + 0.022H]` con `y1 = 0.910`, o sea **[0,910, 0,932]**, y da el solape.

CRITERIOS DECLARADOS ANTES DE EJECUTAR:

  (A) la ROI aprendida SIRVE si enciende en >= 90 % de los 21 clips de la muestra B. Es el mismo
      umbral que mi detector general NO paso (71,4 %), a proposito: si no se compara con el mismo
      numero, no se compara.
  (B) una sede "acierta por la razon equivocada" si el solape entre su ROI aprendida y la franja
      [0,910, 0,932] es < 0,30. Por debajo de 0,30 el detector viejo esta leyendo mayoritariamente
      pixeles que NO son la luz, y su acierto viene de otra parte.

`UMBRAL` = 25 niveles, el mismo de `f13`, sin tocar.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd

import luces as LU
import f4_censo_visual as F4
import f11_recupera_no_lights as F11
import f13_roi_aprendida as F13

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
SEMILLA = 20260915
N_APRENDE = 12
UMBRAL = F13.UMBRAL
FRANJA_FIE = (0.910, 0.932)


def solape(fy: float, fh: float, franja=FRANJA_FIE) -> float:
    """Fraccion de la ROI aprendida que cae dentro de la franja que mira el detector FIE viejo."""
    a0, a1 = fy, fy + fh
    b0, b1 = franja
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-9, a1 - a0)


def main() -> int:
    viejo = pd.read_csv(os.path.join(F4.SCRATCH, "luces_npz_viejo.csv")).set_index("clip")
    mapa = F11.indice_mp4()
    fuera_a = set(pd.read_csv(os.path.join(SALIDA, "verdad_luces_ojo.csv"))["clip"])
    b = pd.read_csv(os.path.join(SALIDA, "muestra_b.csv"))
    fuera_b = set(b["clip"])
    rnd = random.Random(SEMILLA)

    filas_roi, filas_prueba = [], []
    for sede, g in viejo.groupby("sede"):
        if sede == "SIN_SEDE":
            continue
        cands = [c for c in g.index if c in mapa and c not in fuera_a and c not in fuera_b]
        rnd.shuffle(cands)
        cajas: dict[str, list[dict]] = {"roja": [], "verde": []}
        usados = 0
        for clip in cands:
            if usados >= N_APRENDE * 2 or (len(cajas["roja"]) >= N_APRENDE // 2
                                           and len(cajas["verde"]) >= N_APRENDE // 2):
                break
            par = F13.canal_y_lado(viejo.loc[clip])
            if par is None:
                continue
            canal, lado = par
            if len(cajas[canal]) >= N_APRENDE:
                continue
            usados += 1
            try:
                res = LU.detecta(mapa[clip])
            except Exception:
                continue
            lz = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]),
                                                     res["fps"]))
            for d in lz:
                if d["canal"] == canal and d["lado"] == lado:
                    cajas[canal].append(d)
                    break
        rois = {}
        for canal, ds in cajas.items():
            if len(ds) >= 3:
                rois[canal] = {k: float(np.median([d[k] for d in ds]))
                               for k in ("fx", "fy", "fw", "fh")}
                rois[canal]["n"] = len(ds)
        for canal, roi in rois.items():
            filas_roi.append(dict(sede=sede, canal=canal, **{k: round(v, 4) for k, v in roi.items()},
                                  solape_franja_fie=round(solape(roi["fy"], roi["fh"]), 3)))
        if not rois:
            print(f"AVISO {sede}: no se aprende ROI")
            continue

        clip_b = b.loc[b["sede"] == sede, "clip"]
        if not len(clip_b):
            continue
        clip_b = clip_b.iloc[0]
        mejor, mejor_canal = 0.0, ""
        for canal, roi in rois.items():
            try:
                v = F13.lee_roi(mapa[clip_b], roi, canal)
            except Exception:
                continue
            if v > mejor:
                mejor, mejor_canal = v, canal
        filas_prueba.append(dict(sede=sede, clip=clip_b, salto_roi=round(mejor, 1),
                                 canal=mejor_canal, enciende=mejor >= UMBRAL,
                                 general_detecta=bool(b.loc[b["clip"] == clip_b,
                                                            "detecta"].iloc[0])))

    r = pd.DataFrame(filas_roi)
    p = pd.DataFrame(filas_prueba)
    pd.set_option("display.width", 220)
    print("--- ROI aprendida por sede (de sus propios clips, excluidas las muestras A y B) ---")
    print(r.to_string(index=False))

    print("\n--- (A) PRUEBA en los 21 clips de la MUESTRA B ---")
    print(p.to_string(index=False))
    print(f"\n  ROI aprendida   enciende {p['enciende'].sum()}/{len(p)} = "
          f"{100.0 * p['enciende'].mean():.1f} %   (criterio declarado: >= 90 %)")
    print(f"  detector general detecta {p['general_detecta'].sum()}/{len(p)} = "
          f"{100.0 * p['general_detecta'].mean():.1f} %   (el que NO paso)")
    print(f"  VEREDICTO (A): {'PASA' if p['enciende'].mean() >= 0.90 else 'NO PASA'}")
    print(f"  sedes donde la ROI falla: "
          f"{', '.join(p.loc[~p['enciende'], 'sede'].tolist()) or 'ninguna'}")

    print("\n--- (B) el 'acierto por la razon equivocada': solape con la franja FIE [0,910, 0,932] ---")
    fam_a = r[r["sede"].isin(F11.FAMILIA_A)].sort_values("solape_franja_fie")
    print(fam_a[["sede", "canal", "fy", "fh", "solape_franja_fie"]].to_string(index=False))
    malas = fam_a[fam_a["solape_franja_fie"] < 0.30]
    print(f"\n  sedes-canal con solape < 0,30 (lee pixeles que NO son la luz): {len(malas)} de {len(fam_a)}")
    for _, x in malas.iterrows():
        print(f"    {x['sede']:28s} {x['canal']:6s} ROI y {x['fy']:.3f}-{x['fy'] + x['fh']:.3f} "
              f"vs franja 0,910-0,932  solape {x['solape_franja_fie']:.2f}")

    r.to_csv(os.path.join(SALIDA, "roi_por_sede.csv"), index=False)
    p.to_csv(os.path.join(SALIDA, "roi_por_sede_muestra_b.csv"), index=False)
    print(f"\ncsv: {os.path.join(SALIDA, 'roi_por_sede.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
