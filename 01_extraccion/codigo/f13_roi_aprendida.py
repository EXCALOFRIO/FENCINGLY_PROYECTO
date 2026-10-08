"""¿Los clips que el viejo pierde TIENEN luz? La ROI se aprende de los clips que SI funcionan.

DE DONDE VIENE ESTA PREGUNTA. El plan aceptado era arreglar la familia C porque `tokyofoil2022`
pierde el 36,78 % de sus clips. Dos medidas lo han puesto en duda:

  - `f11`: mi detector recupera el 44,0 % de los clips perdidos de la familia A (el CONTROL) y solo
    el 28,0 % de los de `tokyofoil2022`. Recupera IGUAL o MENOS en la sede rota que en las sanas, lo
    que no encaja con "es el diseno del marcador de esa familia".
  - `f12`: el rotulo ESTA presente en el 92-100 % de los clips perdidos, asi que no son repeticiones
    ni planos de grada. Pero al mirarlos, en el ultimo fotograma **no hay ninguna luz**, y varios
    marcadores estan empatados a 0-0 o a 4-4: tienen pinta de acciones que acabaron en alto SIN
    tocado.

Si eso es asi, `status = "no_lights"` no es un fallo: es la respuesta correcta, y los 2.038 clips no
son recuperables porque no hay nada que recuperar. Esa es la hipotesis que mide este script, y si
sale cierta **cambia la prioridad de todo el arreglo**.

EL DISENO, que evita el circulo. La ROI de la luz no se pone a mano ni se copia del detector viejo:
se **APRENDE de los clips de la propia sede que si funcionaron**. Por sede:

  1. APRENDIZAJE  se toman `N_APRENDE` clips con `status = "ok"`, se corre el detector, y se cruza su
                  salida con el canal y el lado que dice el NPZ viejo. La mediana de (cx, cy, fw, fh)
                  de las coincidencias es la ROI de esa sede, medida en sus propios clips.
  2. CONTROL      se aplica esa ROI a OTROS `N_PRUEBA` clips con `status = "ok"`, que no se usaron
                  para aprenderla. Aqui SI hay luz por construccion, asi que esto mide si la lectura
                  de la ROI funciona.
  3. PRUEBA       se aplica la misma ROI a `N_PRUEBA` clips con `status = "no_lights"`.

La lectura de la ROI es deliberadamente tonta: el maximo a lo largo del clip del canal medio en la
ROI, menos su percentil 20. Un solo numero, `salto_roi`.

CRITERIOS DECLARADOS ANTES DE EJECUTAR, en este orden y sin moverlos:

  (i)  la ROI aprendida VALE si `salto_roi >= UMBRAL` en >= 80 % de los clips `ok` del CONTROL.
       Si no llega, la ROI o la lectura estan mal y el resto no se puede interpretar.
  (ii) si (i) vale y la ROI enciende en < 30 % de los clips `no_lights`, entonces esos clips NO
       TIENEN LUZ y `no_lights` es correcto: no hay arreglo que los recupere.
  (iii) si (i) vale y la ROI enciende en >= 60 % de los `no_lights`, entonces la luz esta donde tiene
       que estar y el detector viejo no la vio: el arreglo dirigido si recupera clips, y la cifra de
       recuperables es ese porcentaje por el numero de clips perdidos de la sede.

`UMBRAL` = 25 niveles, declarado: es la mitad del salto mas pequeno medido en una luz real
(`PettyCelebratedGyrfalcon`, 11, es el minimo absoluto; los saltos tipicos van de 28 a 268), asi que
es generoso a proposito para no confundir "no hay luz" con "el umbral era alto".
"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

import luces as LU
import f4_censo_visual as F4
import f11_recupera_no_lights as F11

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
SEMILLA = 20260914
N_APRENDE = 14
N_PRUEBA = 20
UMBRAL = 25.0
GRUPOS = {
    "tokyofoil2022": ["tokyofoil2022"],
    "stpetersburgfoil2019": ["stpetersburgfoil2019"],
    "parisfoil2022": ["parisfoil2022"],
    "CONTROL_familia_A": F11.FAMILIA_A,
}


def canal_y_lado(fila) -> tuple[str, str] | None:
    if pd.notna(fila.get("t_green_R")):
        return ("verde", "R")
    if pd.notna(fila.get("t_red_L")):
        return ("roja", "L")
    return None


def lee_roi(ruta: str, roi: dict, canal: str) -> float:
    """`salto_roi`: maximo temporal menos percentil 20 del canal medio dentro de la ROI."""
    pila, idx, fps, n, W, H = LU.lee_pila(ruta)
    T, h, w = pila.shape[:3]
    s = LU.canales(pila)[canal]
    y0 = int(np.clip((roi["fy"] - LU.BANDA_Y[0]) / (LU.BANDA_Y[1] - LU.BANDA_Y[0]) * h, 0, h - 2))
    y1 = int(np.clip(y0 + max(2, roi["fh"] / (LU.BANDA_Y[1] - LU.BANDA_Y[0]) * h), y0 + 2, h))
    x0 = int(np.clip(roi["fx"] * w, 0, w - 3))
    x1 = int(np.clip(x0 + max(3, roi["fw"] * w), x0 + 3, w))
    serie = s[:, y0:y1, x0:x1].reshape(T, -1).mean(axis=1)
    return float(serie.max() - np.percentile(serie, 20))


def main() -> int:
    viejo = pd.read_csv(os.path.join(F4.SCRATCH, "luces_npz_viejo.csv")).set_index("clip")
    nl = F11.lee_no_lights()
    mapa = F11.indice_mp4()
    rnd = random.Random(SEMILLA)
    resumen = []
    detalle = []

    for etiqueta, sedes in GRUPOS.items():
        oks = [c for c in viejo.index
               if viejo.loc[c, "sede"] in sedes and c in mapa]
        perdidos = [c for s in sedes for c in nl.get(s, []) if c in mapa]
        rnd.shuffle(oks)
        aprende, prueba_ok = oks[:N_APRENDE], oks[N_APRENDE:N_APRENDE + N_PRUEBA]
        prueba_nl = rnd.sample(perdidos, min(N_PRUEBA, len(perdidos)))
        print(f"\n>>> {etiqueta}: aprende={len(aprende)} control_ok={len(prueba_ok)} "
              f"no_lights={len(prueba_nl)}  (perdidos totales {len(perdidos)})")

        cajas: dict[str, list[dict]] = {"roja": [], "verde": []}
        for clip in aprende:
            par = canal_y_lado(viejo.loc[clip])
            if par is None:
                continue
            canal, lado = par
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
                print(f"    ROI {canal:6s} de {len(ds):2d} clips: "
                      f"fx={rois[canal]['fx']:.3f} fy={rois[canal]['fy']:.3f} "
                      f"fw={rois[canal]['fw']:.3f} fh={rois[canal]['fh']:.3f}")
            else:
                print(f"    ROI {canal:6s}: solo {len(ds)} clips, no se aprende")
        if not rois:
            continue

        for grupo, clips in (("control_ok", prueba_ok), ("no_lights", prueba_nl)):
            enciende = 0
            saltos = []
            for clip in clips:
                mejor = 0.0
                mejor_canal = ""
                for canal, roi in rois.items():
                    try:
                        v = lee_roi(mapa[clip], roi, canal)
                    except Exception:
                        continue
                    if v > mejor:
                        mejor, mejor_canal = v, canal
                saltos.append(mejor)
                enciende += mejor >= UMBRAL
                detalle.append(dict(grupo=etiqueta, tipo=grupo, clip=clip,
                                    salto_roi=round(mejor, 1), canal=mejor_canal,
                                    enciende=mejor >= UMBRAL))
            pct = 100.0 * enciende / max(1, len(clips))
            print(f"    {grupo:10s} enciende {enciende}/{len(clips)} = {pct:5.1f} %   "
                  f"salto_roi p50={np.median(saltos):6.1f} p90={np.percentile(saltos, 90):6.1f}")
            resumen.append(dict(grupo=etiqueta, tipo=grupo, n=len(clips), enciende=enciende,
                                pct=round(pct, 1), p50=round(float(np.median(saltos)), 1),
                                perdidos_sede=len(perdidos)))

    d = pd.DataFrame(resumen)
    pd.set_option("display.width", 200)
    print("\n--- RESUMEN ---")
    print(d.to_string(index=False))
    print("\n--- VEREDICTO por grupo, con los criterios declarados ---")
    for etq in GRUPOS:
        s = d[d["grupo"] == etq]
        if len(s) < 2:
            continue
        ok = float(s[s["tipo"] == "control_ok"]["pct"].iloc[0])
        nlp = float(s[s["tipo"] == "no_lights"]["pct"].iloc[0])
        perd = int(s["perdidos_sede"].iloc[0])
        if ok < 80.0:
            ver = f"(i) NO VALE: la ROI solo enciende en el {ok:.1f} % de los ok. No interpretable."
        elif nlp < 30.0:
            ver = (f"(ii) LOS CLIPS NO TIENEN LUZ: enciende en el {nlp:.1f} % de los no_lights. "
                   f"`no_lights` es correcto y NO hay {perd} clips recuperables.")
        elif nlp >= 60.0:
            ver = (f"(iii) LA LUZ ESTA Y EL VIEJO NO LA VIO: {nlp:.1f} % -> "
                   f"~{int(round(perd * nlp / 100)):d} clips recuperables de {perd}.")
        else:
            ver = f"ZONA MUERTA: {nlp:.1f} % de los no_lights. Ni una cosa ni la otra."
        print(f"  {etq:24s} control_ok={ok:5.1f} %   no_lights={nlp:5.1f} %")
        print(f"    -> {ver}")

    pd.DataFrame(detalle).to_csv(os.path.join(SALIDA, "roi_aprendida_detalle.csv"), index=False)
    d.to_csv(os.path.join(SALIDA, "roi_aprendida.csv"), index=False)
    print(f"\ncsv: {os.path.join(SALIDA, 'roi_aprendida.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
