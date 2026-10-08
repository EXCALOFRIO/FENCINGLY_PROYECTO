"""El SUSTITUTO barato del contraataque, medido contra la etiqueta real, y el atajo de las luces.

DE DONDE VIENE. El usuario quiere (a) quedarse con los clips de DOS luces, porque con una sola el
tocado es trivial, y (b) usar los de UNA luz que NO sean contraataque como si fueran de dos, lo que
multiplicaria el corpus. Para (b) hace falta separar los contraataques, y el detector de prioridad
del proyecto NO funciona (69,7 % contra un 73,8 % de no contestar nunca). Asi que la pregunta
accionable es: **hay un sustituto barato del contraataque, y se puede MEDIR?**

EL SUSTITUTO QUE SE PROPONE, declarado antes de medirlo. En un contraataque el que toca NO tiene la
prioridad: no es el que ataca, y tipicamente **no esta avanzando** mientras el otro si. Eso es
geometria y esta en los NPZ viejos sin decodificar un solo video: `boxes (T,2,4)` da la caja de cada
tirador y `pan (T,)` el paneo de camara, con la convencion escrita en el propio `meta_json`
(`x_piste = x_image + pan`). Asi que para cada clip se mide, en los `VENTANA_S` anteriores al tocado,

    v_acerca(tirador) = velocidad de su centro x en coordenadas de PISTA, con signo POSITIVO
                        cuando se acerca al rival

y el sustituto es: **`sospecha_contraataque` = el que enciende NO se acercaba y el rival SI.**

COMO SE VALIDA, que es lo que hace que esto no sea una suposicion. En los clips de DOS luces existe
la etiqueta real: `y` en el `index.jsonl` (0 = gana el izquierdo, 1 = gana el derecho), sacada del
cambio de tanteo. Y existe el atajo ya medido del proyecto: el que encendio PRIMERO gana el 63,51 %
de las veces. Entonces:

  - se reproduce el atajo del orden de luces sobre esta muestra, que es la LINEA BASE GRATIS;
  - se mira si el sustituto de movimiento acierta en los clips donde **el atajo se equivoca**, que
    son los 960 clips del techo del 73,6 % y el sitio donde el proyecto esta atascado.

CRITERIOS DECLARADOS ANTES DE EJECUTAR:

  (1) el sustituto SIRVE si, en los clips de dos luces donde el atajo del orden FALLA, acierta en
      >= 60 % de los casos. Por debajo de eso no separa nada que el orden no separe ya.
  (2) y ademas su tasa de "sospecha" en los clips de UNA luz tiene que ser distinta de la de los de
      DOS con un margen de >= 10 puntos. Si sale igual en los dos, el sustituto esta midiendo
      "alguien se movia" y no "hubo contraataque".

Los dos criterios pueden fallar y entonces se informa: lo que NO se hace es usar el sustituto para
construir un corpus de 18.053 clips sin haberlo medido.

ECONOMIA. Todo se lee de los NPZ (`boxes`, `pan`, `lights`, `meta_json`). Cero decodificacion de
video, cero GPU.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
BASE_VIEJA = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565"
)
DIRS_NPZ = [
    os.path.join(BASE_VIEJA, r"localrun\runs_extra1\data_20260907T005507157743Z\npz"),
    os.path.join(BASE_VIEJA, r"localrun\runs_extra2"),
    os.path.join(BASE_VIEJA, r"localrun\runs_extra3"),
]
VENTANA_S = 0.5
V_MIN = 0.002


def etiquetas() -> dict[str, dict]:
    """clip -> etiqueta real `y` y `touch`, de los `index.jsonl`. Es una lectura de indice."""
    fuera = {}
    for ruta in sorted(glob.glob(os.path.join(BASE_VIEJA, "localrun", "runs*", "*",
                                              "index.jsonl"))):
        with open(ruta, encoding="utf-8") as fh:
            for linea in fh:
                linea = linea.strip()
                if not linea:
                    continue
                o = json.loads(linea)
                if o.get("status") != "ok" or o.get("y") is None:
                    continue
                fuera.setdefault(o["clip_id"], dict(
                    y=int(o["y"]), touch=o.get("touch"), sede=o.get("tournament_id"),
                    label_ok=bool(o.get("label_ok"))))
    return fuera


def mapa_npz() -> dict[str, str]:
    mapa = {}
    for d in DIRS_NPZ:
        for ruta in glob.glob(os.path.join(d, "**", "*.npz"), recursive=True):
            mapa.setdefault(os.path.splitext(os.path.basename(ruta))[0], ruta)
    return mapa


def mide_clip(ruta: str) -> dict | None:
    """Velocidades de acercamiento de los dos tiradores en la ventana previa al tocado."""
    with np.load(ruta, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"]))
        boxes = np.asarray(z["boxes"], np.float32)
        pan = np.asarray(z["pan"], np.float32) if "pan" in z.files else None
        tiempo = np.asarray(z["time_s"], np.float64)
    T = len(boxes)
    if T < 6 or pan is None:
        return None
    t_touch = meta.get("t_touch_s")
    if t_touch is None:
        return None
    k1 = int(np.clip(np.searchsorted(tiempo, t_touch), 3, T - 1))
    k0 = int(np.clip(np.searchsorted(tiempo, t_touch - VENTANA_S), 0, k1 - 2))
    if k1 - k0 < 2:
        return None
    cx = (boxes[:, :, 0] + boxes[:, :, 2]) / 2.0
    ancho = float(meta.get("size", [1920, 1080])[0])
    cx_pista = cx + (pan[:, None] / ancho)
    x0, x1 = cx_pista[k0], cx_pista[k1]
    dt = max(1e-6, float(tiempo[k1] - tiempo[k0]))
    izq = 0 if float(cx[k1, 0]) <= float(cx[k1, 1]) else 1
    der = 1 - izq
    v_izq = float(x1[izq] - x0[izq]) / dt
    v_der = float(x1[der] - x0[der]) / dt
    tl = meta.get("touch_lights") or {}
    return dict(
        n_color=int(tl.get("red_L") is not None) + int(tl.get("green_R") is not None),
        roja=tl.get("red_L") is not None, verde=tl.get("green_R") is not None,
        blanca=(tl.get("white_L") is not None) or (tl.get("white_R") is not None),
        first_light=meta.get("first_light"), dt_ms=meta.get("dt_ms"),
        v_acerca_izq=v_izq, v_acerca_der=-v_der,
        cobertura=min(float(meta.get("coverage_L") or 0), float(meta.get("coverage_R") or 0)),
    )


def main() -> int:
    etq = etiquetas()
    mapa = mapa_npz()
    print(f"clips con etiqueta `y`: {len(etq)}   npz localizados: {len(mapa)}")
    filas = []
    for i, (clip, e) in enumerate(sorted(etq.items())):
        ruta = mapa.get(clip)
        if ruta is None:
            continue
        try:
            m = mide_clip(ruta)
        except Exception:
            continue
        if m is None:
            continue
        filas.append(dict(clip=clip, **e, **m))
        if (i + 1) % 2000 == 0:
            print(f"  leidos {i + 1}")
    d = pd.DataFrame(filas)
    d = d[d["cobertura"] >= 0.9]
    print(f"clips medidos con cobertura >= 0,9 de los dos tiradores: {len(d)}")

    d["lado_luz"] = np.where(d["roja"] & ~d["verde"], "L",
                             np.where(d["verde"] & ~d["roja"], "R", "ambas"))
    """El lado EN CUESTION no es el mismo en los dos casos, y confundirlo fue un fallo de diseno.

    En un clip de UNA luz, el tirador cuya prioridad esta en duda es el que enciende: si no tenia
    prioridad, fue un contraataque. En un clip de DOS luces, el que esta en duda es el que encendio
    PRIMERO, porque el segundo es por definicion el que respondio. La primera version de este script
    solo definia el lado para los clips de una luz, y por eso daba `sospecha_ca` en el 0,00 % de los
    de dos luces: no era un resultado, era un `NaN` disfrazado de cero. Se corrige aqui.
    """
    lado_duda = np.where(d["lado_luz"] == "ambas",
                         np.where(d["first_light"] == "left", "L",
                                  np.where(d["first_light"] == "right", "R", "")),
                         d["lado_luz"])
    d["lado_duda"] = lado_duda
    d["v_duda"] = np.where(lado_duda == "L", d["v_acerca_izq"],
                           np.where(lado_duda == "R", d["v_acerca_der"], np.nan))
    d["v_otro"] = np.where(lado_duda == "L", d["v_acerca_der"],
                           np.where(lado_duda == "R", d["v_acerca_izq"], np.nan))
    d["sospecha_ca"] = (d["v_duda"] <= V_MIN) & (d["v_otro"] > V_MIN)
    d["avanzaba"] = (d["v_duda"] > V_MIN) & (d["v_otro"] <= V_MIN)

    una = d[d["n_color"] == 1]
    dos = d[d["n_color"] == 2]
    print(f"\nclips de UNA luz de color: {len(una)}   de DOS: {len(dos)}")

    print("\n--- LINEA BASE GRATIS: el atajo del orden de luces en los de DOS ---")
    dos = d[d["n_color"] == 2]
    una = d[d["n_color"] == 1]
    dd = dos[dos["first_light"].isin(["left", "right"])].copy()
    primero_der = (dd["first_light"] == "right").astype(int)
    acc_primero = float((primero_der == dd["y"]).mean())
    dd["pred_atajo"] = 1 - primero_der
    acierto = float((dd["pred_atajo"] == dd["y"]).mean())
    print(f"  clips con orden resoluble: {len(dd)} de {len(dos)} "
          f"({100.0 * len(dd) / max(1, len(dos)):.1f} %; el resto son `same_frame`)")
    print(f"  regla 'gana el que enciende PRIMERO':  {100.0 * acc_primero:.2f} %")
    print(f"  regla 'gana el que enciende SEGUNDO':  {100.0 * acierto:.2f} %   "
          f"<- esta es la que reproduce el 63,51 % del proyecto")
    print(f"  mayoritaria (predecir siempre el mismo lado): "
          f"{100.0 * max(dd['y'].mean(), 1 - dd['y'].mean()):.2f} %")

    falla = dd[dd["pred_atajo"] != dd["y"]]
    print(f"\n--- (1) el SUSTITUTO en los de DOS luces, sobre el que estaba en duda ---")
    print(f"  sospecha de contraataque (el que encendio primero NO se acercaba y el otro si): "
          f"{dd['sospecha_ca'].sum()}/{len(dd)} = {100.0 * dd['sospecha_ca'].mean():.2f} %")
    print(f"  al contrario, el que encendio primero SI se acercaba y el otro no: "
          f"{dd['avanzaba'].sum()}/{len(dd)} = {100.0 * dd['avanzaba'].mean():.2f} %")
    pred_mov = np.where(dd["avanzaba"], 1 - dd["pred_atajo"], dd["pred_atajo"])
    acc_mov = float((pred_mov == dd["y"]).mean())
    print(f"  regla combinada (si el primero AVANZABA, gana el primero; si no, el atajo): "
          f"{100.0 * acc_mov:.2f} %  contra {100.0 * acierto:.2f} % del atajo solo  "
          f"({100.0 * (acc_mov - acierto):+.2f} puntos)")
    print(f"\n  y donde el atajo FALLA ({len(falla)} clips):")
    for col, txt in (("avanzaba", "el primero avanzaba"), ("sospecha_ca", "sospecha de contraataque")):
        print(f"    {txt:26s} en los que el atajo falla: {100.0 * falla[col].mean():5.2f} %   "
              f"en los que acierta: "
              f"{100.0 * dd.loc[dd['pred_atajo'] == dd['y'], col].mean():5.2f} %")
    sep = abs(float(falla["avanzaba"].mean()) - float(dd.loc[dd["pred_atajo"] == dd["y"], "avanzaba"].mean()))
    print(f"  separacion del sustituto entre aciertos y fallos del atajo: "
          f"{100.0 * sep:.2f} puntos   (criterio declarado: >= 10)")
    print(f"  VEREDICTO (1): {'PASA' if sep >= 0.10 else 'NO PASA'}")

    print("\n--- (2) la tasa de sospecha en UNA luz contra DOS luces ---")
    s_una = float(una["sospecha_ca"].mean())
    s_dos = float(dd["sospecha_ca"].mean())
    print(f"  una luz: {100.0 * s_una:.2f} %   dos luces: {100.0 * s_dos:.2f} %   "
          f"margen {100.0 * (s_una - s_dos):+.2f} puntos   (criterio: |margen| >= 10)")
    print(f"  VEREDICTO (2): {'PASA' if abs(s_una - s_dos) >= 0.10 else 'NO PASA'}")

    print("\n--- el corpus que saldria de la idea, con el sustituto tal cual ---")
    print(f"  clips de dos luces (conjunto limpio):           {len(dos):6d}")
    print(f"  clips de una luz SIN sospecha de contraataque:  {int((~una['sospecha_ca']).sum()):6d}")
    print(f"  suma:                                           "
          f"{len(dos) + int((~una['sospecha_ca']).sum()):6d}   "
          f"(x{(len(dos) + int((~una['sospecha_ca']).sum())) / max(1, len(dos)):.2f} sobre las de dos)")

    print("\n--- y la etiqueta de las de UNA luz: es trivial? ---")
    u2 = una[una["lado_luz"].isin(["L", "R"])].copy()
    u2["pred_lado"] = (u2["lado_luz"] == "R").astype(int)
    print(f"  predecir 'gana el que enciende': {100.0 * float((u2['pred_lado'] == u2['y']).mean()):.2f} % "
          f"en {len(u2)} clips")

    d.to_csv(os.path.join(SALIDA, "una_luz_y_contraataque.csv"), index=False)
    print(f"\ncsv: {os.path.join(SALIDA, 'una_luz_y_contraataque.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
