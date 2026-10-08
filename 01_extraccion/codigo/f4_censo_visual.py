"""Paso 1 del encargo: el CENSO de los 21 rotulos, con la region de la luz localizada por medida.

Lo que hace, por sede. Decodifica del clip elegido dos cuadros y solo dos que importan:

  `k_on`   el cuadro del tocado, donde la luz esta ENCENDIDA (viene de `t_touch_s` del NPZ viejo)
  `k_off`  dos segundos antes, donde esta APAGADA

y ademas 8 cuadros repartidos por el tramo anterior al tocado, que sirven para un mapa de
ESTABILIDAD TEMPORAL. Con esas tres cosas se localiza la luz sin buscar rojo en todo el cuadro:

  candidata = (cambia mucho entre apagado y encendido) Y (es un pixel ESTATICO en el clip)

El segundo factor es el que hace el trabajo. Un pixel rojo de una bandera o de la pista esta rojo
siempre, asi que no cambia y no es candidato. Un tirador que se mueve cambia muchisimo, pero NO es
estatico, asi que tampoco. Solo el rotulo sobrepuesto cumple las dos cosas: no se mueve y se
enciende.

CRITERIO DECLARADO ANTES DE EJECUTAR, con los umbrales fijados antes de ver un resultado:

  ESTATICO   desviacion tipica temporal del gris < 6,0 niveles en los 8 cuadros previos
  CAMBIO     distancia L1 en BGR entre apagado y encendido > 40 niveles (de 765 posibles)
  AREA MIN   20 px a la resolucion nativa (el punto de Tokio mide 0,018 W ~ 34 px de lado a 1080p)

La cifra que decide si este paso sirve: en cuantas de las 21 sedes la region candidata mas grande
cae DENTRO de una franja del rotulo y no encima de un tirador. Se declara que el paso vale si son
al menos 18 de 21; por debajo de eso el mapa de estabilidad no es una via y hay que decirlo.

POR QUE NO SE USA LA VENTANA DEL NPZ. El NPZ viejo guarda solo [tocado - 4 s, tocado + 0,3 s], y
para el mapa de estabilidad quiero cuadros de antes del `allez`, donde no hay nadie tocando. Asi que
se lee del .mp4 original, con `cap.set` cuadro a cuadro: 10 saltos por clip, 210 en total. El PC del
usuario esta saturado y esto es lo mas barato que cubre la pregunta.

SALIDA. Un PNG por sede en `02_validacion/muestras/luces/censo/` con el cuadro completo, las cajas
candidatas marcadas, y el recorte de la zona apagada y encendida al lado para poder MIRARLO. Y un
CSV con las cajas medidas y su color.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

PROY = r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO"
SCRATCH = (
    r"C:\Users\Alejandro\AppData\Local\Temp\claude"
    r"\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"
)
SALIDA = os.path.join(PROY, r"02_validacion\muestras\luces\censo")

STD_ESTATICO = 6.0
L1_CAMBIO = 40
AREA_MIN = 20
N_ESTABILIDAD = 8
SEG_ANTES = 2.0

EN_CANAL = {"roja": "red", "verde": "green", "blanca": "white off-target"}
EN_CANAL_CORTO = {"roja": "red", "verde": "green", "blanca": "white"}
"""El texto DENTRO de las figuras va en INGLES porque las figuras se publican.

El proyecto va camino de un repositorio publico y el usuario pidio que todo lo que se lee en una
imagen se entienda sin saber espaniol. Los docstrings, los nombres de columna de los CSV y la
seccion del documento maestro siguen en espaniol: lo que se traduce es lo que se PUBLICA como
imagen o como README. `blanca` se rotula `white off-target` y no solo `white`, porque en florete la
luz blanca significa tocado FUERA de superficie valida y `white` a secas no lo dice.
"""


def carga_referencias() -> list[dict]:
    """Une la lista de un clip por torneo con la ruta del mp4 y el `t_touch_s` del NPZ viejo."""
    import pandas as pd

    lista = []
    with open(os.path.join(SCRATCH, "un_clip_por_torneo.txt"), encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                sede, clip = linea.strip().split("\t")
                lista.append((sede, clip))
    indice = {}
    with open(os.path.join(SCRATCH, "indice_mp4.tsv"), encoding="utf-8-sig") as fh:
        for linea in fh:
            linea = linea.rstrip("\r\n")
            if linea:
                stem, camino = linea.split("\t", 1)
                indice.setdefault(stem, camino)
    npz = pd.read_csv(os.path.join(SCRATCH, "luces_npz_viejo.csv")).set_index("clip")
    fuera = []
    for sede, clip in lista:
        fila = dict(sede=sede, clip=clip, ruta=indice.get(clip, ""))
        if clip in npz.index:
            fila["t_touch_s"] = float(npz.loc[clip, "t_touch_s"])
            fila["detector_viejo"] = npz.loc[clip, "light_detector"]
            fila["n_luces_viejo"] = npz.loc[clip, "n_valid_lights"]
        else:
            fila["t_touch_s"] = None
            fila["detector_viejo"] = ""
            fila["n_luces_viejo"] = None
        fuera.append(fila)
    return fuera


def lee_cuadros(ruta: str, t_touch: float | None) -> dict:
    """Decodifica los 10 cuadros que hacen falta y devuelve apagado, encendido y la pila estable."""
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return {}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    k_on = int(round(t_touch * fps)) if t_touch else int(0.85 * (n - 1))
    k_on = max(0, min(n - 1, k_on))
    k_off = max(0, k_on - int(round(SEG_ANTES * fps)))

    def coge(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        return img if ok else None

    encendido = coge(k_on)
    apagado = coge(k_off)
    pila = []
    if k_off > 4:
        for i in range(N_ESTABILIDAD):
            k = int(round(i * (k_off - 1) / max(1, N_ESTABILIDAD - 1)))
            img = coge(k)
            if img is not None:
                pila.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return dict(encendido=encendido, apagado=apagado, pila=pila, k_on=k_on, k_off=k_off,
                fps=fps, n=n)


def candidatas(apagado, encendido, pila) -> tuple[np.ndarray, list[dict]]:
    """Devuelve la mascara candidata y las cajas medidas, ordenadas por area decreciente.

    El color de cada caja se da en BGR medio del cuadro ENCENDIDO y en HSV, porque la clasificacion
    roja / verde / blanca se decidira con eso y conviene tener el numero delante antes de elegir
    umbrales de color.
    """
    d = np.abs(apagado.astype(np.int16) - encendido.astype(np.int16)).sum(axis=2)
    cambio = d > L1_CAMBIO
    if pila:
        est = np.stack(pila, 0).std(axis=0) < STD_ESTATICO
    else:
        est = np.ones(cambio.shape, bool)
    masc = (cambio & est).astype(np.uint8)
    masc = cv2.morphologyEx(masc, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    nlab, lab, stats, cent = cv2.connectedComponentsWithStats(masc)
    H, W = masc.shape
    hsv = cv2.cvtColor(encendido, cv2.COLOR_BGR2HSV)
    cajas = []
    for i in range(1, nlab):
        x, y, w, h, a = stats[i]
        if a < AREA_MIN:
            continue
        m = lab[y:y + h, x:x + w] == i
        bgr = encendido[y:y + h, x:x + w][m].mean(axis=0)
        hsv_m = hsv[y:y + h, x:x + w][m].mean(axis=0)
        cajas.append(dict(x=x, y=y, w=w, h=h, area=int(a),
                          fx=round(x / W, 4), fy=round(y / H, 4),
                          fw=round(w / W, 4), fh=round(h / H, 4),
                          B=round(float(bgr[0]), 1), G=round(float(bgr[1]), 1),
                          R=round(float(bgr[2]), 1),
                          Hm=round(float(hsv_m[0]), 1), S=round(float(hsv_m[1]), 1),
                          V=round(float(hsv_m[2]), 1)))
    cajas.sort(key=lambda c: -c["area"])
    return masc, cajas


def pinta(sede: str, info: dict, cajas: list[dict], destino: str) -> None:
    """Composicion para MIRAR: cuadro completo con cajas, y el recorte apagado / encendido al lado."""
    enc, apa = info["encendido"], info["apagado"]
    H, W = enc.shape[:2]
    vis = enc.copy()
    for c in cajas[:12]:
        cv2.rectangle(vis, (c["x"] - 2, c["y"] - 2), (c["x"] + c["w"] + 2, c["y"] + c["h"] + 2),
                      (255, 255, 0), 2)
    anchura = 960
    esc = anchura / W
    fila1 = cv2.resize(vis, (anchura, int(round(H * esc))))
    cv2.putText(fila1, f"{sede}  frame {info['k_on']} (LIGHT ON)  {W}x{H}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if cajas:
        x0 = max(0, min(c["x"] for c in cajas[:12]) - 30)
        x1 = min(W, max(c["x"] + c["w"] for c in cajas[:12]) + 30)
        y0 = max(0, min(c["y"] for c in cajas[:12]) - 25)
        y1 = min(H, max(c["y"] + c["h"] for c in cajas[:12]) + 25)
    else:
        x0, x1, y0, y1 = 0, W, int(0.70 * H), H
    if x1 - x0 < 80:
        x0, x1 = max(0, x0 - 60), min(W, x1 + 60)
    if y1 - y0 < 40:
        y0, y1 = max(0, y0 - 30), min(H, y1 + 30)
    rec_a = apa[y0:y1, x0:x1]
    rec_e = enc[y0:y1, x0:x1]
    esc2 = min(3.0, anchura / max(1, x1 - x0))
    rec_a = cv2.resize(rec_a, None, fx=esc2, fy=esc2, interpolation=cv2.INTER_NEAREST)
    rec_e = cv2.resize(rec_e, None, fx=esc2, fy=esc2, interpolation=cv2.INTER_NEAREST)
    for img, txt in ((rec_a, f"LIGHTS OFF  frame {info['k_off']}"),
                     (rec_e, f"LIGHT ON  frame {info['k_on']}")):
        cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    tira_full = np.zeros((int(0.30 * H * esc) + 4, anchura, 3), np.uint8)
    banda = enc[int(0.70 * H):H, :]
    banda = cv2.resize(banda, (anchura, tira_full.shape[0] - 4))
    tira_full[2:-2, :] = banda
    cv2.putText(tira_full, "bottom band, y = 0.70-1.00 H", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)

    piezas = [fila1, tira_full, rec_a, rec_e]
    anchos = max(p.shape[1] for p in piezas)
    lienzo = []
    for p in piezas:
        if p.shape[1] != anchos:
            pad = np.zeros((p.shape[0], anchos - p.shape[1], 3), np.uint8)
            p = np.hstack([p, pad])
        lienzo.append(p)
        lienzo.append(np.full((3, anchos, 3), 60, np.uint8))
    cv2.imwrite(destino, np.vstack(lienzo), [int(cv2.IMWRITE_PNG_COMPRESSION), 6])


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    refs = carga_referencias()
    filas = []
    dentro_rotulo = 0
    for r in refs:
        if not r["ruta"]:
            print(f"FALTA ruta: {r['sede']} {r['clip']}")
            continue
        info = lee_cuadros(r["ruta"], r["t_touch_s"])
        if not info or info["encendido"] is None or info["apagado"] is None:
            print(f"FALLO decodificando {r['sede']}")
            continue
        masc, cajas = candidatas(info["apagado"], info["encendido"], info["pila"])
        H, W = masc.shape
        pinta(r["sede"], info, cajas, os.path.join(SALIDA, f"{r['sede']}.png"))
        mayor = cajas[0] if cajas else None
        if mayor and mayor["fy"] > 0.62:
            dentro_rotulo += 1
        print(f"\n### {r['sede']}  {W}x{H}  c_on={info['k_on']} c_off={info['k_off']} "
              f"det_viejo={r['detector_viejo']} n_luces={r['n_luces_viejo']}")
        print(f"    px estaticos: {100.0 * (np.stack(info['pila']).std(0) < STD_ESTATICO).mean():.1f} %"
              if info["pila"] else "    sin pila de estabilidad")
        print(f"    candidatas: {len(cajas)}")
        for c in cajas[:6]:
            print(f"      area={c['area']:6d} fx={c['fx']:.3f} fy={c['fy']:.3f} "
                  f"fw={c['fw']:.3f} fh={c['fh']:.3f}  BGR=({c['B']:.0f},{c['G']:.0f},{c['R']:.0f}) "
                  f"HSV=({c['Hm']:.0f},{c['S']:.0f},{c['V']:.0f})")
        for j, c in enumerate(cajas[:12]):
            filas.append(dict(sede=r["sede"], clip=r["clip"], orden=j, **c))

    destino = os.path.join(SALIDA, "candidatas_censo.csv")
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\ncajas guardadas: {destino}")
    print(f"sedes cuya candidata MAYOR cae en la mitad inferior (fy>0.62): {dentro_rotulo} de {len(refs)}")
    print("criterio declarado: sirve si >= 18 de 21")
    return 0


if __name__ == "__main__":
    sys.exit(main())
