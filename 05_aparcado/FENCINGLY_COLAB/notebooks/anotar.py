"""Bucle de anotacion de la cazoleta: propone, filtra con el prior, y saca una muestra CIEGA a verificar.

EL BUCLE, y por que cada paso es como es:

  1. PROPONER   SAM3 con `the hilt of a sword` sobre el recorte de la muneca armada, ampliado 8x.
                Sin recortar no hay nada: en el cuadro completo todas las palabras del florete
                devuelven ~0 instancias.
  2. FILTRAR    por el PRIOR GEOMETRICO (muneca + 0,6 diametros a lo largo del antebrazo), NUNCA por
                la confianza del modelo: los aciertos promedian 0,238 y los fallos 0,253, o sea que
                los fallos puntuan MAS.
  3. VERIFICAR  una muestra humana, CIEGA y con controles mezclados.
  4. PROPAGAR   solo despues, y sabiendo que la propagacion NO se valida sola: sembrar SAM2-video en
                un senuelo dentro del cuerpo pasa un criterio de consistencia en 10 de 12 casos.

POR QUE LA MUESTRA LLEVA CONTROLES CIEGOS: si quien verifica marca como buenas las cajas generadas por
`a banana`, la verificacion no vale y hay que saberlo. La muestra mezcla tres origenes y el fichero de
respuestas se guarda APARTE:

  objetivo   `the hilt of a sword`, el proponente de verdad
  senuelo    `a banana` en el mismo recorte: sus cajas NO deberian aceptarse
  azar       una caja del tamano correcto en una posicion aleatoria del recorte

Con eso, de las respuestas salen tres cosas: la precision del proponente, la tasa de falso positivo
de quien verifica, y si el prior por si solo bastaria.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

import cazoleta as CZ

ORIGENES = ("objetivo", "senuelo", "azar")
PALABRA_DE = {"objetivo": CZ.PALABRA, "senuelo": CZ.PALABRA_SENUELO}
CELDA = 220


def _caja_azar(diam, rad, rng):
    """Caja del tamano de una cazoleta en posicion aleatoria del recorte, en coordenadas ampliadas."""
    lado = diam * CZ.AMPLIA
    maxi = 2 * rad * CZ.AMPLIA - lado
    x = float(rng.uniform(0, max(maxi, 1)))
    y = float(rng.uniform(0, max(maxi, 1)))
    return (x, y, x + lado, y + lado)


def propone_uno(imagen, poses_t, lado, origen, proc, modelo, rng) -> dict | None:
    """QUE HACE: un candidato de un origen concreto, con su recorte ya ampliado para poder pintarlo."""
    import cv2

    im = np.asarray(imagen)
    H, W = im.shape[:2]
    pe = CZ.punto_esperado(poses_t, lado, W, H)
    if pe is None:
        return None
    cx, cy, diam, _ux, _uy = pe
    rad = max(int(diam * 2.2), 26)
    x0, y0 = int(max(0, cx - rad)), int(max(0, cy - rad))
    x1, y1 = int(min(W, cx + rad)), int(min(H, cy + rad))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    gr = cv2.resize(im[y0:y1, x0:x1], None, fx=CZ.AMPLIA, fy=CZ.AMPLIA,
                    interpolation=cv2.INTER_CUBIC)
    pex, pey = (cx - x0) * CZ.AMPLIA, (cy - y0) * CZ.AMPLIA

    if origen == "azar":
        b = _caja_azar(diam, rad, rng)
        s = float("nan")
    else:
        import torch

        with torch.inference_mode():
            rgb = np.ascontiguousarray(gr[:, :, ::-1])
            ent = proc(images=rgb, text=PALABRA_DE[origen], return_tensors="pt").to(
                next(modelo.parameters()).device)
            out = modelo(**ent)
            res = proc.post_process_instance_segmentation(
                out, threshold=CZ.UMBRAL, mask_threshold=0.5,
                target_sizes=[(gr.shape[0], gr.shape[1])])[0]
        cajas, punt = res.get("boxes"), res.get("scores")
        cajas = cajas.cpu().numpy() if cajas is not None and len(cajas) else np.zeros((0, 4))
        punt = punt.cpu().numpy() if punt is not None and len(punt) else np.zeros((0,))
        if not len(cajas):
            return None
        mejor, dmin = None, 1e18
        for bb, ss in zip(cajas, punt):
            bx, by = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            dd = float(np.hypot(bx - pex, by - pey)) / (diam * CZ.AMPLIA)
            if dd < dmin:
                dmin, mejor, s = dd, tuple(bb), float(ss)
        b = mejor

    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    dist = float(np.hypot(bx - pex, by - pey)) / (diam * CZ.AMPLIA)
    return {"origen": origen, "recorte": gr, "caja": tuple(float(v) for v in b),
            "prior_xy": (pex, pey), "diam_ampliado": diam * CZ.AMPLIA,
            "score": s, "dist_diam": round(dist, 3),
            "pasa_prior": bool(dist <= CZ.TOL_DIAM),
            "guard_x": round(bx / CZ.AMPLIA + x0, 2), "guard_y": round(by / CZ.AMPLIA + y0, 2)}


def hoja_de_contacto(candidatos, salida, por_fila: int = 6, celda: int = CELDA) -> dict:
    """QUE HACE: pinta la muestra numerada en hojas de contacto, SIN decir de que origen es cada celda.

    Cada celda lleva su numero, la caja propuesta en rojo y el circulo del prior en amarillo. No lleva
    ni la palabra ni la puntuacion: eso sesgaria a quien verifica.
    """
    import cv2

    hojas, fila, filas = [], [], []
    for i, c in enumerate(candidatos, 1):
        vis = c["recorte"].copy()
        b = c["caja"]
        cv2.circle(vis, (int(c["prior_xy"][0]), int(c["prior_xy"][1])),
                   int(c["diam_ampliado"] / 2), (0, 255, 255), 2)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        vis = cv2.resize(vis, (celda, celda), interpolation=cv2.INTER_AREA)
        cv2.rectangle(vis, (0, 0), (celda - 1, celda - 1), (60, 60, 60), 1)
        cv2.rectangle(vis, (0, 0), (46, 26), (0, 0, 0), -1)
        cv2.putText(vis, "%d" % i, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fila.append(vis)
        if len(fila) == por_fila:
            filas.append(np.hstack(fila))
            fila = []
    if fila:
        while len(fila) < por_fila:
            fila.append(np.zeros((celda, celda, 3), dtype="uint8"))
        filas.append(np.hstack(fila))
    salida = pathlib.Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if filas:
        cv2.imwrite(str(salida), np.vstack(filas))
        hojas.append(str(salida))
    return {"hojas": hojas, "n": len(candidatos)}


def escribe_muestra(candidatos, meta, dir_salida) -> dict:
    """QUE HACE: escribe la hoja de contacto, el CSV a rellenar y la CLAVE aparte.

    El CSV que se entrega NO lleva la columna `origen`. La clave va a otro fichero para que la muestra
    sea ciega de verdad; si estuvieran juntas, bastaria abrir el CSV para verla.
    """
    import pandas as pd

    d = pathlib.Path(dir_salida)
    d.mkdir(parents=True, exist_ok=True)
    hoja = hoja_de_contacto(candidatos, d / "muestra.png")

    filas_vis, filas_clave = [], []
    for i, (c, m) in enumerate(zip(candidatos, meta), 1):
        filas_vis.append({"n": i, "veredicto": "", "clip": m["clip"], "sede": m["sede"],
                          "alto": m["alto"], "cuadro": m["cuadro"], "lado": m["lado"]})
        filas_clave.append({"n": i, "origen": c["origen"], "score": c["score"],
                            "dist_diam": c["dist_diam"], "pasa_prior": c["pasa_prior"],
                            "guard_x": c["guard_x"], "guard_y": c["guard_y"],
                            "clip": m["clip"], "cuadro": m["cuadro"], "lado": m["lado"]})
    pd.DataFrame(filas_vis).to_csv(d / "verificar.csv", index=False)
    pd.DataFrame(filas_clave).to_csv(d / "CLAVE_no_mirar.csv", index=False)
    (d / "COMO_VERIFICAR.md").write_text(
        "# Como verificar la muestra\n\n"
        "Abre `muestra.png`. Cada celda lleva un numero, un **circulo amarillo** donde el prior dice\n"
        "que deberia estar la cazoleta, y una **caja roja** que es la propuesta del modelo.\n\n"
        "En `verificar.csv`, para cada `n` pon en `veredicto`:\n\n"
        "  `si`     la caja roja esta sobre la cazoleta (o sobre cazoleta + guante-empunadura)\n"
        "  `no`     la caja roja NO esta sobre la cazoleta\n"
        "  `dudoso` no se ve bastante para decidir\n\n"
        "**No mires `CLAVE_no_mirar.csv`.** La muestra lleva **controles mezclados**: algunas cajas\n"
        "las genero un prompt senuelo y otras estan puestas al azar. Sirven para medir si la\n"
        "verificacion es fiable, y solo funcionan si no sabes cuales son.\n\n"
        "Marca lo que veas, sin intentar ser generoso. Un `no` vale igual que un `si`.\n",
        encoding="utf-8")
    print("=" * 78)
    print("MUESTRA CIEGA ESCRITA")
    print("  hoja de contacto ... %s" % (d / "muestra.png"))
    print("  a rellenar ......... %s" % (d / "verificar.csv"))
    print("  clave (no mirar) ... %s" % (d / "CLAVE_no_mirar.csv"))
    print("  celdas ............. %d" % len(candidatos))
    print("=" * 78)
    return {"dir": str(d), "n": len(candidatos), **hoja}


def evalua(dir_muestra) -> dict:
    """QUE HACE: cruza los veredictos con la clave y saca las tres cifras que la muestra existe para dar.

    precision del proponente = si / (si + no) sobre las celdas de origen `objetivo`
    falso positivo de quien verifica = si sobre las de `senuelo` y `azar`
    aporte del prior = precision de las que pasan el prior contra las que no
    """
    import pandas as pd

    d = pathlib.Path(dir_muestra)
    v = pd.read_csv(d / "verificar.csv")
    k = pd.read_csv(d / "CLAVE_no_mirar.csv")
    j = v.merge(k, on="n", suffixes=("", "_k"))
    j["veredicto"] = j["veredicto"].astype(str).str.strip().str.lower()
    resp = j[j.veredicto.isin(["si", "no"])]
    print("respondidas %d de %d  (dudosas %d)" % (
        len(resp), len(j), int((j.veredicto == "dudoso").sum())))
    fuera = {"n": len(j), "respondidas": len(resp)}
    for origen in ORIGENES:
        g = resp[resp.origen == origen]
        if not len(g):
            continue
        tasa = float((g.veredicto == "si").mean())
        fuera[origen] = tasa
        print("  %-10s aceptadas %5.2f  (n=%d)" % (origen, tasa, len(g)))
    obj = resp[resp.origen == "objetivo"]
    if len(obj) and obj.pasa_prior.nunique() > 1:
        for pasa, g in obj.groupby("pasa_prior"):
            print("  objetivo, prior=%s ... %5.2f  (n=%d)" % (
                pasa, float((g.veredicto == "si").mean()), len(g)))
    ctrl = resp[resp.origen.isin(["senuelo", "azar"])]
    if len(ctrl):
        fp = float((ctrl.veredicto == "si").mean())
        print()
        if fp > 0.25:
            print("  AVISO: se aceptan %.0f %% de los CONTROLES. La verificacion no es fiable y la" % (100 * fp))
            print("  precision del proponente no se puede creer con estas respuestas.")
        else:
            print("  Controles rechazados como debian (%.0f %% aceptados): la verificacion es utilizable." % (100 * fp))
    return fuera
