"""Mide si una pose INVENTADA se puede sustituir por una pose MEDIDA volviendo a pasar ese cuadro
por la cabeza `pose` de Sapiens 2, dirigida a la caja estimada `Be`.

MOVIDO AL ARBOL DEL PROYECTO EL 2026-09-11, y no es un detalle de orden: este fichero y su
`rescate_casos.csv` vivian SOLO en el scratchpad de una sesion, que es un directorio temporal con
180 GB dentro y 85 GB libres en C:. La medida de los 650 casos cuesta 233 s de GPU y su evidencia
esta ahora en `01_extraccion/pruebas/rescate_*`. Quien construya la cascada de tres rondas importa
`pose_en_caja` y `gate` de aqui: son la segunda ronda, y reimplementarlas daria otra cifra.

LA IDEA ES DEL USUARIO: "las de poses inventadas, que son las rosas, tambien detecte muchos errores.
Lo que puedes hacer es pasar esos fotogramas otra vez solos a Sapiens 2 para que los saque".

POR QUE PUEDE FUNCIONAR Y POR QUE NO ES LO QUE YA HAY. La tuberia ya redetecta (`tipo = 3`) pero con
YOLO sobre un recorte, y un `origen != 0` es justamente un cuadro donde ESO ya fallo. La cabeza de
pose de Sapiens 2 es TOP-DOWN: no necesita que un detector confirme a la persona, se le da la caja y
saca puntos. Asi que es una via distinta, no un reintento de la misma.

Y POR ESO MISMO EL CONTROL NO ES OPCIONAL: un modelo top-down devuelve puntos SIEMPRE, tambien sobre
una caja donde no hay nadie. "Se detecta en el 60 %" no significa nada sin saber que devuelve el gate
cuando la respuesta es conocida y cuando no hay nadie.

=======================  LOS TRES NUMEROS, DECLARADOS ANTES DE EJECUTAR  =======================

DENOMINADORES, declarados antes de medir porque en este proyecto el mismo nombre da cifras distintas:
  38 clips, NO 39. Hay 39 ficheros `poses_*.npz` pero `poses_SentimentalLargeBedlingtonterrier.npz`
  tiene esquema `(cuadro, kp, lado, sc)` y no es un NPZ de poses entregadas.
  21.950 lados-cuadro dentro de `mask`; 21.300 medidos (97,04 %) y 650 inventados (2,96 %).
  Los 650 son `origen == 1` (interpolada). `origen == 2` (sostenida) es 0 en TODO el array: lo
  quito `recorta_por_poses`. O sea que las poses rosas que se ven son 650 interpoladas y 0
  sostenidas.

N1  COBERTURA. De los 650 lados-cuadro inventados, en cuantos la segunda pasada sobre `Be` da una
    pose que pasa el filtro de calidad del proyecto: tras poner a cero la confianza de los keypoints
    fuera de su propia caja (`limpia_keypoints`, HOLGURA_KP = 0,05, caja = `Be`), al menos
    `PUNTOS_MIN` = 20 de los 63 puntos del cuerpo con confianza >= 0,3. Denominador 650.
    Se da tambien la lectura ESTRICTA del filtro (`gate_estricto`): la anterior Y que ningun punto
    del cuerpo con confianza >= 0,3 cayera fuera de la caja antes de limpiar.

N1neg CONTROL NEGATIVO DEL GATE, y N1 no es evidencia sin el. El mismo gate sobre una caja del MISMO
    tamanio que `Be` colocada en FONDO (IoU 0 con las cajas de los dos tiradores) en el MISMO cuadro.
    Si N1neg sale alto, N1 no mide recuperacion: mide que el modelo siempre devuelve algo.

N2  CONTROL QUE DECIDE. Para cada uno de los 650 se busca un control emparejado: una ventana
    ENTERAMENTE medida (`origen == 0` y `mask`) del MISMO clip y MISMO lado y del MISMO largo, se
    esconde el interior, se reestima la caja con la MISMA interpolacion de `redetecta.estima_cajas`
    y se vuelve a medir la pose sobre esa caja estimada. Contra la verdad que si se conoce, y
    PAREADO sobre el MISMO juego de puntos:
      `err_rescate`  mediana de ||rescatada - verdad|| sobre los puntos utiles del cuerpo, / torso
      `err_interp`   lo mismo para la pose que la tuberia inventaria HOY en ese cuadro
    torso = ||KP[5] - KP[9]||, que es la normalizacion de `valida_inventadas`.

    REGLA DE ADOPCION, FIJADA AHORA Y NO SE MUEVE. El rescate se adopta en un tramo de hueco solo si
    en ESE tramo se cumplen LAS DOS:
      (a) mediana de `err_rescate` < mediana de `err_interp`, pareado sobre los mismos casos
      (b) mediana de `err_rescate` <= 6,0 % del torso, que es el p90 ya publicado de interpolar en
          huecos <= 24, o sea lo peor que el proyecto ya entrega
    Y la cifra de decision de N2: entre los controles donde adoptar el rescate CAMBIARIA la pose de
    verdad (distancia rescate-interpolada > 6,0 % del torso), en que fraccion el rescate queda mas
    cerca de la verdad. Un rescate que solo gana donde da igual no sirve de nada.

N3  DESGLOSE POR HUECO, nunca agregado. Tramos 1 / 2-4 / 5-12 / 13-24 / >= 25, cada uno con su n.
    Motivo declarado: interpolar con hueco <= 24 cuesta 2,2 % del torso (p90 6,0 %) y por encima de
    24 NO hay medida, asi que la pregunta interesante es el tramo >= 25, que son 355 de los 650.

===============================================================================================

LA GEOMETRIA NO SE INVENTA. El recorte es el de `redetecta.redetecta_cuadro`: `MARGEN` = 0,45 sobre
la caja estimada, que es lo que ya corre en la tuberia. La caja que se le pasa a la pose es `Be`
tal cual, que es lo que la tuberia le pasa para una pose medida. Y `Be` se coge del NPZ y no se
reconstruye: reconstruir cajas ya se midio y envenena la estimacion (un clip de 99,5 % a 81,6 %).
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import time

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
CACHE_HF = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\73a45131-1b46-4cbf-95e2-29933d543565\scratchpad\hoja_seg\hf")
os.environ.setdefault("HF_HOME", str(CACHE_HF))

import cv2
import numpy as np

import capas_limpias as CL
import redetecta as RD

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SAL = PRUEBAS
CONF = 0.3
CARA_DESDE = 63
PUNTOS_MIN = 20
TRAMOS = [(1, 1), (2, 4), (5, 12), (13, 24), (25, 10 ** 9)]
SEMILLA = 20260911
CAMBIA = 0.06
"""Distancia rescate-interpolada por encima de la cual adoptar el rescate cambia la pose de verdad.
Es el 6,0 % del torso, el mismo numero que el p90 publicado de interpolar, para no meter un umbral
nuevo."""


def clips():
    """Los `poses_<clip>.npz` con el esquema de entrega. Devuelve 38, no 39."""
    fs = sorted(f for f in PRUEBAS.glob("poses_*.npz") if f.name.count(".") == 1)
    return [f for f in fs if "origen" in np.load(f, allow_pickle=True).files]


def gate(kp, sc_limpio, sc_crudo):
    """QUE DEVUELVE: (pasa, pasa_estricto, n_utiles, n_fuera_con_conf).

    No recibe la caja a proposito: el recorte de fuera-de-caja ya lo aplico `pose_en_caja` con la
    caja en coordenadas del recorte, y aqui solo se cuenta. Pasar la caja otra vez invitaria a
    compararla contra unos puntos que ya estan trasladados al cuadro completo.

    `crudo` es el mismo recuento SIN el recorte de fuera-de-caja, y hace falta para separar dos
    fracasos que se confunden: que el modelo no devuelva nada (crudo bajo) y que devuelva puntos
    pero fuera de la caja donde deberia estar el tirador (crudo alto y `util` bajo).

    `pasa` es el filtro DEL PROYECTO: >= 20 de los 63 puntos del cuerpo con confianza >= 0,3
    DESPUES de poner a cero los que caen fuera de su caja. `pasa_estricto` es la lectura literal de
    "y ningun keypoint fuera de su propia caja": lo anterior y que no hubiera ninguno fuera que
    llegara al umbral antes de limpiar. Se dan las dos porque son numeros distintos.
    """
    fin = np.isfinite(kp[:CARA_DESDE, 0]) & np.isfinite(kp[:CARA_DESDE, 1])
    util = int(((sc_limpio[:CARA_DESDE] >= CONF) & fin).sum())
    crudo = int(((sc_crudo[:CARA_DESDE] >= CONF) & fin).sum())
    fuera = int((((sc_crudo[:CARA_DESDE] >= CONF) & (sc_limpio[:CARA_DESDE] < CONF)) & fin).sum())
    return util >= PUNTOS_MIN, (util >= PUNTOS_MIN and fuera == 0), util, fuera, crudo


def pose_en_caja(cap_pose, frame, caja_norm, W, H):
    """QUE HACE: mide la pose sobre un recorte alrededor de la caja estimada, con la geometria de
    `redetecta.redetecta_cuadro`, y devuelve los puntos en coordenadas del CUADRO COMPLETO.

    El recorte es 0,45 de margen sobre la caja, que es el de la redeteccion que ya corre. La caja
    que se le pasa a la pose es la caja SIN agrandar, que es lo que la tuberia pasa para una pose
    medida; agrandarla cambiaria la escala dentro de la entrada 384x288 del modelo top-down.
    """
    x0, y0, x1, y1 = [float(v) * s for v, s in zip(caja_norm, (W, H, W, H))]
    aw, ah = x1 - x0, y1 - y0
    if not (np.isfinite([x0, y0, x1, y1]).all() and aw > 1 and ah > 1):
        return None
    cx0 = int(max(0, x0 - RD.MARGEN * aw))
    cy0 = int(max(0, y0 - RD.MARGEN * ah))
    cx1 = int(min(W, x1 + RD.MARGEN * aw))
    cy1 = int(min(H, y1 + RD.MARGEN * ah))
    if cx1 - cx0 < 24 or cy1 - cy0 < 24:
        return None
    sub = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1, ::-1])
    caja_rec = [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]
    kp, sc = cap_pose.pose(sub, [caja_rec], limpiar=False)[0]
    _, sc_l = CL.limpia_keypoints(kp, sc, caja_rec)
    kp = kp.copy()
    kp[:, 0] += cx0
    kp[:, 1] += cy0
    return kp.astype("float64"), sc_l, sc, caja_rec


def caja_fondo(cajas_ocupadas, caja_norm, W, H):
    """QUE HACE: una caja del MISMO tamanio colocada donde NO hay ningun tirador, para el control
    negativo del gate. Sin esto, "pasa el filtro el X %" no distingue recuperar de alucinar.

    Se barre una rejilla de centros y se coge el que, teniendo solape CERO con todas las cajas
    ocupadas del cuadro, queda mas lejos de todas ellas.
    """
    w = (caja_norm[2] - caja_norm[0]) * W
    h = (caja_norm[3] - caja_norm[1]) * H
    if not (np.isfinite([w, h]).all() and w > 1 and h > 1) or w >= W or h >= H:
        return None
    oc = []
    for b in cajas_ocupadas:
        if np.isfinite(b).all():
            oc.append([b[0] * W, b[1] * H, b[2] * W, b[3] * H])
    mejor, mejor_d = None, -1.0
    for fx in np.linspace(0.0, 1.0, 13):
        for fy in np.linspace(0.0, 1.0, 9):
            cx = w / 2 + fx * (W - w)
            cy = h / 2 + fy * (H - h)
            c = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            if any(RD._iou(c, o) > 0.0 for o in oc):
                continue
            d = min(abs(cx - (o[0] + o[2]) / 2) + abs(cy - (o[1] + o[3]) / 2) for o in oc) if oc else 0.0
            if d > mejor_d:
                mejor, mejor_d = c, d
    if mejor is None:
        return None
    return [mejor[0] / W, mejor[1] / H, mejor[2] / W, mejor[3] / H]


def casos_reales(origen, hueco, mask, T):
    """Los lados-cuadro inventados, cada uno con sus dos anclas y su `t` de interpolacion."""
    out = []
    for lado in (0, 1):
        hay = np.nonzero((origen[:, lado] == 0) & mask)[0]
        for k in range(T):
            if origen[k, lado] == 0 or not mask[k]:
                continue
            ant, pos = hay[hay < k], hay[hay > k]
            if not (len(ant) and len(pos)):
                continue
            a, b = int(ant[-1]), int(pos[0])
            out.append(dict(k=k, lado=lado, a=a, b=b, h=int(hueco[k, lado]),
                            hcalc=b - a - 1, t=(k - a) / float(b - a)))
    return out


def busca_control(rng, origen, mask, T, lado, a, b, k, usados):
    """QUE HACE: busca una ventana ENTERAMENTE medida del mismo lado y del MISMO largo que el hueco
    real, para poder esconder su interior y comparar contra la verdad.

    Se exige que TODA la ventana [a', b'] sea `origen == 0` y `mask`, no solo los extremos: asi el
    control es de verdad "tapar una pose medida y buena", que es lo que pidio el usuario, y el cuadro
    tapado tiene verdad conocida.
    """
    L = b - a
    off = k - a
    bueno = (origen[:, lado] == 0) & mask
    if L + 1 > T:
        return None
    acum = np.concatenate([[0], np.cumsum(bueno.astype("int64"))])
    cand = [s for s in range(0, T - L) if acum[s + L + 1] - acum[s] == L + 1]
    cand = [s for s in cand if (s, lado) not in usados] or cand
    if not cand:
        return None
    s = int(cand[rng.integers(len(cand))])
    return dict(a=s, b=s + L, k=s + off, lado=lado)


def err_pareado(kp_v, sc_v, kp_a, kp_b_):
    """QUE DEVUELVE: (err_a, err_b, torso, n) con las dos estimaciones juzgadas sobre el MISMO juego
    de puntos, que es lo que hace que la comparacion sea pareada.

    La cara se excluye a proposito: 245 de los 308 puntos y el proyecto no los usa, asi que
    promediarlos taparia el error de lo que si importa. Es el criterio de `valida_inventadas`.
    """
    ok = ((sc_v[:CARA_DESDE] >= CONF) & np.isfinite(kp_v[:CARA_DESDE, 0])
          & np.isfinite(kp_a[:CARA_DESDE, 0]) & np.isfinite(kp_b_[:CARA_DESDE, 0]))
    if int(ok.sum()) < 8:
        return None
    da = np.linalg.norm(kp_a[:CARA_DESDE][ok] - kp_v[:CARA_DESDE][ok], axis=1)
    db = np.linalg.norm(kp_b_[:CARA_DESDE][ok] - kp_v[:CARA_DESDE][ok], axis=1)
    dab = np.linalg.norm(kp_a[:CARA_DESDE][ok] - kp_b_[:CARA_DESDE][ok], axis=1)
    torso = float("nan")
    if sc_v[5] >= CONF and sc_v[9] >= CONF:
        tt = float(np.linalg.norm(kp_v[5] - kp_v[9]))
        torso = tt if tt > 1 else float("nan")
    return (float(np.median(da)), float(np.median(db)), float(np.median(dab)), torso,
            int(ok.sum()))


def main():
    fs = clips()
    solo = os.environ.get("SOLO")
    if solo:
        fs = [f for f in fs if f.name[6:-4] in solo.split(",")]
    print("clips con esquema de entrega: %d" % len(fs))
    cap_pose = CL.Capas(["pose"])
    rng = np.random.default_rng(SEMILLA)
    filas = []
    t0 = time.perf_counter()
    for idx, f in enumerate(fs):
        cid = f.name[6:-4]
        d = np.load(f, allow_pickle=True)
        KP, SC, B, Be = d["KP"], d["SC"], d["B"], d["Be"]
        origen, hueco, mask = d["origen"], d["hueco"], d["mask"]
        W, H, T = int(d["W"]), int(d["H"]), int(d["T"])
        vid = str(d["video"])
        reales = casos_reales(origen, hueco, mask, T)
        usados = set()
        for r in reales:
            c = busca_control(rng, origen, mask, T, r["lado"], r["a"], r["b"], r["k"], usados)
            r["ctrl"] = c
            if c:
                usados.add((c["a"], c["lado"]))
        need = set()
        for r in reales:
            need.add(r["k"])
            if r["ctrl"]:
                need.add(r["ctrl"]["k"])
        frames = {}
        cap = cv2.VideoCapture(vid)
        for k in range(T):
            ok, im = cap.read()
            if not ok:
                break
            if k in need:
                frames[k] = im
        cap.release()
        ncf = 0
        for r in reales:
            k, lado = r["k"], r["lado"]
            fila = dict(clip=cid, k=k, lado=lado, h=r["h"], hcalc=r["hcalc"], t=r["t"],
                        W=W, H=H, real_ok=0, real_util=-1, real_fuera=-1, real_estricto=0,
                        neg_ok=-1, neg_util=-1, ctrl="", cerr_res=float("nan"),
                        cerr_int=float("nan"), cd_res_int=float("nan"), ctorso=float("nan"),
                        cn=-1, cgate=-1, cgate_util=-1, cgate_estricto=-1,
                        real_d_res_int=float("nan"), real_torso=float("nan"),
                        real_crudo=-1, neg_crudo=-1, cgate_crudo=-1,
                        be_dentro=float("nan"), be_alto=float("nan"))
            bb = Be[k, lado]
            if np.isfinite(bb).all():
                ax = max(0.0, min(1.0, bb[2]) - max(0.0, bb[0]))
                ay = max(0.0, min(1.0, bb[3]) - max(0.0, bb[1]))
                tot = max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1e-9)
                fila["be_dentro"] = float(ax * ay / tot)
                fila["be_alto"] = float((bb[3] - bb[1]) * H)
            im = frames.get(k)
            if im is not None:
                p = pose_en_caja(cap_pose, im, Be[k, lado], W, H)
                if p is not None:
                    kp, sc_l, sc_c, cr = p
                    g, ge, nu, nf, ncr = gate(kp, sc_l, sc_c)
                    fila.update(real_ok=int(g), real_estricto=int(ge), real_util=nu, real_fuera=nf,
                                     real_crudo=ncr)
                    kpi = KP[k, lado].astype("float64")
                    sci = SC[k, lado]
                    okp = ((sci[:CARA_DESDE] >= CONF) & np.isfinite(kpi[:CARA_DESDE, 0])
                           & np.isfinite(kp[:CARA_DESDE, 0]))
                    if int(okp.sum()) >= 8:
                        dd = np.linalg.norm(kp[:CARA_DESDE][okp] - kpi[:CARA_DESDE][okp], axis=1)
                        fila["real_d_res_int"] = float(np.median(dd))
                        if sci[5] >= CONF and sci[9] >= CONF:
                            tt = float(np.linalg.norm(kpi[5] - kpi[9]))
                            if tt > 1:
                                fila["real_torso"] = tt
                cf = caja_fondo([Be[k, 0], Be[k, 1], B[k, 0], B[k, 1]], Be[k, lado], W, H)
                if cf is not None:
                    pn = pose_en_caja(cap_pose, im, cf, W, H)
                    if pn is not None:
                        kpn, scn_l, scn_c, crn = pn
                        gn, _gen, nun, _nfn, ncrn = gate(kpn, scn_l, scn_c)
                        fila.update(neg_ok=int(gn), neg_util=nun, neg_crudo=ncrn)
            c = r["ctrl"]
            if c:
                imc = frames.get(c["k"])
                if imc is not None and np.isfinite(B[c["a"], lado]).all() and np.isfinite(B[c["b"], lado]).all():
                    tt = (c["k"] - c["a"]) / float(c["b"] - c["a"])
                    bce = (1 - tt) * B[c["a"], lado] + tt * B[c["b"], lado]
                    p = pose_en_caja(cap_pose, imc, bce, W, H)
                    if p is not None:
                        kpr, scr_l, scr_c, crc = p
                        g, ge, nu, _nf, ncrc = gate(kpr, scr_l, scr_c)
                        fila.update(cgate=int(g), cgate_util=nu, cgate_estricto=int(ge),
                                    cgate_crudo=ncrc)
                        kpint = ((1 - tt) * KP[c["a"], lado].astype("float64")
                                 + tt * KP[c["b"], lado].astype("float64"))
                        e = err_pareado(KP[c["k"], lado].astype("float64"), SC[c["k"], lado],
                                        kpr, kpint)
                        if e is not None:
                            fila.update(ctrl="%d:%d:%d" % (c["a"], c["k"], c["b"]),
                                        cerr_res=e[0], cerr_int=e[1], cd_res_int=e[2],
                                        ctorso=e[3], cn=e[4])
                            ncf += 1
            filas.append(fila)
        print("[%2d/%2d] %-46s inventadas %4d  controles con verdad %4d  %5.1f s"
              % (idx + 1, len(fs), cid[:46], len(reales), ncf, time.perf_counter() - t0))
        sys.stdout.flush()
    with open(SAL / "rescate_casos.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("escritas %d filas en %s" % (len(filas), SAL / "rescate_casos.csv"))
    print("ms por llamada de pose: %.1f  (n = %d)"
          % (cap_pose.ms["pose"] / max(cap_pose.n["pose"], 1), cap_pose.n["pose"]))
    with open(SAL / "rescate_meta.json", "w") as fh:
        json.dump(dict(clips=len(fs), casos=len(filas), semilla=SEMILLA,
                       margen=RD.MARGEN, conf=CONF, puntos_min=PUNTOS_MIN,
                       ms_pose=cap_pose.ms["pose"] / max(cap_pose.n["pose"], 1),
                       n_pose=cap_pose.n["pose"]), fh, indent=1)


if __name__ == "__main__":
    main()
