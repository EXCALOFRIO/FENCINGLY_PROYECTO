"""ETAPA E, paso 29: LA CASCADA DE TRES RONDAS de la pose, con los keypoints GUARDADOS.

LAS TRES RONDAS, que es lo que hoy pasa de verdad cuando un tirador no se detecta:

    RONDA 1   YOLO sobre el cuadro, y si falla, YOLO sobre un recorte (`redetecta`, `tipo = 3`).
              Cuando tampoco, el cuadro queda SIN POSE: es el `origen != 0` que se ve en violeta.
    RONDA 2   la cabeza `pose` de Sapiens 2, **top-down**, dirigida a la caja estimada `Be`. Es una
              via DISTINTA y no un reintento: un modelo top-down no necesita que un detector
              confirme a la persona, se le da la caja y saca puntos. Idea del usuario.
    RONDA 3   interpolar entre las dos anclas medidas, que es lo que la tuberia entrega hoy.

POR QUE ESTE FICHERO EXISTE Y NO BASTA `rescate_casos.csv`. Aquel CSV tiene los 650 casos con 30
columnas de **metricas** y **ningun keypoint**: con el se puede decidir si el rescate vale, pero no
se puede pintar ni entregar la pose rescatada. Aqui se vuelve a pasar la ronda 2 guardando los
puntos, que es lo que cuesta los 233 s de GPU de la tirada completa.

SOLO SE RESCATA CON HUECO <= 24, Y ESO NO ES PRUDENCIA: ESTA MEDIDO. Del informe
`rescate_decision.txt`, sobre los mismos 650 casos:

    hueco        inventadas   pasan el gate    el gate sobre FONDO VACIO (falso positivo)
    <= 24              295        50 (16,9 %)   0,0-3,8 %
    >= 25              355         2 (0,6 %)    **45 de 355 (12,7 %)**

O sea que por encima de 24 cuadros el gate pasa MENOS veces con un tirador delante (0,6 %) que con
el fondo vacio (12,7 %). Ahi no hay a quien medir: lo poco que pasa no se distingue de una
alucinacion del modelo top-down, y la p estimada de «hay tirador de verdad en la caja» sale 0,0 %.
Rescatar en ese tramo seria cambiar una pose inventada y MARCADA por una pose inventada y **sin
marcar**, que es peor.

Y EN EL TRAMO QUE SI SE ADOPTA, lo que se evita, con el error de interpolar medido en el control
pareado (ventanas enteramente medidas, escondidas y remedidas):

    hueco     rescata   err_interp   err_rescate
    1              13        2,19 %       0,88 %
    2-4           12        2,87 %       1,11 %
    5-12          12        6,93 %       1,75 %
    13-24         13       16,46 %       1,88 %

LA PROCEDENCIA SE PUBLICA COMO `origen = 3`, un codigo NUEVO, y no se reetiqueta como 0:

    0  pose MEDIDA en la ronda 1        1  INTERPOLADA (ronda 3)
    2  SOSTENIDA (no se genera ya)      3  MEDIDA EN LA RONDA 2 sobre la caja ESTIMADA

Un 3 no vale lo mismo que un 0 y hay que poder separarlos: el 0 se midio sobre una caja que un
detector confirmo, y el 3 sobre una caja **interpolada**, asi que su error hereda el de la caja. Es
la misma disciplina que `campo_origen` en el cono.

QUE ESCRIBE, Y POR QUE UN FICHERO APARTE. `rescate_poses_<clip>.npz`, con el MISMO indexado
`(T, 2, ...)` que `poses_<clip>.npz`, **sin tocar el NPZ entregado**. Reescribir en sitio un dato ya
entregado es destructivo y no es mia esa decision; con el sidecar la procedencia «se puede leer
despues», que es lo que hacia falta, y `aplica()` deja el array fusionado en memoria para quien lo
quiera. Adoptarlo dentro de `poses_*.npz` es una linea cuando el usuario lo autorice.

USO:  python e29_tres_rondas.py            los 38 clips con esquema de entrega
      SOLO=clip1,clip2 python e29_tres_rondas.py
      con el interprete de transformers + torch + CUDA (el de `capas_limpias.Capas`)
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

CODIGO = pathlib.Path(__file__).parent
sys.path.insert(0, str(CODIGO))

import numpy as np

import cv2
import capas_limpias as CL
import rescate as RS

PRUEBAS = RS.PRUEBAS
HUECO_MAX = 24
"""El unico tramo donde la medida sostiene el rescate. Ver el docstring: con hueco >= 25 el gate pasa
0,6 % con tirador y 12,7 % sobre fondo vacio, o sea que no discrimina. **No es un umbral barrido**:
24 es el hueco mas largo para el que existe error de interpolar medido, que es el mismo corte que ya
usaba `galeria_inventadas.HUECO_MEDIDO`."""
ORIGEN_RESCATADA = 3


def casos_del_clip(z, solo_hueco_medido=True):
    """Los lados-cuadro inventados con sus dos anclas, filtrados al tramo con medida.

    Se reutiliza `rescate.casos_reales` tal cual para no tener dos definiciones de «caso»: aquella
    exige ancla medida a los dos lados, que es lo que hace interpolable el cuadro, y devuelve el
    hueco calculado `hcalc = b - a - 1` al lado del `hueco` del NPZ. Se filtra por `hcalc`, que es
    el que se puede recomputar, y se avisa si los dos no coinciden.
    """
    origen, hueco, mask = z["origen"], z["hueco"], z["mask"]
    T = int(z["T"])
    cs = RS.casos_reales(origen, hueco, mask, T)
    discrepa = sum(1 for r in cs if r["h"] != r["hcalc"])
    if solo_hueco_medido:
        cs = [r for r in cs if r["hcalc"] <= HUECO_MAX]
    return cs, discrepa


def un_clip(cap_pose, f, solo_hueco_medido=True):
    """Ronda 2 sobre cada caso del clip, guardando los puntos. Devuelve el dict del NPZ y un resumen.

    NO SE VUELVE A CALCULAR NI LA CAJA NI EL CONTROL NEGATIVO. `Be` se lee del NPZ -- reconstruir
    cajas ya se midio y envenena la estimacion, de 99,5 % a 81,6 % en un clip -- y el control
    negativo del gate (la caja en fondo vacio) ya esta medido en `rescate_casos.csv` por tramo: es
    el numero que autoriza este tramo y no hace falta repetirlo para entregar los puntos.
    """
    cid = f.name[len("poses_"):-len(".npz")]
    z = np.load(f, allow_pickle=True)
    KP, SC, Be = z["KP"], z["SC"], z["Be"]
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
    vid = str(z["video"])
    cs, discrepa = casos_del_clip(z, solo_hueco_medido)
    N = KP.shape[2]
    rKP = np.full((T, 2, N, 2), np.nan, "float32")
    rSC = np.zeros((T, 2, N), "float32")
    pasa = np.zeros((T, 2), bool)
    util = np.full((T, 2), -1, "int16")
    hue = np.full((T, 2), -1, "int32")
    if not cs:
        return None, {"clip": cid, "casos": 0, "rescatadas": 0, "discrepa_hueco": discrepa}
    need = set(r["k"] for r in cs)
    frames = {}
    cap = cv2.VideoCapture(vid)
    for k in range(T):
        ok, im = cap.read()
        if not ok:
            break
        if k in need:
            frames[k] = im
    cap.release()
    n_ok = 0
    for r in cs:
        k, lado = r["k"], r["lado"]
        hue[k, lado] = r["hcalc"]
        if k not in frames:
            continue
        bb = Be[k, lado]
        if not np.isfinite(bb).all():
            continue
        out = RS.pose_en_caja(cap_pose, frames[k], bb, W, H)
        if out is None:
            continue
        kp, sc_l, sc_crudo, _caja = out
        ok_gate, _estricto, n_util, _fuera, _crudo = RS.gate(kp, sc_l, sc_crudo)
        rKP[k, lado] = kp[:N]
        rSC[k, lado] = sc_l[:N]
        util[k, lado] = n_util
        pasa[k, lado] = bool(ok_gate)
        """SE GUARDAN TAMBIEN LAS QUE NO PASAN EL GATE. Son las que el panel 2 de la galeria tiene
        que poder ensenar diciendo `FALLA el filtro`: si solo se guardaran las buenas, la figura
        ensenaria solo los aciertos y no se podria juzgar el modo de fallo."""
        n_ok += int(ok_gate)
    org3 = z["origen"].copy()
    org3[pasa] = ORIGEN_RESCATADA
    d = {"KP_rescate": rKP, "SC_rescate": rSC, "pasa_gate": pasa, "n_util": util,
         "hueco_caso": hue, "origen_tres_rondas": org3, "origen_original": z["origen"],
         "W": W, "H": H, "T": T, "video": vid, "hueco_max": HUECO_MAX,
         "leyenda": LEYENDA}
    return d, {"clip": cid, "casos": len(cs), "rescatadas": n_ok, "discrepa_hueco": discrepa}


LEYENDA = (
    "KP_rescate/SC_rescate: la pose de la RONDA 2, la cabeza pose de Sapiens 2 top-down sobre la "
    "caja ESTIMADA Be. NaN donde no se intento. pasa_gate: si esa pose pasa el filtro del proyecto "
    "(>= 20 de los 63 puntos del cuerpo con confianza >= 0,3 tras poner a cero los de fuera de su "
    "caja). n_util: ese recuento, -1 si no se intento. hueco_caso: largo del hueco del caso, -1 si "
    "no es un caso. origen_tres_rondas: 0 medida en la ronda 1, 1 interpolada, 2 sostenida, "
    "**3 MEDIDA EN LA RONDA 2 sobre caja estimada**. Un 3 NO vale lo mismo que un 0: el 0 se midio "
    "sobre una caja que un detector confirmo y el 3 sobre una caja interpolada, asi que hereda el "
    "error de la caja. Solo se intenta con hueco <= 24: con hueco >= 25 el gate pasa 0,6 % con "
    "tirador y 12,7 % sobre fondo vacio, o sea que no discrimina.")


def aplica(clip):
    """QUE DEVUELVE: `(KP, SC, origen)` de `poses_<clip>.npz` con la ronda 2 ya metida donde paso.

    Es la funcion que hace innecesario reescribir el NPZ entregado. Quien quiera la cascada de tres
    rondas la llama; quien quiera el dato tal como se publico abre el NPZ y no pasa por aqui.
    """
    z = np.load(PRUEBAS / ("poses_%s.npz" % clip), allow_pickle=True)
    KP, SC, org = z["KP"].copy(), z["SC"].copy(), z["origen"].copy()
    f = PRUEBAS / ("rescate_poses_%s.npz" % clip)
    if not f.is_file():
        return KP, SC, org
    r = np.load(f, allow_pickle=True)
    m = r["pasa_gate"]
    if m.shape != org.shape:
        raise ValueError("%s: pasa_gate %s no cuadra con origen %s" % (clip, m.shape, org.shape))
    ks, ls = np.nonzero(m)
    KP[ks, ls] = r["KP_rescate"][ks, ls]
    SC[ks, ls] = r["SC_rescate"][ks, ls]
    org[ks, ls] = ORIGEN_RESCATADA
    return KP, SC, org


def main():
    fs = RS.clips()
    solo = os.environ.get("SOLO")
    if solo:
        fs = [f for f in fs if f.name[6:-4] in solo.split(",")]
    print("=" * 104)
    print("LA CASCADA DE TRES RONDAS  ·  ronda 2 con los keypoints GUARDADOS  ·  hueco <= %d"
          % HUECO_MAX)
    print("=" * 104)
    print("  clips con esquema de entrega: %d" % len(fs))
    print("  hueco >= 25 NO se intenta: ahi el gate pasa 0,6 %% con tirador y 12,7 %% sobre fondo")
    print("  vacio, o sea que no discrimina (`rescate_decision.txt`).")
    print()
    cap_pose = CL.Capas(["pose"])
    res, t0 = [], time.perf_counter()
    for i, f in enumerate(fs):
        d, s = un_clip(cap_pose, f)
        if d is not None:
            np.savez_compressed(PRUEBAS / ("rescate_poses_%s.npz" % s["clip"]), **d)
        res.append(s)
        print("  [%2d/%2d] %-44s casos %3d  rescatadas %3d  %6.1f s"
              % (i + 1, len(fs), s["clip"][:44], s["casos"], s["rescatadas"],
                 time.perf_counter() - t0))
    tc = sum(r["casos"] for r in res)
    tr = sum(r["rescatadas"] for r in res)
    td = sum(r["discrepa_hueco"] for r in res)
    print()
    print("  casos con hueco <= %d: %d  ·  rescatadas (pasan el gate): **%d (%.1f %%)**"
          % (HUECO_MAX, tc, tr, 100.0 * tr / max(tc, 1)))
    print("  casos donde `hueco` del NPZ discrepa del recalculado: %d" % td)
    print("  la cifra de referencia de `rescate_decision.txt` es 50 de 295 (16,9 %): si esto se")
    print("  aparta mucho, algo ha cambiado entre las dos tiradas y hay que mirarlo antes de usarlo.")
    (PRUEBAS / "rescate_tres_rondas.json").write_text(json.dumps(
        {"hueco_max": HUECO_MAX, "origen_rescatada": ORIGEN_RESCATADA, "clips": res,
         "casos": tc, "rescatadas": tr, "leyenda": LEYENDA},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("  guardado %s" % (PRUEBAS / "rescate_tres_rondas.json"))


if __name__ == "__main__":
    main()
