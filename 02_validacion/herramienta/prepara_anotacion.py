"""PREPARA LO QUE EL USUARIO NECESITA PARA ANOTAR FRONTERAS: las tiras y el CSV vacio.

El usuario ofrecio: *"igual me pides que te diga donde empieza y acaba una marcha, o un romper, o un
fondo, para que lo aprendas mejor"*. Y hace falta, porque hay un hueco: lo medido es DETECCION
(recall 85-90 %) y CLASIFICACION (5 de 6), pero **nadie ha medido si las FRONTERAS son correctas**.
Un evento puede existir, ser del tipo correcto, y empezar 8 cuadros tarde.

EL CONVENIO DE FRONTERA ESTA DECLARADO ANTES en `02_validacion/pista/BOUNDARY_CONVENTION.md` y va
ESCRITO EN LA PROPIA IMAGEN. Sin eso, el usuario anotaria "cuando empieza a moverse" y el detector
mide "cuando despega", que difieren 2-4 cuadros, y la diferencia se leeria como error.

QUE SACA, por accion elegida:

  una tira con TODOS los cuadros de la accion, uno a uno, numerados, con el marcador de cada pie en
  un color distinto, sobre el fotograma **ESTABILIZADO** para que el suelo no se mueva. Asi se puede
  leer el numero del cuadro en que la suela deja de tocar, que es la frontera del convenio.

  y una fila en `anotar/fronteras.csv` con las columnas ya puestas y el candidato del detector al
  lado, para que se pueda corregir en vez de escribir de cero.

**EL CANDIDATO DEL DETECTOR VA EN COLUMNAS APARTE Y NO EN LAS QUE SE RELLENAN.** Ponerlo en la misma
casilla sesgaria la anotacion hacia lo que ya detecto, que es la forma mas facil de fabricar un
acuerdo falso.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista as PI
import posicion as PO

SALIDA = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista\anotar")
PIE_COL = ((255, 220, 0), (255, 0, 220))
VIA_TIPO = {"marcha": "march", "romper": "retreat", "fondo": "lunge", "alto": "halt",
            "brazo": "arm", "bote": "balestra", "accion": "action", "?": "?"}
"""Traduccion SOLO PARA MOSTRAR de los tipos. `alto` -> `halt` y NUNCA `parry`."""
POR_FILA = 8
MARGEN = 4
"""Cuadros de margen antes y despues de la accion. Hacen falta para poder ver el contacto ANTES del
despegue: si la tira empieza en el despegue, no se puede juzgar que ese es el despegue."""
COLUMNAS = ["clip", "lado", "id_accion", "cuadro_primero_tira", "cuadro_ultimo_tira",
            "ANOTA_cuadro_inicio", "ANOTA_cuadro_fin", "ANOTA_tipo", "ANOTA_pie", "ANOTA_notas",
            "detector_tipo", "detector_inicio", "detector_fin", "detector_pie"]


def tira_accion(clip, res, poses, lado, i, j, idc, cand):
    """QUE HACE: la tira de una accion, cuadro a cuadro y numerada, sobre el cuadro estabilizado."""
    r, Mk, frames = res["_r"], res["_Mk"], res["_frames"]
    W, H = r["W"], r["H"]
    k0 = max(int(np.nonzero(r["mask"])[0][0]), i - MARGEN)
    k1 = min(int(np.nonzero(r["mask"])[0][-1]), j + MARGEN, len(frames) - 1)
    ks = list(range(k0, k1 + 1))
    if len(ks) < 4:
        return None
    caj = []
    for k in ks:
        if not r["pres"][k, lado] or not np.isfinite(r["B"][k, lado]).all():
            continue
        b = r["B"][k, lado]
        p0 = Mk[k] @ np.array([b[0] * W, b[1] * H, 1.0])
        p1 = Mk[k] @ np.array([b[2] * W, b[3] * H, 1.0])
        caj.append([p0[0] / p0[2], p0[1] / p0[2], p1[0] / p1[2], p1[1] / p1[2]])
    if len(caj) < 4:
        return None
    caj = np.asarray(caj)
    alt = float(np.median(caj[:, 3] - caj[:, 1]))
    cx = float(np.median(0.5 * (caj[:, 0] + caj[:, 2])))
    by0 = float(np.median(caj[:, 1]))
    by1 = float(np.max(caj[:, 3]))
    ancho = 1.35 * alt
    trozos = []
    for k in ks:
        im = cv2.warpAffine(frames[k], Mk[k][:2], (W, H), flags=cv2.INTER_LINEAR,
                            borderValue=(20, 20, 20))
        for t, p in enumerate(PO._pie_pose(poses["KP"], poses["SC"], k, lado)
                              if k < poses["KP"].shape[0] else [None, None]):
            if p is None:
                continue
            q = Mk[k] @ np.array([p[0], p[1], 1.0])
            cv2.drawMarker(im, (int(q[0] / q[2]), int(q[1] / q[2])), PIE_COL[t],
                           cv2.MARKER_TILTED_CROSS, 16, 2)
        x0, x1 = int(cx - 0.5 * ancho), int(cx + 0.5 * ancho)
        y0, y1 = int(by0 + 0.42 * alt), int(by1 + 0.18 * alt)
        pad = np.full((max(y1 - y0, 10), max(x1 - x0, 10), 3), 20, "uint8")
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            pad[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)] = \
                im[sy0:sy1, sx0:sx1]
        c = cv2.resize(pad, (300, int(300.0 * pad.shape[0] / pad.shape[1])),
                       interpolation=cv2.INTER_CUBIC)
        """EL NUMERO DE CUADRO VA GRANDE Y EN CADA CASILLA, porque es lo que el usuario tiene que
        escribir. Si hay que ampliar la imagen para leerlo, la anotacion se hace mal."""
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 26), (0, 0, 0), -1)
        cv2.putText(c, "%d" % k, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.rectangle(c, (0, 0), (c.shape[1] - 1, c.shape[0] - 1), (110, 110, 110), 1)
        trozos.append(c)
    h = min(t.shape[0] for t in trozos)
    fil = []
    for a in range(0, len(trozos), POR_FILA):
        f = [t[:h] for t in trozos[a:a + POR_FILA]]
        while len(f) < POR_FILA:
            f.append(np.full_like(f[0], 20))
        fil.append(np.hstack(f))
    cuerpo = np.vstack(fil)
    cab = np.zeros((92, cuerpo.shape[1], 3), "uint8")
    lin = ["%s  side %d (%s)  action %s  ·  frames %d to %d"
           % (clip, lado, "left" if lado == 0 else "right", idc, k0, k1),
           "CONVENTION: the START is the frame where the foot NO LONGER TOUCHES the floor; the END,"
           " the frame where it TOUCHES AGAIN.",
           "If a frame is doubtful, write the LOWER one. If the foot is not visible, write ?"
           "  ·  cyan = one foot, magenta = the other",
           "detector says: %s  start %s  end %s   (do not trust it, correct it)"
           % (VIA_TIPO.get(cand.get("tipo"), cand.get("tipo", "-")),
              cand.get("inicio", "-"), cand.get("fin", "-"))]
    for t, s in enumerate(lin):
        cv2.putText(cab, s, (8, 20 + 22 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52 if t == 0 else 0.44, (255, 255, 255) if t != 3 else (150, 220, 255),
                    1, cv2.LINE_AA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    p = SALIDA / ("accion_%s_l%d_c%04d.png" % (clip, lado, i))
    cv2.imwrite(str(p), np.vstack([cab, cuerpo]))
    return {"ruta": p, "k0": k0, "k1": k1}


def una(clip, lado, max_acciones=7):
    """QUE HACE: las tiras y las filas de un tirador.

    SE ELIGEN LAS ACCIONES MAS SEPARADAS EN EL TIEMPO y no las primeras, para que la muestra cubra el
    asalto entero: los primeros segundos de un clip son puesta en guardia y anotar sólo eso daría una
    cifra de frontera de los gestos más lentos.
    """
    res = PI.analiza(clip)
    poses = PO.carga_poses(clip)
    if poses is None or not res.get("_r"):
        print("%-38s sin pose o sin pareja" % clip)
        return []
    if res.get("ok"):
        ap = PO.apoyos(res, poses)
        fps = PO._fps(PI.indice_videos()[clip])
        ev = [e for e in PO.eventos(ap, res["_r"], fps, poses, res=res)
              if e["tipo"] in ("marcha", "romper", "fondo") and e.get("lado") == lado]
    else:
        ev = []
    co = PO.contactos(res, poses, lado)
    cands = []
    if ev:
        ev.sort(key=lambda e: e["inicio"])
        paso = max(1, len(ev) // max_acciones)
        for e in ev[::paso][:max_acciones]:
            cands.append({"tipo": e["tipo"], "inicio": e["inicio"], "fin": e["fin"],
                          "pie": e.get("pies_movidos")})
    elif co.get("ok"):
        """SIN HOMOGRAFIA no hay eventos metricos, asi que los candidatos salen de los VUELOS de la
        primitiva de contactos, que no necesita escala."""
        for j in (0, 1):
            a = co["apoyo"][:, j]
            k = 0
            while k < len(a):
                if a[k]:
                    k += 1
                    continue
                e = k
                while e + 1 < len(a) and not a[e + 1]:
                    e += 1
                if e - k + 1 >= PO.VUELO_MIN:
                    cands.append({"tipo": "?", "inicio": k, "fin": e, "pie": j})
                k = e + 1
        cands.sort(key=lambda z: z["inicio"])
        paso = max(1, len(cands) // max_acciones)
        cands = cands[::paso][:max_acciones]
    filas = []
    for n, cd in enumerate(cands, 1):
        idc = "%s_l%d_%02d" % (clip[:12], lado, n)
        t = tira_accion(clip, res, poses, lado, cd["inicio"], cd["fin"], idc, cd)
        if t is None:
            continue
        filas.append({"clip": clip, "lado": lado, "id_accion": idc,
                      "cuadro_primero_tira": t["k0"], "cuadro_ultimo_tira": t["k1"],
                      "ANOTA_cuadro_inicio": "", "ANOTA_cuadro_fin": "", "ANOTA_tipo": "",
                      "ANOTA_pie": "", "ANOTA_notas": "",
                      "detector_tipo": cd["tipo"], "detector_inicio": cd["inicio"],
                      "detector_fin": cd["fin"], "detector_pie": cd.get("pie")})
    print("%-38s lado %d: %d tiras" % (clip, lado, len(filas)))
    return filas


if __name__ == "__main__":
    filas = []
    for a in sys.argv[1:]:
        c, _, l = a.partition(":")
        filas += una(c, int(l or 0))
    if filas:
        SALIDA.mkdir(parents=True, exist_ok=True)
        with open(SALIDA / "fronteras.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)
        print("\n%d acciones para anotar en %s" % (len(filas), SALIDA))
        print("convenio en 02_validacion/pista/BOUNDARY_CONVENTION.md")
