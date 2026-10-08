"""ETAPA E, paso 16: la VERDAD-TERRENO del contacto de hojas, anotada A CIEGAS.

POR QUE A CIEGAS, Y POR QUE ESO NO ES UN ADORNO. El paso 15 propone el solape de conos como proxy
del contacto entre hojas. Para medir si sirve hace falta saber en que cuadros hay contacto de verdad,
y no hay ninguna anotacion de paradas en el proyecto: el detector de prioridad se cerro con **0
paradas anotadas**. Asi que la anotacion la hace este agente mirando el video.

Y ahi esta el peligro: si al anotar se ve el cono pintado, o el nombre del clip, o el valor de
`peso_max`, la anotacion deja de ser independiente de lo que se quiere validar y la medida se vuelve
circular. El proyecto ya tiene el escarmiento de sembrar SAM2 en un senuelo dentro del cuerpo y que
el criterio de consistencia temporal lo aprobara igual (10 de 12).

ASI QUE ESTE PASO GENERA LAS LAMINAS SIN NINGUNA PISTA:
  - **sin conos**, sin esqueleto y sin ninguna marca de la geometria
  - **sin el nombre del clip** ni el numero de cuadro visibles
  - **en orden aleatorio**, con una semilla fija para poder repetirlo
  - identificadas solo por un ID opaco
La clave (`ID -> clip, cuadro, peso_max`) se escribe en un fichero APARTE que no hay que abrir hasta
despues de anotar. El paso 17 es el que une anotacion y clave y mide.

EL MUESTREO, DECLARADO ANTES DE MIRAR. Se estratifica por `peso_max` del paso 15, mitad y mitad:
  - **la mitad con `peso_max` alto** (los candidatos a contacto segun el proxy)
  - **la mitad con `peso_max` = 0** (los que el proxy dice que NO)
Estratificar asi es lo correcto para medir acierto Y falsos negativos con pocas muestras: un muestreo
uniforme daria casi solo cuadros sin contacto, porque el contacto es raro, y no permitiria medir el
acierto en los positivos. **Y tiene un precio que hay que declarar: las tasas que salgan NO son
prevalencias del corpus**, son condicionadas al estrato, y para llevarlas a tasas del corpus hay que
repesar por el tamano de cada estrato.

QUE SE MIRA EN LA LAMINA. Un recorte que cubre las dos manos con margen, ampliado, para poder ver si
las hojas se tocan. La respuesta que hay que dar es una de tres:
    `si`    las dos hojas estan en contacto o cruzadas en este cuadro
    `no`    no se tocan
    `nose`  no se puede decidir (desenfoque, oclusion, hoja invisible)
El `nose` es obligatorio que exista: sin el, la duda se reparte entre `si` y `no` y contamina las dos.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL

import cv2

PRUEBAS = CL.PRUEBAS
SAL = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\contacto_hojas")
SAL.mkdir(parents=True, exist_ok=True)
N_POR_ESTRATO = 11
SEMILLA = 7
ZOOM = 2.2
SOLO_1080 = True


def carga_columnas(clip):
    f = PRUEBAS / ("cono_cono_%s.csv" % clip)
    if not f.is_file():
        return None
    out = {}
    with f.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["valido"] != "1":
                continue
            out[int(r["cuadro"])] = {
                "peso_max": float(r["peso_max"]) if r["peso_max"] else 0.0,
                "d_manos_cm": float(r["d_manos_cm"]) if r["d_manos_cm"] else float("nan")}
    return out


cands_alto, cands_cero = [], []
for f in sorted(PRUEBAS.glob("cono_cono_*.csv")):
    clip = f.name[len("cono_cono_"):-4]
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        H = int(z["H"])
    if SOLO_1080 and H < 1000:
        continue
    col = carga_columnas(clip)
    if not col:
        continue
    for k, d in col.items():
        """Con `if/elif` explicito y no con una expresion condicional encadenada. La version anterior
        era `(cands_alto if alto else (cands_cero if cero else None) or [])` y **tiraba en silencio
        todos los del estrato CERO**: mientras `cands_cero` estaba vacia, `cands_cero or []` devuelve
        una lista NUEVA vacia y el `append` se perdia. Salieron 367 candidatos del estrato alto y 0
        del cero, o sea que el estrato que sirve para medir FALSOS NEGATIVOS desaparecio sin error."""
        fila = (clip, k, d["peso_max"], d["d_manos_cm"])
        if d["peso_max"] > 0.20:
            cands_alto.append(fila)
        elif d["peso_max"] == 0.0:
            cands_cero.append(fila)

rng = np.random.default_rng(SEMILLA)


def elige(cands, n):
    """Elige n repartidos entre clips, para que no salgan todos del mismo asalto."""
    porc = {}
    for c in cands:
        porc.setdefault(c[0], []).append(c)
    for v in porc.values():
        rng.shuffle(v)
    out, j = [], 0
    while len(out) < n and any(j < len(v) for v in porc.values()):
        for v in porc.values():
            if j < len(v) and len(out) < n:
                out.append(v[j])
        j += 1
    return out


sel = elige(cands_alto, N_POR_ESTRATO) + elige(cands_cero, N_POR_ESTRATO)
orden = list(range(len(sel)))
rng.shuffle(orden)
print("=" * 100)
print("LAMINAS PARA ANOTAR A CIEGAS  ·  %d candidatos con peso_max>0,20 y %d con peso_max=0" % (
    len(cands_alto), len(cands_cero)))
print("  elegidos: %d del estrato ALTO y %d del estrato CERO  ->  %d laminas" % (
    min(N_POR_ESTRATO, len(cands_alto)), min(N_POR_ESTRATO, len(cands_cero)), len(sel)))
print("=" * 100)

clave, hechas = [], 0
caps = {}
for pos, idx in enumerate(orden):
    clip, k, pm, dm = sel[idx]
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, video = int(z["W"]), int(z["H"]), str(z["video"])
    if clip not in caps:
        caps[clip] = cv2.VideoCapture(video)
    cap = caps[clip]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
    okf, im = cap.read()
    if not okf:
        continue
    manos = []
    for lado in (0, 1):
        m, _d = CL.mano_armada(kp, sc, B, W, H, lado)
        if m is None:
            break
        manos.append(kp[k, lado, CL.MANOS[m]["root"]])
    if len(manos) != 2 or not all(np.isfinite(p).all() for p in manos):
        continue
    p0, p1 = manos
    c = 0.5 * (p0 + p1)
    r = max(140.0, 0.9 * float(np.linalg.norm(p0 - p1)))
    x0, x1 = int(max(0, c[0] - r)), int(min(W, c[0] + r))
    y0, y1 = int(max(0, c[1] - r)), int(min(H, c[1] + r))
    if x1 - x0 < 60 or y1 - y0 < 60:
        continue
    """LAS DOS MANOS VAN MARCADAS, y esto NO rompe el ciego. Sin la marca no se sabe **de que pareja**
    se pregunta: en `L03` el recorte salieron cuatro tiradores de DOS pistas distintas y la pregunta
    era indecidible por ambigua, no por falta de resolucion. Marcar las manos dice cual es la
    PREGUNTA; lo que seguiria sesgando seria decir la RESPUESTA, o sea pintar el cono, el eje o
    `peso_max`. Las dos marcas son del MISMO color a proposito: distinguirlas por color insinuaria
    quien ataca y eso tampoco hace falta para contestar si las hojas se tocan."""
    vis = im.copy()
    for p in (p0, p1):
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 9, (0, 230, 255), 2, cv2.LINE_AA)
    sub = vis[y0:y1, x0:x1]
    sub = cv2.resize(sub, (int(sub.shape[1] * ZOOM), int(sub.shape[0] * ZOOM)),
                     interpolation=cv2.INTER_CUBIC)
    ident = "L%02d" % pos
    barra = np.zeros((26, sub.shape[1], 3), np.uint8)
    barra[:] = (14, 14, 20)
    cv2.putText(barra, "%s   se tocan las hojas de las DOS manos marcadas?" % ident, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
    """La barra lleva SOLO el ID. Ni clip, ni cuadro, ni peso_max: son las tres cosas que podrian
    sesgar la anotacion."""
    cv2.imwrite(str(SAL / ("%s.png" % ident)), np.vstack([barra, sub]))
    clave.append({"id": ident, "clip": clip, "cuadro": int(k), "peso_max": pm,
                  "d_manos_cm": dm, "estrato": "alto" if pm > 0.20 else "cero",
                  "recorte": [x0, y0, x1, y1]})
    hechas += 1
for c in caps.values():
    c.release()

(PRUEBAS / "contacto_CLAVE_no_abrir_antes_de_anotar.json").write_text(
    json.dumps({"semilla": SEMILLA, "n_por_estrato": N_POR_ESTRATO, "clave": clave}, indent=1,
               ensure_ascii=False, default=float), encoding="utf-8")
plant = SAL / "anotacion.csv"
if not plant.is_file():
    with plant.open("w", encoding="utf-8") as fh:
        fh.write("id,contacto,nota\n")
        for c in sorted(clave, key=lambda d: d["id"]):
            fh.write("%s,,\n" % c["id"])
    print("  plantilla de anotacion VACIA -> %s" % plant)
else:
    print("  la plantilla ya existe y NO se sobrescribe: %s" % plant)
print("  %d laminas en %s" % (hechas, SAL))
print("  clave (no abrir antes de anotar) -> %s" %
      (PRUEBAS / "contacto_CLAVE_no_abrir_antes_de_anotar.json"))
