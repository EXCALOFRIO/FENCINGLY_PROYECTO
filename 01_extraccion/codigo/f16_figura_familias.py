"""La figura de cabecera del censo, EN INGLES: 21 torneos, 5 familias de marcador, y las dos cifras.

POR QUE ESTA FIGURA Y POR QUE EN INGLES. El proyecto va a un repositorio publico y el usuario pidio
que el texto de las imagenes este en ingles para que se entienda sin saber espaniol. Y de todo lo
medido hoy, esta es la figura que explica de un golpe por que un detector de luces no generaliza: se
ven las cinco familias de marcador una debajo de otra, con la luz marcada en cada una y su cifra de
cobertura al lado.

LAS DOS CIFRAS QUE VAN DENTRO DE LA IMAGEN, y van dentro a proposito para que no se puedan leer las
demas sin ellas:

    8,11 % de 25.139 clips no tienen NINGUNA luz detectada
    36,78 % en una sola sede (`tokyofoil2022`)

Sin esas dos, cualquiera que vea «una familia cubre el 81 % de los clips» concluiria que el problema
esta resuelto al 81 %, y no lo esta.

TERMINOLOGIA, la que pidio el coordinador y la correcta en esgrima:
    lights, scoring apparatus, red light, green light, **white light (off-target)**, touch,
    valid target, scoreboard / overlay, scoreboard family, counterattack, right of way.
`white light (off-target)` y no `white` a secas porque en florete el blanco significa tocado FUERA
de superficie valida, y esa es la parte que alguien de fuera no puede adivinar.

No mide nada nuevo: dibuja lo ya medido en `f2`, `f3`, `f7` y el censo mirado.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces\figuras")
ANCHO = 1280

FAMILIAS = [
    ("anaheimfoil2018", "A", "FIE classic overlay",
     "16 venues, 20,427 clips (81% of corpus)",
     "light = coloured BAR filling the bottom edge of the name plate",
     "red at x=0.28 W, green at x=0.73 W, both at y=0.91 H  (same in all 16 venues, 2018-2023)"),
    ("kazanfoil2020", "B", "flat blue banner (Kazan / St Petersburg)",
     "2 venues, 1,902 clips (8%)",
     "light = coloured DISC beside the separate clock box",
     "green at x=0.68 W, red at x=0.33 W, y=0.94 H"),
    ("tokyofoil2022", "C", "Tokyo 2022 dark plates",
     "1 venue, 1,433 clips (6%)",
     "light = THIN bar along the bottom edge of the plate",
     "the overlay also has PERMANENT red and green accent lines: the worst confuser of all 21"),
    ("tokyoolympics2020", "D", "OBS Olympic graphics",
     "1 venue, 1,010 clips (4%)",
     "light = two small DOTS flanking the period box",
     "and the scoring fencer's name plate also flashes pink: two regions, one light"),
    ("parisfoil2022", "E", "Paris 2022 compact overlay",
     "1 venue, 367 clips (1%)",
     "light = ONE single square between the two scores, no left/right",
     "side cannot be read from position: the old detector invented a side and reported 2 lights"),
]
PIE = [
    "MEASURED COVERAGE OF THE EXISTING LIGHT DETECTOR, over 25,139 clips:",
    "   8.11% of clips have NO light detected at all  (2,038 clips)",
    "   36.78% in a single venue (tokyofoil2022) -- but measured separately, only 10% of those",
    "   lost clips actually contain a light: the rest are actions that ended with no touch.",
    "Light order is a shortcut, not a solution: 'the fencer who lights SECOND wins' is right 64.96%",
    "of the time on 1,815 two-light clips, and only 21% of clips have two lights at all.",
]


def banda_de(sede: str, luz_esperada: str) -> np.ndarray | None:
    """Banda del marcador con la luz encendida y su caja marcada, para la sede pedida."""
    refs = {r["sede"]: r for r in F4.carga_referencias()}
    r = refs.get(sede)
    if r is None or not r["ruta"]:
        return None
    res = LU.detecta(r["ruta"])
    lz = LU.resumen_clip(LU.grupo_del_tocado(LU.agrupa(res["luces"], res["fps"]), res["fps"]))
    if not lz:
        return None
    luz = max(lz, key=lambda d: d["area"])
    cap = cv2.VideoCapture(r["ruta"])
    k = min(res["n"] - 1, luz["cuadro_encendido"] + 12)
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    H, W = img.shape[:2]
    for d in lz:
        cv2.rectangle(img, (int(d["fx"] * W) - 3, int(d["fy"] * H) - 3),
                      (int((d["fx"] + d["fw"]) * W) + 3, int((d["fy"] + d["fh"]) * H) + 3),
                      (255, 255, 0), max(2, W // 500))
    y0 = int(0.845 * H)
    esc = ANCHO / W
    return cv2.resize(img[y0:H], (ANCHO, int(round((H - y0) * esc))))


def texto(lineas: list[tuple[str, float, tuple]], alto: int) -> np.ndarray:
    lienzo = np.full((alto, ANCHO, 3), 18, np.uint8)
    y = 22
    for txt, esc, col in lineas:
        cv2.putText(lienzo, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, esc, col, 1, cv2.LINE_AA)
        y += int(26 * esc / 0.5)
    return lienzo


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    piezas = [texto([
        ("FENCINGLY - foil scoring lights: 21 tournaments, 5 scoreboard families", 0.75,
         (255, 255, 255)),
        ("In foil the scoring apparatus shows a RED light and a GREEN light for a valid touch "
         "(one per fencer),", 0.5, (200, 200, 200)),
        ("and a WHITE LIGHT (OFF-TARGET) for a touch outside the valid target. On TV they are drawn "
         "inside the", 0.5, (200, 200, 200)),
        ("broadcast scoreboard overlay, and that overlay changes with every tournament and every "
         "broadcaster.", 0.5, (200, 200, 200)),
        ("One family covers 16 of the 21 venues and 81% of the clips: the problem is 5 cases, "
         "not 21.", 0.5, (120, 255, 255)),
    ], 150)]

    for sede, letra, nombre, cobertura, luz, nota in FAMILIAS:
        banda = banda_de(sede, luz)
        cab = texto([
            (f"FAMILY {letra}  -  {nombre}  -  {cobertura}", 0.62, (120, 255, 255)),
            (f"    {luz}", 0.5, (255, 255, 255)),
            (f"    {nota}", 0.45, (170, 170, 170)),
            (f"    example: {sede}   (cyan box = light detected by luces.py)", 0.42, (140, 140, 140)),
        ], 128)
        piezas.append(np.full((4, ANCHO, 3), 90, np.uint8))
        piezas.append(cab)
        if banda is not None:
            piezas.append(banda)
        else:
            print(f"AVISO: sin banda para {sede}")

    piezas.append(np.full((4, ANCHO, 3), 90, np.uint8))
    piezas.append(texto([(t, 0.52 if i == 0 or i == 4 else 0.48,
                          (120, 255, 255) if i == 0 else (225, 225, 225))
                         for i, t in enumerate(PIE)], 190))
    destino = os.path.join(SALIDA, "scoreboard_families_EN.png")
    cv2.imwrite(destino, np.vstack(piezas))
    print(f"escrito {destino}  {np.vstack(piezas).shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
