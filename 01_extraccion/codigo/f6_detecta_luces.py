"""Corre `luces.detecta` sobre la muestra de un clip por torneo y saca el CSV de luces.

Que hace y que NO hace. Corre el detector con los umbrales DECLARADOS en `luces.py` sobre las 21
sedes, escribe una fila por luz detectada y una tabla resumen por sede. No compara contra nada:
la comparacion contra la verdad de referencia a ojo y contra el detector viejo va en
`f7_generaliza_luces.py`, separada a proposito para que el ajuste y la evaluacion no se toquen en el
mismo fichero.

Columnas del CSV, que son las que pidio el usuario:
    clip, sede, luz (roja/verde/blanca), cuadro_encendido, cuadro_apagado, orden, confianza
y ademas lado, cx, cy, area, salto, base y los tres factores de la confianza por separado, porque
una confianza agregada no dice por que un caso es dudoso.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import luces as LU
import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")


def main() -> int:
    os.makedirs(SALIDA, exist_ok=True)
    refs = F4.carga_referencias()
    filas = []
    for r in refs:
        if not r["ruta"]:
            continue
        t0 = time.time()
        try:
            res = LU.detecta(r["ruta"])
        except Exception as exc:
            print(f"FALLO {r['sede']}: {exc}")
            continue
        agr = LU.agrupa(res["luces"], res["fps"])
        gru = LU.resumen_clip(LU.grupo_del_tocado(agr, res["fps"]))
        claves = {(d["canal"], d["lado"], d["cuadro_encendido"]) for d in gru}
        for d in agr:
            d["del_tocado"] = (d["canal"], d["lado"], d["cuadro_encendido"]) in claves
        seg = time.time() - t0
        print(f"\n### {r['sede']:28s} {res['W']}x{res['H']} n={res['n']} fps={res['fps']:.3f} "
              f"T={res['T']}  {seg:.1f}s  brutas={len(res['luces'])} agrupadas={len(agr)} "
              f"del_tocado={len(gru)}")
        print(f"    viejo: det={r['detector_viejo']} n_luces={r['n_luces_viejo']} "
              f"t_touch={r['t_touch_s']}")
        for d in gru:
            print(f"  TOCADO {d['orden']}  {d['canal']:7s} {d['lado']}  "
                  f"c{d['cuadro_encendido']:4d}-{d['cuadro_apagado']:4d}  {d['duracion_s']:5.2f}s "
                  f"cx={d['cx']:.3f} cy={d['cy']:.3f} area={d['area']:5d} conf={d['confianza']:.3f}"
                  f"{'  DISCORDANTE' if d['discordante'] else ''}")
        for d in agr:
            print(f"    {d['orden']}  {d['canal']:7s} {d['lado']}  c{d['cuadro_encendido']:4d}"
                  f"-{d['cuadro_apagado']:4d}  {d['duracion_s']:5.2f}s  cx={d['cx']:.3f} "
                  f"cy={d['cy']:.3f} area={d['area']:5d} salto={d['salto']:6.1f} "
                  f"base={d['base']:6.1f} conf={d['confianza']:.3f}"
                  f"{'  DISCORDANTE' if d['discordante'] else ''}")
        for d in agr:
            filas.append(dict(clip=r["clip"], sede=r["sede"], luz=d["canal"], **{
                k: v for k, v in d.items() if k != "canal"}))

    destino = os.path.join(SALIDA, "luces_detectadas.csv")
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\nfilas: {len(filas)}   csv: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
