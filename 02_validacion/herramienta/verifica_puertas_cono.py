"""Las tres puertas de atribucion del cono: cuanto filtran y que se pierde por el camino.

## POR QUE EXISTE ESTE FICHERO

`CHECKS.md` no tiene **ni una fila** sobre la hoja ni sobre el cono: toda esa parte del proyecto
vivia fuera de la tabla de controles reproducibles. Y los tres defectos de atribucion estaban
medidos en `e17_defectos_del_cono.py` desde hacia tiempo y **no se aplicaban en ningun sitio**.

## LOS TRES DEFECTOS

    D1  el vertice esta pegado al tronco          no puede ser una mano armada extendida
    D2  la mano cae DENTRO de la caja del rival   los dos conos salen del mismo cuerpo
    D3  la otra mano esta mas cerca del rival     el criterio (a) de `mano_armada` se contradice

No son teoricos. La anotacion ciega de `02_validacion/muestras/contacto_hojas/anotacion.csv` los
recoge a mano: *"DEFECTO las dos marcas caen en el MISMO tirador y la de abajo esta en la cadera"*
(L01), *"...una en la cadera o espalda"* (L06), *"DEFECTO la mano armada es el brazo TRASERO del que
se lanza a fondo"* (L10). Son D2, D2 y D3.

## LAS DOS CIFRAS QUE HAY QUE DAR JUNTAS

Una puerta se juzga por lo que quita **y** por lo que cuesta. Aqui:

    cuanto filtra     que fraccion de los conos publicados cae
    que cuesta        cuantos de los conos con HOJA VISTA se pierden

El segundo es el que importa, porque la hoja vista es el unico dato real que hay: sobre los tres
clips completos son **6 de 1.700 cuadros-lado (0,35 %)**, y los otros 192 conos son el antebrazo
extrapolado con una envolvente de +-33 grados. Una puerta que quitara conos de hoja vista estaria
tirando justo lo unico medido.

    python verifica_puertas_cono.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import cono3d_lib as C3  # noqa: E402

PRUEBAS = pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "pruebas"


def _lect(kp, k, lado):
    return {i: {"x": float(kp[k, lado, i, 0]), "y": float(kp[k, lado, i, 1])}
            for i in range(kp.shape[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    print("LAS TRES PUERTAS DE ATRIBUCION DEL CONO")
    print("=" * 92)
    print("%-34s %7s %7s %9s  %s" % ("clip", "conos", "hoja", "filtrados", "desglose"))
    print("-" * 92)
    TC = TH = TD = TDH = 0
    TL = 0
    glob_tot = collections.Counter()
    for p in sorted(glob.glob(str(PRUEBAS / "cono3d_completo_*.npz"))):
        d = np.load(p, allow_pickle=True)
        kp, B, ap_, org = d["kp"], d["B"], d["apert"], d["origen"]
        W, H = int(d["W"]), int(d["H"])
        man = [str(v) for v in np.asarray(d["mano"]).ravel()[:2]]
        nc = nh = nd = ndh = 0
        c = collections.Counter()
        for k in range(kp.shape[0]):
            for lado in (0, 1):
                if not np.isfinite(ap_[k, lado]):
                    continue
                nc += 1
                hoja = int(org[k, lado]) == 0
                nh += 1 if hoja else 0
                f = C3.defectos_de_atribucion(_lect(kp, k, lado), _lect(kp, k, 1 - lado),
                                              man[lado], B[k, 1 - lado], W, H)
                if f:
                    nd += 1
                    ndh += 1 if hoja else 0
                    for x in f:
                        c[x] += 1
                        glob_tot[x] += 1
        TC += nc
        TH += nh
        TD += nd
        TDH += ndh
        TL += int(np.isfinite(ap_).size)
        print("%-34s %7d %7d %9d  %s"
              % (os.path.basename(p)[16:50][:34], nc, nh, nd, dict(c)))
    print("-" * 92)
    if TC == 0:
        print("no hay npz de cono en %s. Sin cifras." % PRUEBAS)
        return 0
    print("%-34s %7d %7d %9d  %s" % ("TOTAL", TC, TH, TD, dict(glob_tot)))
    print()
    print("conos publicados hoy             %d de %d cuadros-lado  (%.1f %%)"
          % (TC, TL, 100.0 * TC / TL))
    print("   de ellos con HOJA VISTA       %d  (%.2f %% de los cuadros-lado)"
          % (TH, 100.0 * TH / TL))
    print("   el resto es ANTEBRAZO extrapolado con envolvente de +-33 grados")
    print()
    print("filtrados por las puertas        %d de %d  = %.1f %%" % (TD, TC, 100.0 * TD / TC))
    print("   de los de hoja vista se pierden  %d de %d" % (TDH, TH))
    print("sobreviven                       %d de %d" % (TC - TD, TC))
    print()
    if TDH == 0:
        print("LAS PUERTAS NO CUESTAN SENAL: quitan solo conos de antebrazo mal atribuido y no")
        print("tocan ninguno de los que tienen hoja vista, que son el unico dato medido.")
    else:
        print("ATENCION: las puertas se llevan %d conos de HOJA VISTA. Eso es caro y hay que" % TDH)
        print("mirarlos uno a uno antes de adoptarlas.")
    print()
    print("AVISO SOBRE ESTOS NPZ: se generaron con `L = HOJA_M = 0,90 m` y sin puertas. Desde que")
    print("`ALCANCE_M = 1,10` esta cableado, `am2` y `am3` -- el veredicto de amenaza -- cambian,")
    print("asi que hay que regenerarlos antes de citar ninguna cifra de amenaza.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
