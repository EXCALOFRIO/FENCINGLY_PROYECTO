"""Comprueba que un interprete NUEVO saca lo MISMO que el que produjo lo que ya hay en disco.

## POR QUE ESTO NO ES OPCIONAL

El 2026-09-23 hubo que reconstruir el entorno de Python desde cero, porque los venv vivian en un
directorio temporal y un borrado se los llevo. El entorno nuevo **no es identico**: `pandas` paso
de 3.0.5 a 3.0.6, `ultralytics` de 8.4.142 a 8.4.160, `sklearn` de 1.9.0 a 1.9.1. `numpy` y
`torch` se clavaron a la version de antes (2.5.2 y 2.14.0+cu126) y `transformers` a 5.16.1.

Si el entorno nuevo sacara poses distintas, el corpus quedaria **mezclado**: unos clips extraidos
con un entorno y otros con otro, sin nada que lo diga. Eso no se arregla luego. Asi que antes de
seguir extrayendo hay que medir la diferencia, y si la hay, decirla.

## EL CRITERIO, DECLARADO ANTES DE MEDIR

    identico          maximo |delta| = 0 en KP, SC, B y pres  -> se sigue sin mas
    dentro de fp16    p50 < 0,02 px y maximo < 0,5 px         -> se sigue y se DEJA ESCRITO
    distinto          cualquier cosa por encima                -> se para y se investiga

El limite de 0,5 px es 1/4 del error de pose publicado (2,26 px), el mismo que se uso para juzgar
el lote del modelo. `pres` y `mask` son booleanos: **cualquier diferencia ahi es "distinto"**, sin
tolerancia, porque cambia que cuadros se entregan y no solo su valor.

    python verifica_reproduce.py [--n 5]
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402

P50_MAX = 0.02
MAX_MAX = 0.5
CONF = 0.3
"""El mismo umbral de confianza con el que el resto del proyecto decide si un keypoint se usa."""


def reextrae(clips, destino, py=None):
    """QUE HACE: corre las etapas 1 y 4 sobre `clips` escribiendo en `destino`, con el interprete
    que toque. `FENCINGLY_TRABAJO` apunta al destino, asi que **no se toca nada de lo ya hecho**."""
    py = py or RU.interprete()
    env = dict(os.environ)
    env["FENCINGLY_TRABAJO"] = str(destino)
    env["PYTHONIOENCODING"] = "utf-8"
    codigo = (
        "import sys\n"
        "sys.path.insert(0, %r)\nsys.path.insert(0, %r)\n"
        "import video_revision as VR\n"
        "cl = %r\n"
        "VR.fase1(cl)\n"
        "VR.fase2(cl, video=False)\n" % (str(RU.CODIGO), str(RU.HERRAMIENTA), list(clips)))
    r = subprocess.run([py, "-c", codigo], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")[-1500:]


def compara(a, b):
    """QUE DEVUELVE: lista de (campo, veredicto, detalle) comparando dos `poses_*.npz`."""
    out = []
    with np.load(a, allow_pickle=True) as za, np.load(b, allow_pickle=True) as zb:
        for k in ("pres", "mask", "origen", "T", "W", "H"):
            if k not in za.files or k not in zb.files:
                out.append((k, "FALTA", "no esta en uno de los dos"))
                continue
            x, y = za[k], zb[k]
            if x.shape != y.shape:
                out.append((k, "DISTINTO", "forma %s contra %s" % (x.shape, y.shape)))
            elif not np.array_equal(x, y):
                n = int((x != y).sum())
                out.append((k, "DISTINTO", "%d valores distintos de %d" % (n, x.size)))
            else:
                out.append((k, "identico", ""))
        for k in ("KP", "SC", "B"):
            if k not in za.files or k not in zb.files:
                continue
            x, y = za[k], zb[k]
            if x.shape != y.shape:
                out.append((k, "DISTINTO", "forma %s contra %s" % (x.shape, y.shape)))
                continue
            m = np.isfinite(x) & np.isfinite(y)
            if k == "KP":
                """SOLO LOS KEYPOINTS QUE ALGUIEN USA, y no es una manga ancha: es que comparar
                los demas mide basura. El modelo de pose devuelve coordenadas FINITAS y absurdas
                para puntos de confianza casi nula -- medido aparte, hasta 8.553 px fuera de un
                cuadro de 1.920 de ancho -- y una diferencia de un bit en fp16 los manda a OTRO
                sitio absurdo. Sin este filtro el maximo salia **724 px** con la mediana en cero,
                que es exactamente esa firma. Todo lo que consume el NPZ filtra por `CONF`, asi
                que lo que hay que reproducir es lo que pasa ese filtro."""
                sa, sb = za["SC"], zb["SC"]
                usable = (sa >= CONF) & (sb >= CONF)
                m = m & usable[..., None]
            if not m.any():
                out.append((k, "identico", "todo NaN o nada usable en los dos"))
                continue
            d = np.abs(x[m] - y[m])
            p50, mx = float(np.median(d)), float(d.max())
            nan_a, nan_b = int((~np.isfinite(x)).sum()), int((~np.isfinite(y)).sum())
            det = "p50 %.6f  max %.6f  (n=%d%s)" % (
                p50, mx, int(m.sum()),
                ", usables>=%.1f" % CONF if k == "KP" else "")
            if mx == 0 and nan_a == nan_b:
                out.append((k, "identico", det))
            elif p50 < P50_MAX and mx < MAX_MAX and nan_a == nan_b:
                out.append((k, "dentro de fp16", det))
            else:
                out.append((k, "DISTINTO", det))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    py = RU.interprete()
    print("REPRODUCIBILIDAD DEL ENTORNO")
    print("=" * 84)
    print("interprete nuevo : %s" % py)
    print("lo ya extraido   : %s" % RU.TRABAJO)
    print()

    clips = a.clips or sorted(
        os.path.basename(p)[6:-4]
        for p in glob.glob(str(RU.TRABAJO / "poses_*.npz")))[:a.n]
    clips = [c for c in clips if RU.video(c)][:a.n]
    if not clips:
        print("no hay clips con poses Y video para comparar.")
        return 2
    print("se reextraen %d clips en un directorio aparte: %s" % (len(clips), ", ".join(clips)))
    print()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fz_repro_"))
    try:
        rc, salida = reextrae(clips, tmp, py)
        if rc != 0:
            print("la reextraccion FALLO (codigo %d):" % rc)
            print(salida[-1200:])
            return 2
        mal = fp16 = ident = 0
        for c in clips:
            nuevo = tmp / ("poses_%s.npz" % c)
            viejo = RU.TRABAJO / ("poses_%s.npz" % c)
            if not nuevo.is_file():
                print("%-38s NO SE ESCRIBIO" % c[:38])
                mal += 1
                continue
            print("%s" % c)
            for campo, ver, det in compara(viejo, nuevo):
                print("   %-8s %-16s %s" % (campo, ver, det))
                if ver == "DISTINTO" or ver == "FALTA":
                    mal += 1
                elif ver == "dentro de fp16":
                    fp16 += 1
                else:
                    ident += 1
            print()
        print("=" * 84)
        print("campos identicos %d, dentro de fp16 %d, DISTINTOS %d" % (ident, fp16, mal))
        if mal:
            print()
            print("EL ENTORNO NUEVO NO REPRODUCE. No se sigue extrayendo con el hasta saber por")
            print("que: mezclar dos entornos en un mismo corpus no se arregla despues.")
            return 1
        if fp16:
            print()
            print("Reproduce dentro del ruido de fp16. Queda ESCRITO que el corpus tiene clips de")
            print("dos entornos y cuanto se mueven entre si.")
        else:
            print()
            print("Reproduce EXACTAMENTE. El corpus es homogeneo.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
