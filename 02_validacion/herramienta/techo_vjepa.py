"""¿Supera V-JEPA al acuerdo humano de verdad, o es un artefacto? (`VJEPA2_2026-10-09.md`, comprobacion
pendiente de la version que decide).

B (0,703) y C (0,705) superan el 0,688 de `catalogo.techo_humano()`. Pero ese 0,688 es otra cosa: el
acuerdo ENTRE DOS PERSONAS en 911 clips con dos anotaciones. Lo comparable es el acuerdo del modelo
CON CADA PERSONA, en esos mismos clips con dos o mas anotaciones que caen en el reservado. Un modelo
que aprende el consenso puede coincidir con cada anotador mas de lo que ellos coinciden entre si.
Lo que seria sospechoso es que, en los clips en que los anotadores DISCREPAN, coincidiera con la
etiqueta del catalogo mucho mas que con la otra persona: el modelo estaria aprendiendo algo del
anotador concreto, no de la esgrima.

    python techo_vjepa.py [pred_V0_initiated.npz]
"""
from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parents[2]


def main():
    f = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
        RAIZ / "02_validacion" / "auditoria" / "vjepa_ojos_2026-10-09" / "pred_V0_initiated.npz")
    P = np.load(f, allow_pickle=True)
    clips = list(P["clips"])
    y = P["y"]
    crudo = collections.defaultdict(list)
    for fila, _n in CA.filas_psql(CA.FORM):
        if fila == "__pie__":
            break
        v = fila["initiated"].strip().lower()
        if fila["stats_id"] and v:
            crudo[fila["stats_id"]].append(v)
    idx = [i for i, c in enumerate(clips) if len(crudo.get(c, [])) >= 2]
    print("reservado: %d clips; con >= 2 anotaciones de initiated: %d" % (len(clips), len(idx)))
    if not idx:
        return 0
    clase = {k: j for j, k in enumerate(CA.INICIA)}
    acuerdo = np.array([len(set(crudo[clips[i]])) == 1 for i in idx])
    print("   los anotadores coinciden en %d de %d (%.3f); techo global del catalogo 0,688"
          % (acuerdo.sum(), len(idx), acuerdo.mean()))
    print()
    print("%-6s %-34s %-34s %-34s" % ("", "con la etiqueta del catalogo (y)",
                                     "con cada anotador (media)", "en los que DISCREPAN"))
    for k in ("A", "Ap", "B", "C", "Cp"):
        pr = P["pred_" + k]
        ok_y = np.array([pr[i] == y[i] for i in idx], dtype=float)
        con_cada = np.array([np.mean([pr[i] == clase.get(v, -9) for v in crudo[clips[i]]])
                             for i in idx])
        dis = ~acuerdo
        if dis.any():
            """En los que discrepan: la etiqueta del catalogo es una de las respuestas. Si el modelo
            elige la del catalogo mucho mas que las demas, aprende al anotador."""
            otra = np.array([np.mean([pr[i] == clase.get(v, -9) for v in crudo[clips[i]]
                                      if clase.get(v, -9) != y[i]] or [np.nan])
                             for i in np.array(idx)[dis]])
            txt = "catalogo %.3f, la otra %.3f (n=%d)" % (ok_y[dis].mean(), np.nanmean(otra), dis.sum())
        else:
            txt = "-"
        print("%-6s %-34s %-34s %-34s" % (k.replace("p", "'"), "%.3f (coinciden %.3f, n=%d)" % (
            ok_y.mean(), ok_y[acuerdo].mean(), acuerdo.sum()), "%.3f" % con_cada.mean(), txt))
    print()
    print("Lectura: el acuerdo humano comparable es el de la columna 'coinciden' de los anotadores")
    print("entre si (%.3f en estos clips). Un modelo por encima de eso con CADA anotador no es"
          % acuerdo.mean())
    print("sospechoso por si solo; lo seria una gran diferencia catalogo/otra en los que discrepan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
