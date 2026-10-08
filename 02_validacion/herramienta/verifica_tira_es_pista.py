"""EL CONTROL QUE FALTABA: comprobar que la tira rectificada es LA PISTA y no el suelo de al lado.

## DE DONDE SALE ESTE CONTROL

Mirando los tableros de anotacion de los tres primeros clips que llegan a tener homografia. En los
tres, la "tira rectificada" **no era la pista**: era el suelo verde con sus costuras, la mesa del
marcador con su rotulo, y las vallas de publicidad del fondo. Las marcas que el emparejamiento
recibia eran costuras de moqueta y bordes de cartel.

Eso reordena el problema entero. Se llevaba la sesion midiendo y arreglando `identifica_marcas` --
el 27 % de patrones imposibles, la completitud, el residuo contaminado -- y todo eso es cierto pero
**irrelevante mientras la entrada no sea la pista**. Es el mismo error que el proyecto ya tiene
documentado dos veces: arreglar el calculo cuando el fallo estaba en la ruta de lectura.

## EL CONTROL, Y POR QUE ES FISICO Y NO OPINABLE

Los dos tiradores **estan sobre la pista**. No es una heuristica ni un supuesto estadistico: es la
definicion del deporte, y sus pies no pueden estar en otro sitio mientras se tira. Asi que:

    si la tira es la pista   ->   los apoyos caen entre los dos bordes,  u en [0, 1]
    si la tira no lo es      ->   los apoyos caen fuera, y se ve cuanto

`u` es la fraccion del ancho entre el borde A (u = 0) y el borde B (u = 1). Se admite un margen: la
caja del tirador da el apoyo con sesgo, y en fondo el pie puede pisar el borde. Se declara **antes
de medir**:

    APOYO DENTRO     -0,25 <= u <= 1,25
    TIRA ES PISTA    al menos el 70 % de los apoyos dentro

El 0,70 y no el 1,00 porque hay cuadros con el tirador saltando, con la caja mal, o fuera de la
pista entre acciones. Y el margen de 0,25 porque un apoyo justo en el borde no es un fallo de la
tira.

## LO QUE ESTE NUMERO ACOTA

Si la tira solo es la pista en una fraccion `p` de los clips, entonces **ninguna mejora del
emparejamiento puede subir la posicion por encima de `p`**, porque en el resto no hay nada que
emparejar. Es un techo, igual que el 0,619 del acuerdo humano lo es por el otro lado.

    python verifica_tira_es_pista.py --n 40
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import pista as PI  # noqa: E402
import rutas as RU  # noqa: E402

MARGEN_U = 0.25
FRAC_MIN = 0.70


def apoyos_en_tira(res, t):
    """QUE DEVUELVE: el vector `u` de todos los apoyos de los dos tiradores en el sistema de la
    tira, o None. `u = 0` es el borde A y `u = 1` el borde B.

    Se usa el apoyo de CAJA y no la pose porque es el que existe siempre; la pose es mas fina pero
    tiene huecos, y aqui lo que se mide es de que lado de la pista cae, que no necesita finura.
    """
    r = res.get("_r")
    Mk = res.get("_Mk")
    if r is None:
        return None
    P = PI._pies_caja(r)
    pts = []
    for k in np.nonzero(r["mask"])[0]:
        for lado in (0, 1):
            p = P[k, lado]
            if not np.isfinite(p).all():
                continue
            if Mk is not None and k < len(Mk):
                v = Mk[k] @ np.array([p[0], p[1], 1.0])
                p = v[:2] / v[2]
            pts.append(p)
    if len(pts) < 8:
        return None
    ct = PI.a_tira(t, np.asarray(pts, "float64"))
    u = PI.u_de_fila(t, ct[:, 1])
    return u[np.isfinite(u)]


def una(clip):
    r = PI.analiza(clip)
    if not r or not r.get("ok"):
        return {"clip": clip, "ok": False, "motivo": (r or {}).get("motivo", "sin homografia")}
    t = PI.tira_rectificada(r)
    if t is None or "tira" not in t:
        return {"clip": clip, "ok": False, "motivo": "sin tira"}
    u = apoyos_en_tira(r, t)
    if u is None or len(u) < 8:
        return {"clip": clip, "ok": False, "motivo": "menos de 8 apoyos"}
    dentro = float(((u >= -MARGEN_U) & (u <= 1.0 + MARGEN_U)).mean())
    ch = r.get("_hg", {}).get("ctrl_h") or {}
    if not ch:
        try:
            ch = PI.control_estatura(r["_hg"], r["_r"], r.get("_Mk"))
        except Exception:
            ch = {}
    h = ch.get("h_p50")
    """LA ESTATURA REPROYECTADA ES LA MEDIDA INDEPENDIENTE DEL ANCHO, y `mide_pista` ya lo dice con
    todas las letras: como todas las distancias escalan con `ancho_m`, si la estatura sale `h`
    cuando deberia salir 1,75 m, el ancho verdadero de la banda es `ancho_m * 1,75 / h`. O sea que
    con `h` se sabe si los dos bordes detectados son los de la pista o abarcan medio pabellon, y no
    hace falta mirar ninguna imagen para saberlo."""
    anch = (PI.ANCHO_NOMINAL * 1.75 / h) if (h and np.isfinite(h) and h > 0.05) else float("nan")
    return {"clip": clip, "ok": True, "frac_dentro": dentro,
            "u_p10": float(np.percentile(u, 10)), "u_p50": float(np.median(u)),
            "u_p90": float(np.percentile(u, 90)), "n": len(u),
            "h_p50": float(h) if h else float("nan"), "ancho_implicito_m": float(anch),
            "es_pista": bool(dentro >= FRAC_MIN and np.isfinite(anch) and 1.2 <= anch <= 2.6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    clips = a.clips or [os.path.basename(p)[6:-4] for p in
                        sorted(glob.glob(str(RU.TRABAJO / "poses_*.npz")))][:a.n]
    print("LA TIRA RECTIFICADA, ES LA PISTA?")
    print("=" * 88)
    print("criterio declarado: apoyo dentro si -%.2f <= u <= %.2f ; tira es pista si >= %.0f %%"
          % (MARGEN_U, 1 + MARGEN_U, FRAC_MIN * 100))
    print()
    print("%-34s %7s %7s %7s %8s %8s  %s"
          % ("clip", "dentro", "u_p10", "u_p90", "estatura", "anchoimp", ""))
    print("-" * 88)
    fil, sin = [], 0
    for c in clips:
        try:
            v = una(c)
        except Exception as ex:
            print("%-34s EXCEPCION %s" % (c[:34], type(ex).__name__))
            continue
        if not v.get("ok"):
            sin += 1
            print("%-34s %s" % (c[:34], str(v.get("motivo"))[:46]))
            continue
        fil.append(v)
        print("%-34s %7.2f %7.2f %7.2f %8.2f %8.2f  %s"
              % (c[:34], v["frac_dentro"], v["u_p10"], v["u_p90"],
                 v["h_p50"], v["ancho_implicito_m"],
                 "PISTA" if v["es_pista"] else "NO"))
    print()
    if not fil:
        print("ningun clip con tira. No se da ninguna cifra.")
        return 0
    ok = sum(1 for v in fil if v["es_pista"])
    print("clips con homografia:            %d de %d  (%d sin)" % (len(fil), len(clips), sin))
    print("de esos, la tira ES la pista:    %d de %d  = %.1f %%"
          % (ok, len(fil), 100.0 * ok / len(fil)))
    print("sobre el total de clips:         %d de %d  = %.1f %%"
          % (ok, len(clips), 100.0 * ok / len(clips)))
    hh = np.array([v["h_p50"] for v in fil], "float64")
    aa = np.array([v["ancho_implicito_m"] for v in fil], "float64")
    hh, aa = hh[np.isfinite(hh)], aa[np.isfinite(aa)]
    if len(hh):
        print("estatura reproyectada (m):       p10 %.2f  p50 %.2f  p90 %.2f   (deberia ser 1,75)"
              % (np.percentile(hh, 10), np.median(hh), np.percentile(hh, 90)))
        print("ancho implicito de la banda (m): p10 %.2f  p50 %.2f  p90 %.2f   (pista real 1,5-2,0)"
              % (np.percentile(aa, 10), np.median(aa), np.percentile(aa, 90)))
    d = np.array([v["frac_dentro"] for v in fil])
    print("fraccion de apoyos dentro:       p10 %.2f  p50 %.2f  p90 %.2f"
          % (np.percentile(d, 10), np.median(d), np.percentile(d, 90)))
    print()
    print("ESTE NUMERO ES UN TECHO. Si la tira solo es la pista en el %.0f %% de los clips con"
          % (100.0 * ok / len(clips)))
    print("homografia, ninguna mejora del emparejamiento puede pasar de ahi: en el resto no hay")
    print("marcas de pista que emparejar, hay costuras de moqueta y bordes de cartel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
