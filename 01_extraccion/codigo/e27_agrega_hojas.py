"""ETAPA E, paso 27: LA MUESTRA GRANDE DE HOJAS, con exclusiones, estratificada por plano.

`e23` midio el campo con 31 hojas y dijo que no llegaba: IC del p90 de 13,5 grados contra los 5 del
margen, y 24 de las 31 de UN asalto. `e25` recogio hojas en clips nuevos de 1080p. Esto agrega las
dos cosas y **vuelve a pasar el MISMO criterio declarado**, sin moverlo: razon de areas <= 0,90, con
su IC por remuestreo.

Y lleva tres cosas que `e23` no podia tener:

  EXCLUSIONES. `PowerfulAllLarva` sale entero de la muestra, por DOS motivos independientes y los dos
  verificados a ojo: (1) la seccion 18 tiene medido que **su lado 0 es el ARBITRO**, chaqueta oscura
  en primer plano, que paso las tres puertas geometricas y sigue sin arreglarse; un arbitro no lleva
  florete, asi que cualquier «hoja» de ahi es otra cosa. (2) es **el mismo asalto** que
  `LiveDecimalLadybird` (CHEUNG Ka Long contra CHOI Nicholas Edward, R3 14-10 contra R1 2-2), asi que
  ni siquiera aportaria un asalto nuevo. Se dice cuantas detecciones se pierden con ello.

  ESTRATIFICACION POR REGIMEN DE PLANO, y esto es un sesgo de seleccion del peor tipo si no se hace.
  La tuberia **descarta los primeros planos** (seccion 12), asi que una hoja detectada en un cuadro
  de primer plano viene de un regimen sobre el que el cono **no va a correr nunca**. Calibrar con
  ellas mejoraria todas las cifras y empeoraria el producto. Se declara el estrato antes de mirar:

      cuadro que la tuberia ENTREGA   ->  alto de caja <= 0,75 del alto del cuadro
      cuadro de PRIMER PLANO          ->  alto de caja  > 0,75

  Se publica la forma **con y sin** los cuadros de primer plano. Si coinciden, se usan todas y se
  dice. Si no coinciden, **manda la de los planos que la tuberia entrega**, aunque tenga menos n. Y si
  al quitar el primer plano no queda muestra, ese es el resultado: **la hoja solo se ve en los planos
  que descartamos**, que seria una conclusion del proyecto y no un fracaso.

  EL DESGLOSE POR ASALTO usa la tabla `e23.ASALTO`, que agrupa los clips que son el mismo encuentro.
  Sin ella «por asalto» mentiria justo en lo que tiene que medir.

Y UN AVISO DE INDEPENDENCIA QUE NO ES DE ASALTO SINO DE TIRADOR: `GentleBlueIndianspinyloach` es
**KOROBEYNIKOVA contra A. VOLPI**, y `AdorableFittingAsiansmallclawedotter` -- el asalto que dominaba
la muestra vieja -- es **VOLPI contra KONDRICZ**. Son asaltos distintos pero **la misma tiradora esta
en los dos**. Si la forma del desvio fuera una propiedad de la tecnica de cada tirador y no de la
esgrima, esos dos asaltos no serian independientes. Va dicho porque no se puede descartar con la n
que hay.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL
import e23_campo_de_probabilidad as E23

PRUEBAS = CL.PRUEBAS
LIM_PLANO = 0.75
EXCLUIR_CLIP = {"PowerfulAllLarva"}
MOTIVO_EXCL = ("lado 0 es el ARBITRO (seccion 18, verificado a ojo) y el asalto ya esta "
               "representado por LiveDecimalLadybird")


def alto_frac(clip, k, lado, cache={}):
    """Alto de la caja en fraccion del alto del cuadro, que es la variable del filtro de plano."""
    if clip not in cache:
        try:
            import e20_estado_del_cono as E20
            d = E20.carga(clip)
            cache[clip] = (d["B"], d["H"], d["T"])
        except Exception:
            cache[clip] = None
    if cache[clip] is None:
        return float("nan")
    B, H, T = cache[clip]
    if k >= T:
        return float("nan")
    b = B[k, lado]
    return float(b[3] - b[1]) if np.isfinite(b).all() else float("nan")


def analiza(det, nom):
    """M1 y el criterio de areas sobre un subconjunto. Devuelve None si no llega la n."""
    sg = np.array([m["sg"] for m in det])
    ab = np.array([m["abs"] for m in det])
    if sg.size < 8:
        print("  %-34s n=%d: no llega" % (nom, sg.size))
        return None
    w_geo = 2.0 * E23.pct(ab, 90)
    v_lo, v_hi, w_emp = E23.ventana_minima(sg)
    razon = w_emp / max(w_geo, 1e-9)
    rb = []
    for _ in range(E23.N_BOOT):
        j = E23.RNG.integers(0, sg.size, sg.size)
        rb.append(E23.ventana_minima(sg[j])[2] / max(2.0 * E23.pct(ab[j], 90), 1e-9))
    rb = np.asarray(rb)
    ic = (E23.pct(rb, 2.5), E23.pct(rb, 97.5))
    lo90, hi90 = E23.ic_boot(ab, 90)
    lo50, hi50 = E23.ic_boot(sg, 50)
    print("  %-34s n=%3d | p50 signo %+6.1f [%+.1f,%+.1f] | p90 |dev| %5.1f [%.1f,%.1f] | "
          "razon %.3f [%.3f,%.3f] %s" % (
              nom, sg.size, E23.pct(sg, 50), lo50, hi50, E23.pct(ab, 90), lo90, hi90,
              razon, ic[0], ic[1],
              "ADOPTA" if ic[1] < E23.MARGEN_AREA else "no"))
    return {"n": int(sg.size), "p50_signo": E23.pct(sg, 50), "ic_p50": [lo50, hi50],
            "p90_abs": E23.pct(ab, 90), "ic_p90": [lo90, hi90], "ancho_ic_p90": hi90 - lo90,
            "w_geo": w_geo, "w_emp": w_emp, "ventana": [v_lo, v_hi], "razon": razon,
            "ic_razon": list(ic), "pasa_puerta": bool(razon <= E23.MARGEN_AREA),
            "establecido": bool(ic[1] < E23.MARGEN_AREA)}


def temblor(det, nom):
    vis = {(m["clip"], m["cuadro"], m["lado"]): m for m in det}
    pares = [abs(E23.envuelve(vis[(c, k + 1, l)]["sg"] - m["sg"]))
             for (c, k, l), m in vis.items() if (c, k + 1, l) in vis]
    if len(pares) < 5:
        print("  %-34s solo %d pares consecutivos" % (nom, len(pares)))
        return None
    pa = np.array(pares)
    s = float(E23.pct(pa, 68) / np.sqrt(2.0))
    print("  %-34s %4d pares  ·  |cambio| p50 %5.1f  p68 %5.1f  ->  sigma de hoja VISTA "
          "**%.1f gr**" % (nom, len(pares), E23.pct(pa, 50), E23.pct(pa, 68), s))
    return {"n_pares": len(pares), "p50": E23.pct(pa, 50), "p68": E23.pct(pa, 68), "sigma": s}


def main():
    raw = E23.crudas()
    todo = [m for m in (E23.mide(r) for r in raw
                        if r["linea"] and not r["contaminado"]) if m is not None]
    for m in todo:
        m["asalto"] = E23.asalto(m["clip"])
        m["alto_frac"] = alto_frac(m["clip"], m["cuadro"], m["lado"])
        m["primer_plano"] = bool(np.isfinite(m["alto_frac"]) and m["alto_frac"] > LIM_PLANO)
    print("=" * 116)
    print("MUESTRA AGREGADA DE HOJAS  ·  exclusiones, desglose por asalto y estratificada por plano")
    print("=" * 116)
    fuera = [m for m in todo if m["clip"] in EXCLUIR_CLIP]
    todo = [m for m in todo if m["clip"] not in EXCLUIR_CLIP]
    print("  EXCLUIDO %s: **%d detecciones fuera**  ·  %s" % (
        ", ".join(sorted(EXCLUIR_CLIP)), len(fuera), MOTIVO_EXCL))
    delante = [m for m in todo if m["delante"]]
    print("  con recta %d  ·  delante de la mano **%d**  ·  asaltos distintos **%d**" % (
        len(todo), len(delante), len(set(m["asalto"] for m in delante))))

    print()
    print("  POR ASALTO, antes de agregar nada")
    print("  %-34s %5s %7s %9s %9s" % ("asalto", "n", "p50 sg", "p90 |dev|", "primer plano"))
    por = {}
    for a in sorted(set(m["asalto"] for m in delante)):
        s = [m for m in delante if m["asalto"] == a]
        sg = np.array([m["sg"] for m in s])
        ab = np.array([m["abs"] for m in s])
        pp = float(np.mean([m["primer_plano"] for m in s]))
        print("  %-34s %5d %+7.1f %9.1f %8.0f %%" % (
            a[:34], len(s), np.median(sg), E23.pct(ab, 90) if sg.size >= 3 else float("nan"),
            100 * pp))
        por[a] = {"n": len(s), "p50_signo": float(np.median(sg)),
                  "p90_abs": E23.pct(ab, 90) if sg.size >= 3 else None, "frac_primer_plano": pp}

    print()
    print("  EL CRITERIO DECLARADO, sin moverlo: razon de areas <= %.2f, y se ADOPTA solo si el IC"
          % E23.MARGEN_AREA)
    print("  entero queda por debajo del limite")
    res = {"excluidas": len(fuera), "motivo_exclusion": MOTIVO_EXCL, "por_asalto": por,
           "lim_plano": LIM_PLANO}
    ent = [m for m in delante if not m["primer_plano"]]
    pri = [m for m in delante if m["primer_plano"]]
    res["todas"] = analiza(delante, "TODAS")
    res["entrega_tuberia"] = analiza(ent, "solo plano que la tuberia ENTREGA")
    res["primer_plano"] = analiza(pri, "solo PRIMER PLANO")
    dom = max(por, key=lambda a: por[a]["n"]) if por else None
    if dom:
        res["sin_dominante"] = analiza([m for m in delante if m["asalto"] != dom],
                                       "TODAS menos %s" % dom[:18])

    print()
    print("  ¿COINCIDEN LOS DOS REGIMENES? Es lo que decide si se pueden usar las hojas de primer")
    print("  plano o hay que calibrar solo con las que la tuberia entrega.")
    a, b = res["entrega_tuberia"], res["primer_plano"]
    if a and b:
        dp = abs(a["p90_abs"] - b["p90_abs"])
        solapan = not (a["ic_p90"][1] < b["ic_p90"][0] or b["ic_p90"][1] < a["ic_p90"][0])
        print("  p90 |dev|: entrega %.1f [%.1f,%.1f]  ·  primer plano %.1f [%.1f,%.1f]  ·  "
              "difieren %.1f gr" % (a["p90_abs"], a["ic_p90"][0], a["ic_p90"][1],
                                    b["p90_abs"], b["ic_p90"][0], b["ic_p90"][1], dp))
        print("  los IC %s  ->  %s" % (
            "SE SOLAPAN" if solapan else "NO se solapan",
            "no hay prueba de que los dos regimenes difieran: se pueden agregar, y va dicho"
            if solapan else
            "LOS REGIMENES DIFIEREN: manda la calibracion del plano que la tuberia entrega"))
        res["regimenes_solapan"] = bool(solapan)
    elif a is None and b is not None:
        print("  **SIN MUESTRA en el plano que la tuberia entrega.** Entonces la conclusion es que")
        print("  la hoja SOLO se ve en los planos que descartamos, y eso es un resultado del")
        print("  proyecto: el cono con direccion medida no es construible donde el cono corre.")
        res["regimenes_solapan"] = None

    print()
    print("  EL TEMBLOR DE LA HOJA VISTA, con mas pares que los 8 de `e23`")
    res["temblor_todas"] = temblor(delante, "TODAS")
    res["temblor_entrega"] = temblor(ent, "solo plano que la tuberia ENTREGA")

    print()
    manda = res["entrega_tuberia"] or res["todas"]
    if manda:
        anc = manda["ancho_ic_p90"]
        n_f = int(np.ceil(manda["n"] * (anc / E23.OBJETIVO_IC) ** 2))
        print("  ¿LLEGA LA n? El IC del p90 mide **%.1f gr** con n=%d (era 13,5 con n=31). Objetivo"
              % (anc, manda["n"]))
        print("  declarado %.0f gr  ->  harian falta **n ~ %d** (x%.1f de lo que hay ahora)" % (
            E23.OBJETIVO_IC, n_f, n_f / max(manda["n"], 1)))
        res["cuantas_faltan"] = {"ic_ancho": anc, "n_actual": manda["n"], "n_necesaria": n_f}

    (PRUEBAS / "cono_e27_agregado.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("guardado %s" % (PRUEBAS / "cono_e27_agregado.json"))


if __name__ == "__main__":
    main()
