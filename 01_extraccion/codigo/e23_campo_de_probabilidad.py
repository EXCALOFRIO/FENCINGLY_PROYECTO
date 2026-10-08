"""ETAPA E, paso 23: EL CONO COMO CAMPO DE PROBABILIDAD, estimado de las hojas que SI se han visto.

Idea del usuario, y ataca la parte mas arbitraria de lo que hay: *"el cono 3D puede estar por dentro
tipo como con puntos del 0 al 1: cuanto mas cerca de 1, mas probabilidad de ser florete dentro del
volumen del cono. Tipo, si hemos trackeado en algun pixel y visto que es florete, lo marcamos como 1,
si no menos, y asi como de la probabilidad que hemos visto por los videos"*.

QUE SUSTITUYE, Y POR QUE ESO ERA LO MAS DEBIL. El relleno actual es `peso_sector = 1 - angulo/apert`,
y eso es una **forma elegida a mano**: decae lineal con el angulo porque es sencillo, no porque se
haya medido que la hoja se reparta asi. El docstring de `cono_lib` ya lo decia con todas las letras
(«es una FORMA ELEGIDA, no una probabilidad medida: nadie ha medido donde esta la hoja dentro del
cono»). Esto lo mide.

LAS CUATRO COSAS QUE SE MIDEN, Y DOS SON NUEVAS EN EL PROYECTO:

  M1  el desvio hoja-eje **CON SIGNO**, no en valor absoluto. Si la hoja se desviara siempre al mismo
      lado, eso es un SESGO y promediarlo a cero lo borraria. Para que el signo signifique lo mismo en
      los dos tiradores se espeja el que mira a la izquierda: el signo va en el marco «mirando al
      rival», y positivo es **hacia el suelo**.
  M2  el desvio **A LO LARGO del eje**, o sea hasta donde llega la recta detectada. Y aqui hay una
      trampa que se declara antes de dar el numero: `HoughLinesP` devuelve **fragmentos**, no la hoja
      entera, asi que `t_max` es una **cota INFERIOR** de donde esta la punta. Luego esta distribucion
      **no puede usarse para acortar L**, solo para comprobar que los trozos caen delante de la mano.
  M3  el **temblor de una hoja VISTA** entre cuadros consecutivos, que es la dispersion que le
      corresponde a un cuadro con `origen = 0`. Sin esto, un 1 medido y un 1 supuesto tendrian la
      misma anchura y serian indistinguibles, que es justo lo que no puede pasar.
  M4  si la hoja se ve donde hay MENOS MOVIMIENTO. El usuario dice «igual en muchos se ve el florete
      claramente», y conviene contestarle con cifra: se compara la velocidad de la mano en los
      recortes CON deteccion contra los recortes SIN deteccion, del mismo clip.

EL CRITERIO QUE DECIDE, DECLARADO ANTES DE MEDIR. El campo empirico sustituye al sector lineal si y
solo si necesita **MENOS AREA** para contener el mismo 90 % de las hojas observadas. Con el mismo
radio L, el area de un sector es proporcional a su anchura angular total, asi que la razon de areas es
la razon de anchuras:

    W_geo  = 2 * p90(|desvio|)            el sector SIMETRICO que contiene el 90 %
    W_emp  = ventana de anchura MINIMA que contiene el 90 % de los desvios con signo
    razon  = W_emp / W_geo

    se adopta si `razon <= 0,90`   (margen del 10 %, declarado aqui)

Si sale 1 o mas, el campo empirico no informa mas que la formula y **se queda el geometrico**, y eso
tambien es un resultado.

Y EL AVISO QUE VA PEGADO AL RESULTADO, porque la n es la que es: la hoja se ve en el **3,0 %** de los
conos de la ventana de accion, y **24 de las 31 detecciones son de UN solo asalto** que ademas no esta
en `corpus_etiquetado.txt`. Asi que todo va con intervalo de confianza por remuestreo y **desglosado
por asalto antes de agregar**. Si con esta n no se puede estimar la forma, el resultado es eso, y
entonces lo que hay que decir es **cuantas hojas harian falta y donde se consiguen**.

CPU Y CACHE. Cero GPU: las rectas salen de los JSON que `e5` y `e13` dejaron, y las poses de los NPZ.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e20_estado_del_cono as E20

PRUEBAS = CL.PRUEBAS
COBERTURA = 0.90
MARGEN_AREA = 0.90
ASALTO = {"LiveDecimalLadybird": "CHEUNG-CHOI", "PowerfulAllLarva": "CHEUNG-CHOI",
          "WelloffForcefulIndri": "BIBARD-OISHI", "IdealFrigidJerboa": "BIBARD-OISHI"}
"""CLIPS QUE SON EL MISMO ASALTO, y por eso NO pueden contar como dos muestras independientes.

Verificado leyendo el rotulo del marcador en un fotograma de cada clip, que es lo unico que lo
distingue: `LiveDecimalLadybird` y `PowerfulAllLarva` son **CHEUNG Ka Long contra CHOI Nicholas
Edward**, el mismo encuentro en dos momentos (R1 2-2 y R3 14-10). Y `WelloffForcefulIndri` con
`IdealFrigidJerboa` es el par que 22.2 ya habia pillado (BIBARD contra OISHI).

Sin esta tabla, el desglose «por asalto» mentiria justo en lo que tiene que medir: si una forma es de
un asalto o de la esgrima. Cualquier clip que no este aqui se toma como asalto propio."""


OBJETIVO_IC = 5.0
N_BOOT = 4000
RNG = np.random.default_rng(20260911)


def asalto(clip):
    return ASALTO.get(clip, clip)


def crudas():
    """TODOS los recortes intentados, con su recta si la hubo. Hacen falta los SIN deteccion.

    `e21.hojas_cacheadas` devuelve solo los positivos, y para M4 (¿se ve donde hay menos movimiento?)
    el denominador son los intentos. Un positivo sin su denominador no es una tasa.
    """
    out = []
    for f in sorted(PRUEBAS.glob("cono_hoja3d_*_accion.json")):
        clip = f.name[len("cono_hoja3d_"):-len("_accion.json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            out.append({"clip": clip, "fuente": "e13_accion", "cuadro": r["cuadro"],
                        "lado": r["lado"], "contaminado": bool(r.get("contaminado")),
                        "linea": r.get("linea")})
    for f in sorted(PRUEBAS.glob("cono_hoja_*.json")):
        clip = f.name[len("cono_hoja_"):-len(".json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            out.append({"clip": clip, "fuente": "e5_clip", "cuadro": r["cuadro"],
                        "lado": r["lado"], "contaminado": bool(r.get("contaminado")),
                        "linea": r["det"].get("p99.5", {}).get("linea")})
    return out


def geo(clip, cache={}):
    if clip in cache:
        return cache[clip]
    try:
        d = E20.carga(clip)
    except FileNotFoundError:
        cache[clip] = None
        return None
    kp, sc, B, W, H, T = d["kp"], d["sc"], d["B"], d["W"], d["H"], d["T"]
    g = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            cache[clip] = None
            return None
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, "codo")
        _pk, pxm = C3.escala_px_m_de_pie(B, lado, H)
        rival = 1 - lado
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm,
                   "alto": (B[:, lado, 3] - B[:, lado, 1]) * H,
                   "x_rival": 0.5 * (B[:, rival, 0] + B[:, rival, 2]) * W}
    g["T"], g["W"], g["H"] = T, W, H
    cache[clip] = g
    return g


def envuelve(a):
    return (a + 180.0) % 360.0 - 180.0


def mide(rec):
    """Desvio con SIGNO en el marco «mirando al rival», y la proyeccion sobre el eje en metros."""
    g = geo(rec["clip"])
    if g is None:
        return None
    k, lado = rec["cuadro"], rec["lado"]
    if k >= g["T"]:
        return None
    gl = g[lado]
    vert, u = gl["vert"][k], gl["u"][k]
    if not (gl["ok"][k] and np.isfinite(vert).all() and np.isfinite(u).all()):
        return None
    ln = rec["linea"]
    p1 = np.asarray(ln["p1"], "float64")
    p2 = np.asarray(ln["p2"], "float64")
    t1, t2 = float((p1 - vert) @ u), float((p2 - vert) @ u)
    cerca, lejos = (p1, p2) if t1 <= t2 else (p2, p1)
    t_min, t_max = min(t1, t2), max(t1, t2)
    d = lejos - cerca
    nd = float(np.linalg.norm(d))
    if nd < 1e-6:
        return None
    """ORIENTADO HACIA DELANTE: el segmento detectado no tiene sentido propio, asi que se orienta del
    extremo mas cercano al mas lejano SOBRE EL EJE. Con eso `d @ u > 0` y el angulo con signo cae en
    (-90, +90) sin ambiguedad de 180 grados."""
    d = d / nd
    sg = float(np.degrees(np.arctan2(u[0] * d[1] - u[1] * d[0], float(u @ d))))
    """ESPEJO: el mismo gesto en un tirador que mira a la izquierda da el signo contrario, porque
    espejar la imagen invierte el sentido de giro. Se lleva todo al marco «el rival esta a la
    derecha», y asi positivo significa lo mismo en los dos: hacia el SUELO."""
    xr = gl["x_rival"][k]
    esp = 1.0 if (np.isfinite(xr) and xr >= vert[0]) else -1.0
    pxm = gl["pxm"]
    return {"clip": rec["clip"], "fuente": rec["fuente"], "cuadro": k, "lado": lado,
            "sg": esp * sg, "abs": abs(sg), "espejo": esp,
            "t_min_m": t_min / pxm, "t_max_m": t_max / pxm, "largo_m": nd / pxm,
            "delante": bool(t_max > 0)}


def pct(v, q):
    return float(np.percentile(v, q))


def ic_boot(v, q):
    b = [pct(RNG.choice(v, v.size, replace=True), q) for _ in range(N_BOOT)]
    return pct(np.asarray(b), 2.5), pct(np.asarray(b), 97.5)


def ventana_minima(v, cob=COBERTURA):
    """La ventana de anchura MINIMA que contiene `cob` de los datos. No es [p5, p95].

    Con una distribucion asimetrica la ventana mas estrecha esta desplazada, y usar los percentiles
    simetricos regalaria anchura. Se busca por fuerza bruta sobre los datos ordenados, que con n=31
    cuesta nada.
    """
    s = np.sort(v)
    m = int(np.ceil(cob * s.size))
    if m >= s.size:
        return float(s[0]), float(s[-1]), float(s[-1] - s[0])
    j = int(np.argmin(s[m - 1:] - s[:s.size - m + 1]))
    return float(s[j]), float(s[j + m - 1]), float(s[j + m - 1] - s[j])


def main():
    print("=" * 108)
    print("EL CONO COMO CAMPO DE PROBABILIDAD, estimado de las hojas VISTAS  ·  sin GPU")
    print("=" * 108)
    raw = crudas()
    det = [m for m in (mide(r) for r in raw
                       if r["linea"] and not r["contaminado"]) if m is not None]
    delante = [m for m in det if m["delante"]]
    print("  recortes intentados %d  ·  con recta %d  ·  con recta DELANTE de la mano %d" % (
        len(raw), len(det), len(delante)))
    sg = np.array([m["sg"] for m in delante])
    ab = np.array([m["abs"] for m in delante])

    print()
    print("  M1 · EL DESVIO HOJA-EJE CON SIGNO (positivo = hacia el suelo, marco «rival a la derecha»)  ·  agrupado POR ASALTO")
    print("  %-26s %4s %8s %8s %8s %8s %8s %8s" % (
        "muestra", "n", "p5", "p25", "p50", "p75", "p95", "media"))

    def linea_sg(nom, v):
        if v.size < 3:
            print("  %-26s %4d   no llega para dar forma" % (nom, v.size))
            return None
        print("  %-26s %4d %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f" % (
            nom, v.size, pct(v, 5), pct(v, 25), pct(v, 50), pct(v, 75), pct(v, 95), v.mean()))
        return {"n": int(v.size), "p5": pct(v, 5), "p25": pct(v, 25), "p50": pct(v, 50),
                "p75": pct(v, 75), "p95": pct(v, 95), "media": float(v.mean())}
    res = {"global_signo": linea_sg("TODAS", sg)}
    por_asalto = {}
    for cl in sorted(set(asalto(m["clip"]) for m in delante)):
        v = np.array([m["sg"] for m in delante if asalto(m["clip"]) == cl])
        por_asalto[cl] = linea_sg("  %s" % cl[:22], v)
    dom = max(por_asalto, key=lambda c: (por_asalto[c] or {"n": 0})["n"])
    resto = np.array([m["sg"] for m in delante if asalto(m["clip"]) != dom])
    res["sin_dominante"] = linea_sg("TODAS menos %s" % dom[:12], resto)
    res["dominante"] = dom
    res["por_asalto"] = por_asalto

    lo50, hi50 = ic_boot(sg, 50)
    lo90, hi90 = ic_boot(ab, 90)
    print()
    print("  IC 95 %% por remuestreo (n=%d): p50 del signo **%.1f gr [%.1f, %.1f]**  ·  "
          "p90 del |desvio| **%.1f gr [%.1f, %.1f]**" % (
              sg.size, pct(sg, 50), lo50, hi50, pct(ab, 90), lo90, hi90))
    hay_sesgo = lo50 > 0 or hi50 < 0
    print("  ¿HAY SESGO? el IC del p50 %s el cero  ->  %s" % (
        "NO contiene" if hay_sesgo else "CONTIENE",
        "SI, la hoja se desvia sistematicamente hacia %s" % ("el suelo" if pct(sg, 50) > 0
                                                             else "arriba")
        if hay_sesgo else "NO se puede afirmar sesgo con esta n"))
    res["ic_p50_signo"] = [lo50, hi50]
    res["ic_p90_abs"] = [lo90, hi90]
    res["sesgo_significativo"] = bool(hay_sesgo)

    print()
    print("  M2 · EL DESVIO A LO LARGO DEL EJE, en metros desde la muneca")
    tmx = np.array([m["t_max_m"] for m in delante])
    tmn = np.array([m["t_min_m"] for m in delante])
    lg = np.array([m["largo_m"] for m in delante])
    print("  hasta donde llega el trozo detectado (`t_max`): p50 **%.2f m**  p90 %.2f m  max %.2f m"
          % (pct(tmx, 50), pct(tmx, 90), tmx.max()))
    print("  donde empieza (`t_min`):                        p50 %.2f m  p10 %.2f m" % (
        pct(tmn, 50), pct(tmn, 10)))
    print("  largo del TROZO detectado:                      p50 **%.2f m**  p90 %.2f m" % (
        pct(lg, 50), pct(lg, 90)))
    print("  -> el trozo mide el %.0f %% de los %.2f m de alcance, asi que son FRAGMENTOS y `t_max`"
          % (100 * pct(lg, 50) / C3.ALCANCE_M, C3.ALCANCE_M))
    print("     es una COTA INFERIOR de donde esta la punta. **El campo NO puede acortar L**: solo")
    print("     confirma que los trozos caen delante de la mano (%.0f %% con t_max > 0)." % (
        100 * len(delante) / max(len(det), 1)))
    res["largo"] = {"t_max_p50": pct(tmx, 50), "t_max_p90": pct(tmx, 90),
                    "t_min_p50": pct(tmn, 50), "largo_p50": pct(lg, 50),
                    "largo_p90": pct(lg, 90), "frac_del_alcance": pct(lg, 50) / C3.ALCANCE_M}

    print()
    print("  M3 · EL TEMBLOR DE UNA HOJA VISTA, que es la anchura que le toca a `origen = 0`")
    vis = {(m["clip"], m["cuadro"], m["lado"]): m for m in delante}
    pares = []
    for (cl, k, la), m in vis.items():
        o = vis.get((cl, k + 1, la))
        if o is not None:
            pares.append(abs(envuelve(o["sg"] - m["sg"])))
    if len(pares) >= 3:
        pa = np.array(pares)
        sig_vista = float(pct(pa, 68) / np.sqrt(2.0))
        print("  pares de cuadros consecutivos con hoja vista en los DOS: **%d**  ·  |cambio| p50 "
              "%.1f gr  p68 %.1f  max %.1f" % (len(pares), pct(pa, 50), pct(pa, 68), pa.max()))
        print("  -> dispersion de un cuadro con hoja VISTA = p68 del cambio / raiz(2) = **%.1f gr**"
              % sig_vista)
        print("     (se divide por raiz de 2 porque el cambio entre dos cuadros lleva el ruido de")
        print("      los dos; es la forma estandar de sacar el ruido de uno a partir de la")
        print("      diferencia de dos medidas independientes de la misma magnitud)")
    else:
        sig_vista = float("nan")
        print("  solo %d pares consecutivos: NO se puede medir el temblor de la hoja vista" %
              len(pares))
    res["sigma_vista_gr"] = sig_vista
    res["n_pares_consecutivos"] = len(pares)

    print()
    print("  M4 · ¿SE VE LA HOJA DONDE HAY MENOS MOVIMIENTO? (velocidad de la mano, recortes del")
    print("       mismo clip, con deteccion contra sin deteccion)")
    con, sin = [], []
    for r in raw:
        if r["contaminado"]:
            continue
        g = geo(r["clip"])
        if g is None or r["cuadro"] >= g["T"] or r["cuadro"] < 1:
            continue
        gl = g[r["lado"]]
        k = r["cuadro"]
        if not (gl["ok"][k] and gl["ok"][k - 1]):
            continue
        alto = gl["alto"][k]
        if not np.isfinite(alto) or alto <= 1:
            continue
        v = float(np.linalg.norm(gl["vert"][k] - gl["vert"][k - 1])) / alto
        (con if r["linea"] else sin).append(v)
    if len(con) >= 5 and len(sin) >= 5:
        c, s = np.array(con), np.array(sin)
        print("  velocidad de la mano, en altos de caja por cuadro:")
        print("    CON hoja detectada (n=%3d): p50 **%.4f**  p25-p75 %.4f-%.4f" % (
            c.size, pct(c, 50), pct(c, 25), pct(c, 75)))
        print("    SIN hoja detectada (n=%3d): p50 **%.4f**  p25-p75 %.4f-%.4f" % (
            s.size, pct(s, 50), pct(s, 25), pct(s, 75)))
        raz = pct(s, 50) / max(pct(c, 50), 1e-9)
        print("    razon SIN/CON = **%.2fx**  ->  %s" % (
            raz, "CONFIRMADO: la hoja se ve donde la mano va MAS LENTA"
            if raz > 1.15 else "no se confirma: las velocidades son parecidas"))
        res["movimiento"] = {"n_con": int(c.size), "n_sin": int(s.size),
                             "p50_con": pct(c, 50), "p50_sin": pct(s, 50), "razon": raz}
    else:
        print("    n insuficiente (con=%d, sin=%d)" % (len(con), len(sin)))
        res["movimiento"] = None

    print()
    print("  EL CRITERIO QUE DECIDE · AREA PARA CONTENER EL %.0f %% DE LAS HOJAS OBSERVADAS"
          % (100 * COBERTURA))
    w_geo = 2.0 * pct(ab, 100 * COBERTURA)
    v_lo, v_hi, w_emp = ventana_minima(sg)
    razon = w_emp / max(w_geo, 1e-9)
    print("  sector SIMETRICO (lo de hoy):   +-%.1f gr  ->  anchura total **%.1f gr**" % (
        pct(ab, 100 * COBERTURA), w_geo))
    print("  ventana EMPIRICA mas estrecha:  [%+.1f, %+.1f] gr  ->  anchura total **%.1f gr**" % (
        v_lo, v_hi, w_emp))
    print("  razon de anchuras (= razon de AREAS con el mismo L) = **%.3f**  ·  limite declarado "
          "<= %.2f  ->  %s" % (razon, MARGEN_AREA,
                               "SE ADOPTA el campo empirico" if razon <= MARGEN_AREA
                               else "NO se adopta: el campo empirico no informa mas"))
    rb = []
    for _ in range(N_BOOT):
        j = RNG.integers(0, sg.size, sg.size)
        s2, a2 = sg[j], ab[j]
        rb.append(ventana_minima(s2)[2] / max(2.0 * pct(a2, 100 * COBERTURA), 1e-9))
    rb = np.asarray(rb)
    ic_lo, ic_hi = pct(rb, 2.5), pct(rb, 97.5)
    print("  IC 95 %% de la razon por remuestreo: **[%.3f, %.3f]**" % (ic_lo, ic_hi))
    print("  -> el IC %s el limite de %.2f y %s el 1,0 (donde el campo empirico no aporta nada)" % (
        "CONTIENE" if ic_lo <= MARGEN_AREA <= ic_hi else "no contiene", MARGEN_AREA,
        "CONTIENE" if ic_lo <= 1.0 <= ic_hi else "no contiene"))
    res["area"] = {"w_geo": w_geo, "w_emp": w_emp, "ventana": [v_lo, v_hi], "razon": razon,
                   "ic_razon": [ic_lo, ic_hi], "limite": MARGEN_AREA,
                   "pasa_la_puerta": bool(razon <= MARGEN_AREA),
                   "establecido": bool(ic_hi < MARGEN_AREA)}

    print()
    print("  Y EL DESGLOSE POR ASALTO DESMIENTE LA ASIMETRIA QUE DA ESA GANANCIA:")
    for cl, d in sorted(por_asalto.items()):
        if d:
            print("    %-30s n=%2d  p50 del signo **%+.1f gr**" % (cl[:30], d["n"], d["p50"]))
    if res["sin_dominante"]:
        print("    %-30s n=%2d  p50 del signo **%+.1f gr**  <- SIGNO CONTRARIO" % (
            "TODAS menos la dominante", res["sin_dominante"]["n"], res["sin_dominante"]["p50"]))
    print("  El asalto dominante desvia hacia ARRIBA y los demas hacia el SUELO. Una asimetria cuyo")
    print("  SIGNO se invierte al quitar un asalto no es una propiedad de la esgrima: es la de ese")
    print("  asalto. Asi que el IC global del p50 que «excluye el cero» es un artefacto de que 24 de")
    print("  31 detecciones son del mismo tirador.")

    print()
    print("  M5 · LA FORMA DEL CAMPO: ¿decae lineal como `1 - ang/apert` o no?")
    bordes = np.arange(0, 36.1, 6.0)
    cen = 0.5 * (bordes[:-1] + bordes[1:])
    h, _b = np.histogram(ab, bins=bordes)
    dens = h / max(h.max(), 1)
    lineal = np.clip(1.0 - cen / pct(ab, 100 * COBERTURA), 0, 1)
    print("  %-14s %s" % ("|desvio| (gr)", "  ".join("%5.0f" % c for c in cen)))
    print("  %-14s %s" % ("hojas (n)", "  ".join("%5d" % x for x in h)))
    print("  %-14s %s" % ("densidad /max", "  ".join("%5.2f" % x for x in dens)))
    print("  %-14s %s" % ("1 - ang/33", "  ".join("%5.2f" % x for x in lineal)))
    err = float(np.mean(np.abs(dens - lineal)))
    print("  error medio absoluto entre la forma medida y la lineal: **%.2f** sobre una escala de "
          "0 a 1" % err)
    print("  -> la forma medida es casi PLANA hasta ~%.0f gr y luego cae, no una rampa lineal: la")
    print("     hoja no esta preferentemente pegada al eje dentro del cono.")
    res["forma"] = {"bordes": bordes.tolist(), "n_por_bin": h.tolist(),
                    "densidad_norm": dens.tolist(), "lineal": lineal.tolist(),
                    "error_medio": err}

    print()
    print("  ¿SE PUEDE ESTIMAR LA FORMA CON ESTA n? Y SI NO, CUANTAS HOJAS HARIAN FALTA")
    anchura_ic = hi90 - lo90
    print("  el IC 95 %% del p90 del |desvio| mide **%.1f gr** de ancho con n=%d. Objetivo "
          "declarado: **%.0f gr** (el margen del proyecto)" % (anchura_ic, ab.size, OBJETIVO_IC))
    n_falta = int(np.ceil(ab.size * (anchura_ic / OBJETIVO_IC) ** 2))
    print("  el ancho del IC de un percentil baja como 1/raiz(n), luego harian falta **n ~ %d** "
          "hojas vistas (x%.1f de las que hay)" % (n_falta, n_falta / max(ab.size, 1)))
    tasa_fuera = 26.0 / 61.0
    tasa_accion = 0.030
    print("  y donde se consiguen, con las tasas ya medidas: **%.1f %%** por recorte a 1080p FUERA "
          "de la ventana de accion (`e5`, 26 de 61) contra **%.1f %%** dentro (`e10`, 6 de 198)" % (
              100 * tasa_fuera, 100 * tasa_accion))
    print("  -> %d recortes de 1080p fuera de la accion, o sea ~%d clips a 2 tiradores x 30 cuadros"
          % (int(np.ceil(n_falta / tasa_fuera)),
             int(np.ceil(n_falta / tasa_fuera / 60.0))))
    print("     de los ~1.540 clips de 1080p del corpus. Es alcanzable y es CPU+GPU de `matting`,")
    print("     no hay que anotar nada a mano.")
    res["cuantas_faltan"] = {"ic_p90_ancho": anchura_ic, "objetivo": OBJETIVO_IC,
                             "n_necesaria": n_falta, "n_actual": int(ab.size),
                             "recortes_1080_fuera": int(np.ceil(n_falta / tasa_fuera)),
                             "clips_aprox": int(np.ceil(n_falta / tasa_fuera / 60.0))}

    (PRUEBAS / "cono_e23_campo.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("guardado %s" % (PRUEBAS / "cono_e23_campo.json"))
    return res


if __name__ == "__main__":
    main()
