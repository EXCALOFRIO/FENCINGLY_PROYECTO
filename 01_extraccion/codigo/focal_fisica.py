"""La focal elegida por FISICA y no por el angulo de las marcas, y el ancho de pista DEDUCIDO.

## LA MEDIDA QUE ORDENA ESTE MODULO

Sobre los 78 clips con diagnostico completo (`02_validacion/pista/homografia.csv`):

    llegan a tener homografia                                    43 de 78   (55 %)
    ... Y ADEMAS con estatura reproyectada en 1,60-2,00 m        10 de 41   (24 %)
    -> homografias metricamente sanas                            10 de 78   (13 %)

    estatura reproyectada:   p10 1,07   p50 1,88   p90 4,16 m
    f_rel (focal / ancho):   min 0,30   p50 0,85   max 6,00

**El 13 % es el techo real de la posicion**, y no tiene nada que ver con el emparejamiento de
marcas. Se puede arreglar `identifica_marcas` hasta la perfeccion y la posicion no subira de ahi.

Y el discriminador entre los 10 buenos y los 33 malos es uno solo. No es el numero de marcas (4,5
contra 4,0), ni las identificadas (4,0 contra 4,0), ni el angulo de fuga vertical (0,24 contra
0,17), ni el ancho en pixeles (132 contra 131). Es **la focal**: 0,91 contra 0,71, y pegada a los
dos topes del rango -- `f_rel = 0,30` es un campo horizontal de **118 grados**, o sea un ojo de pez,
y `f_rel = 6,00` son **9,5 grados**, un teleobjetivo con el que no caben dos tiradores separados
4 m. Ninguna de las dos existe en una retransmision.

## POR QUE LA FOCAL SALE MAL HOY

`_f_y_v2` barre la focal y se queda con la que hace que las marcas transversales apunten mejor a
V2. Es correcto pero **debil**: las marcas son cortas, casi paralelas entre si y ruidosas. El propio
modulo lo sabe y por eso guarda `curva_f`, *"que es la medida de lo bien determinada que esta: si el
residuo no sube, la focal no esta medida, esta elegida"*.

Y mientras tanto hay una medida **fuerte** en el mismo clip que no se usa para decidir nada: la
ESTATURA. Dos tiradores por unos 50 cuadros son del orden de **100 observaciones de una cantidad
conocida** (1,75 m). Hoy eso solo se mira al final, como aprobado/suspenso, y suspende en 31 de 41.

## LO QUE HACE ESTE MODULO

Se recorre la misma familia de un parametro, pero el ancho de pista **deja de imponerse**:

    para cada f:   V2(f) = (w V1) x (w Vz)              igual que hoy
                   H con ancho unitario                  todo queda en unidades de ancho de pista
                   s(f) = estatura reproyectada          en anchos de pista
                   ancho(f) = 1,75 / s(f)                el ancho que DEDUCE esa focal
                   hc(f) = hc_unitaria * ancho(f)        la altura de camara que implica

    se descarta f si   el ancho deducido se sale de 1,40-2,10 m   (el reglamento dice 1,50-2,00)
                       o la altura de camara se sale de 1,2-15 m
    entre las que quedan, gana la del criterio angular de siempre.

O sea: la estatura fija la escala y el **ancho reglamentario de la pista pasa a ser una restriccion
de verdad** en vez de un numero impuesto. Eso convierte una focal mal determinada por un residuo
plano en una focal acotada por dos hechos fisicos.

## EL PRECIO, Y HAY QUE DECIRLO

**La estatura reproyectada deja de ser un control independiente.** Hoy lo es -- los tiradores no
entran en la homografia, solo en la semilla -- y `control_estatura` lo documenta con razon. En
cuanto la estatura elige la focal, ese control pasa a ser el objetivo, y comprobarlo despues seria
el mismo error que el proyecto tiene documentado con el cono: *"la hoja esta a 0 grados de su propio
centro"*.

Los controles que SIGUEN siendo independientes despues de este cambio, y son los unicos que pueden
justificar adoptarlo:

    el ancho deducido dentro del rango FIE     no se impone, se comprueba
    el patron de marcas contra el reglamento   `control_marcas`, no toca a los tiradores
    `strip_location`                           verdad humana, 3.964 clips, externa a todo esto

**Criterio declarado antes de medir:** se adopta si sube el numero de clips con homografia
metricamente sana (hoy 10 de 78) **y** la posicion mejora contra `strip_location` (hoy pendiente
0,037, acierto 0,380 contra 0,430 de contestar siempre `middle`). Subir lo primero sin lo segundo no
vale: seria haber cambiado un numero por otro.
"""
from __future__ import annotations

import numpy as np

import pista as PI

OBJETIVO_H_M = 1.75
"""La estatura a la que se ajusta, y es la MEDIANA de la altura de caja, no la estatura de verdad.

`control_estatura` ya advierte por que no son lo mismo: *"la caja de YOLO no acaba en la coronilla
ni empieza en la planta del pie. En un fondo el tirador esta inclinado, asi que la caja mide menos
que su estatura"*. Se usa 1,75 sobre la MEDIANA porque es el convenio que el modulo ya emplea en
`mide_pista` (`ancho_real = ancho_m * 1,75 / h`), y mezclar dos convenios con el mismo nombre es el
fallo que este proyecto ya ha cometido cinco veces en un dia. `reestima` devuelve tambien el p90
para que se vea cuanto cambiaria el resultado con el otro convenio."""

RANGO_ANCHO_M = (1.40, 2.10)
"""El reglamento FIE dice 1,50-2,00 m. Se deja un 7 % de margen a cada lado porque el borde
detectado no es exactamente el canto de la plancha conductora."""

N_REJILLA = 140
MIN_APOYOS = 10


def _pares(res, nmax=60):
    """QUE DEVUELVE: lista de (pie_img, cab_img) en el sistema ESTABILIZADO, o [].

    Se toma el centro del borde inferior de la caja como pie y el del superior como cabeza, que es
    lo que hace `control_estatura`: aqui no se cambia el estimador, solo quien decide con el.
    """
    r, Mk = res.get("_r"), res.get("_Mk")
    if r is None:
        return []
    B, pres, mask, W, H = r["B"], r["pres"], r["mask"], r["W"], r["H"]
    ks = list(np.nonzero(mask)[0])
    if len(ks) > nmax:
        ks = [ks[i] for i in np.linspace(0, len(ks) - 1, nmax).astype(int)]
    out = []
    for k in ks:
        M = Mk[k] if (Mk is not None and k < len(Mk)) else np.eye(3)
        for lado in (0, 1):
            if not pres[k, lado] or not np.isfinite(B[k, lado]).all():
                continue
            b = B[k, lado]
            xc = 0.5 * (b[0] + b[2]) * W
            pie = M @ np.array([xc, b[3] * H, 1.0])
            cab = M @ np.array([xc, b[1] * H, 1.0])
            if abs(pie[2]) < 1e-9 or abs(cab[2]) < 1e-9:
                continue
            out.append((pie[:2] / pie[2], cab[:2] / cab[2]))
    return out


def _estatura(hgl, pares):
    """QUE DEVUELVE: (p50, p90, n) de la altura de caja reproyectada, en las unidades de la
    homografia que se le pase. Con ancho unitario, en anchos de pista."""
    v = []
    for pie, cab in pares:
        try:
            h = PI.altura_m(hgl, pie, cab)
        except Exception:
            h = None
        if h is not None and np.isfinite(h) and h > 0:
            v.append(h)
    if len(v) < MIN_APOYOS:
        return None, None, len(v)
    v = np.asarray(v)
    return float(np.median(v)), float(np.percentile(v, 90)), len(v)


def reestima(res, objetivo=OBJETIVO_H_M, rango_ancho=RANGO_ANCHO_M):
    """QUE DEVUELVE: dict con `ok`, `f`, `f_rel`, `ancho_deducido_m`, `hc_m`, `h_p50`, `h_p90`,
    `f_rel_antes`, `ancho_antes_m`, `h_p50_antes`, `n_candidatas`, o `motivo` si no hay ninguna
    focal que cumpla las dos condiciones fisicas.

    NO ESCRIBE NADA Y NO TOCA `pista.py`. La extraccion esta corriendo y cambiar el criterio a
    mitad dejaria el corpus con dos convenios mezclados, que es exactamente el fallo que el
    proyecto tiene documentado como "el mismo nombre con un numero distinto".
    """
    hg, bo, mar, r = res.get("_hg"), res.get("_bo"), res.get("_mar"), res.get("_r")
    if not (hg and hg.get("ok") and bo and bo.get("ok") and r):
        return {"ok": False, "motivo": "el clip no tiene homografia de partida"}
    W, H = r["W"], r["H"]
    pp = PI._pp(W, H)
    V1 = np.asarray(hg["V1"], "float64")
    Vz = np.asarray(hg["Vz"], "float64")
    LAc = PI._a_centrada(bo["recta_A"], pp)
    LBc = PI._a_centrada(bo["recta_B"], pp)
    usada = hg.get("marca_usada")
    if usada is None:
        return {"ok": False, "motivo": "sin marca de origen"}
    Oc = np.cross(PI._a_centrada(usada["L"], pp), LAc)
    if abs(Oc[2]) < 1e-12:
        return {"ok": False, "motivo": "la marca de origen no corta el borde A"}
    Oc = Oc / Oc[2]
    pares = _pares(res)
    if len(pares) < MIN_APOYOS:
        return {"ok": False, "motivo": "menos de %d apoyos" % MIN_APOYOS}
    Ls = [li["L"] for li in ((mar.get("lineas") if mar else None) or []) if li.get("afinada")]
    A = np.asarray(Ls) if Ls else None
    dentro = (np.asarray(bo["centro_pista"], "float64") - pp) \
        if bo.get("centro_pista") is not None else None
    """El mismo `dentro_c` que usa `homografia`: un punto que se sabe DENTRO de la pista, en
    coordenadas centradas. Sirve para exigir que V2 caiga por encima de la pista y no por debajo,
    que es la puerta fisica que ya existe y que aqui hay que respetar igual."""

    cand = []
    for f in np.exp(np.linspace(np.log(PI.F_REL_RANGO[0] * W),
                                np.log(PI.F_REL_RANGO[1] * W), N_REJILLA)):
        w = np.array([1.0 / (f * f), 1.0 / (f * f), 1.0])
        V2 = np.cross(w * V1, w * Vz)
        nv = np.linalg.norm(V2)
        if nv < 1e-14:
            continue
        V2 = V2 / nv
        if dentro is not None and not PI._v2_posible_simple(V2, LAc, W, dentro):
            continue
        con = PI._construye_H(V1, V2, f, LBc, Oc, 1.0, pp)
        if not con.get("ok"):
            continue
        hgl = {"H": con["H"], "Hi": con["Hi"], "vz_h": con["vz_h"]}
        s50, s90, n = _estatura(hgl, pares)
        if not s50 or s50 <= 1e-6:
            continue
        ancho = objetivo / s50
        """AQUI ES DONDE EL ANCHO DEJA DE IMPONERSE. Hoy vale 1,50 por decreto y la estatura sale
        de 1,07 a 4,16 m; al reves, la estatura fija la escala y el ancho queda DEDUCIDO, con lo
        que el rango reglamentario se convierte en una restriccion que descarta focales."""
        if not (rango_ancho[0] <= ancho <= rango_ancho[1]):
            continue
        hc = con["hc_m"] * ancho
        if not (PI.ALTURA_CAMARA_M[0] <= hc <= PI.ALTURA_CAMARA_M[1]):
            continue
        if A is not None and len(A):
            ang = PI._angulos(A, V2)
            inl = int((ang <= PI.ANG_INLIER_V2).sum())
            med = float(np.median(ang[ang <= PI.ANG_INLIER_V2])) if inl else float(np.median(ang))
        else:
            inl, med = 0, 90.0
        cand.append({"f": float(f), "f_rel": float(f / W), "V2": V2,
                     "ancho_deducido_m": float(ancho), "hc_m": float(hc),
                     "h_p50": objetivo, "h_p90_rel": float(s90 / s50) if s90 else float("nan"),
                     "n_alturas": n, "inl": inl, "ang_p50": med})
    if not cand:
        return {"ok": False, "motivo": "ninguna focal da ancho en %.2f-%.2f m con camara en "
                                       "%.1f-%.1f m" % (rango_ancho + PI.ALTURA_CAMARA_M),
                "n_candidatas": 0}
    cand.sort(key=lambda c: (-c["inl"], c["ang_p50"]))
    """ENTRE LAS QUE PASAN LA FISICA, DECIDE EL CRITERIO DE SIEMPRE. Asi el cambio se puede leer
    como lo que es: no se sustituye el criterio angular, se le quita el trabajo que no podia hacer.
    El rango de focales que sobrevive se devuelve entero para poder decir si la focal queda MEDIDA
    o solo acotada -- que es la misma pregunta que `curva_f` hacia y casi nadie miraba."""
    m = cand[0]
    fr = np.array([c["f_rel"] for c in cand])
    ch_antes = {}
    try:
        ch_antes = PI.control_estatura(hg, r, res.get("_Mk")) or {}
    except Exception:
        pass
    h_antes = ch_antes.get("h_p50")
    return {"ok": True, "f": m["f"], "f_rel": m["f_rel"], "V2": m["V2"],
            "ancho_deducido_m": m["ancho_deducido_m"], "hc_m": m["hc_m"],
            "h_p90_sobre_p50": m["h_p90_rel"], "n_alturas": m["n_alturas"],
            "inl": m["inl"], "ang_p50": m["ang_p50"],
            "n_candidatas": len(cand),
            "f_rel_min": float(fr.min()), "f_rel_max": float(fr.max()),
            "focal_medida": bool(fr.max() / max(fr.min(), 1e-9) <= 1.30),
            "f_rel_antes": float(hg["f"] / W),
            "ancho_antes_m": float(hg.get("ancho_m", PI.ANCHO_NOMINAL)),
            "h_p50_antes": float(h_antes) if h_antes else float("nan")}


def homografia_corregida(res, objetivo=OBJETIVO_H_M, rango_ancho=RANGO_ANCHO_M):
    """QUE DEVUELVE: (hg_nueva, info) con la homografia reconstruida con la focal y el ancho que
    salen de `reestima`, o (None, info). `hg_nueva` tiene la misma forma que la de `pista` para que
    `a_metros`, `altura_m`, `tira_rectificada` y la posicion funcionen sin cambiar nada."""
    info = reestima(res, objetivo, rango_ancho)
    if not info.get("ok"):
        return None, info
    hg, bo, r = res["_hg"], res["_bo"], res["_r"]
    W, H = r["W"], r["H"]
    pp = PI._pp(W, H)
    LAc = PI._a_centrada(bo["recta_A"], pp)
    LBc = PI._a_centrada(bo["recta_B"], pp)
    Oc = np.cross(PI._a_centrada(hg["marca_usada"]["L"], pp), LAc)
    Oc = Oc / Oc[2]
    con = PI._construye_H(np.asarray(hg["V1"], "float64"), info["V2"], info["f"],
                          LBc, Oc, info["ancho_deducido_m"], pp)
    if not con.get("ok"):
        return None, {"ok": False, "motivo": con.get("motivo", "no construye")}
    nuevo = dict(hg)
    nuevo.update({"H": con["H"], "Hi": con["Hi"], "hc_m": con["hc_m"], "vz_h": con["vz_h"],
                  "mu": con["mu"], "K": con["K"], "f": info["f"], "V2": info["V2"],
                  "ancho_m": info["ancho_deducido_m"], "fuente_f": "estatura + ancho FIE"})
    return nuevo, info


def reescala_por_estatura(res, objetivo=OBJETIVO_H_M, rango_ancho=RANGO_ANCHO_M):
    """QUE HACE: corrige la ESCALA METRICA de cualquier homografia usando la estatura, sin tocar su
    forma proyectiva. Vale para las dos vias, tambien para las 23 de 43 que vienen del DLT y no
    tienen V1 ni Vz que barrer.

    LA TRANSFORMACION, y es exacta. Escalar el mundo por `s` es

        H_nueva = H . diag(1/s, 1/s, 1)        vz_nueva = vz / s

    Con eso, para un mismo punto de imagen `p_pie = H (X,Y,1)` no cambia -- porque las coordenadas
    del mundo se agrandan justo lo que la matriz encoge -- y como
    `h = -(m x p_pie)_i / (m x vz)_i`, dividir `vz` por `s` multiplica TODAS las alturas por `s`.
    Las distancias sobre el suelo se multiplican por `s` por el mismo motivo. O sea que una sola
    constante mueve la escala entera y nada mas: los puntos de fuga, el horizonte y la forma
    proyectiva quedan intactos.

    POR QUE ESTO ARREGLA LO GORDO. La estatura reproyectada hoy va de 1,07 a 4,16 m cuando deberia
    ser 1,75: es un error de escala de hasta 4x, y un error de escala en la pista es exactamente un
    error de POSICION -- un tirador a 3 m del centro sale a 12 m, o sea fuera de una pista de 14.

    LO QUE NO ARREGLA: la FORMA. Si la focal esta mal, la perspectiva esta mal y el error no es un
    factor constante a lo largo de la pista. Por eso el ancho deducido se devuelve y se comprueba
    contra el rango FIE: es lo unico independiente que queda para detectar ese caso.
    """
    hg, r = res.get("_hg"), res.get("_r")
    if not (hg and hg.get("ok") and r):
        return None, {"ok": False, "motivo": "el clip no tiene homografia"}
    pares = _pares(res)
    if len(pares) < MIN_APOYOS:
        return None, {"ok": False, "motivo": "menos de %d apoyos" % MIN_APOYOS}
    h50, h90, n = _estatura(hg, pares)
    if not h50 or h50 <= 1e-6:
        return None, {"ok": False, "motivo": "sin estatura reproyectable"}
    s = objetivo / h50
    Hn = np.asarray(hg["H"], "float64") @ np.diag([1.0 / s, 1.0 / s, 1.0])
    try:
        Hin = np.linalg.inv(Hn)
    except np.linalg.LinAlgError:
        return None, {"ok": False, "motivo": "homografia escalada singular"}
    nuevo = dict(hg)
    nuevo.update({"H": Hn, "Hi": Hin, "vz_h": np.asarray(hg["vz_h"], "float64") / s,
                  "hc_m": float(hg.get("hc_m", float("nan"))) * s,
                  "ancho_m": float(hg.get("ancho_m", PI.ANCHO_NOMINAL)) * s,
                  "fuente_escala": "estatura reproyectada = %.2f m" % objetivo})
    ancho = float(nuevo["ancho_m"])
    info = {"ok": True, "s": float(s), "h_p50_antes": float(h50),
            "h_p90_antes": float(h90) if h90 else float("nan"), "n_alturas": n,
            "ancho_deducido_m": ancho, "hc_m": float(nuevo["hc_m"]),
            "ancho_en_rango_fie": bool(rango_ancho[0] <= ancho <= rango_ancho[1]),
            "hc_en_rango": bool(PI.ALTURA_CAMARA_M[0] <= nuevo["hc_m"] <= PI.ALTURA_CAMARA_M[1])}
    """`ancho_en_rango_fie` ES EL CONTROL QUE SOBREVIVE AL CAMBIO. La estatura ya no puede validar
    nada porque es el objetivo; el ancho de la pista no se ha usado para ajustar y sigue siendo una
    prediccion que el reglamento puede desmentir."""
    return nuevo, info


MIN_POSITIVAS = 0.95
DISP_MAX = 0.35


def _camara_de_H(Hm, f, pp, ancho_m):
    """QUE DEVUELVE: (H con signo arreglado, vz_h, hc_m) para una focal dada. Es la misma
    descomposicion que hace `pista.completa`, extraida para poder barrerla."""
    K = np.array([[f, 0.0, pp[0]], [0.0, f, pp[1]], [0.0, 0.0, 1.0]])
    A = np.linalg.inv(K) @ Hm
    mu = 0.5 * (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if mu < 1e-15:
        return None, None, None
    Hs = Hm.copy()
    if float((Hs @ np.array([0.0, 0.5 * ancho_m, 1.0]))[2]) < 0:
        Hs = -Hs
        A = np.linalg.inv(K) @ Hs
        mu = 0.5 * (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2 = A[:, 0] / mu, A[:, 1] / mu
    r3 = np.cross(r1, r2)
    tv = A[:, 2] / mu
    vzh = mu * (K @ r3)
    p0 = Hs @ np.array([0.0, 0.5 * ancho_m, 1.0])
    p1 = p0 + vzh
    if p1[2] != 0 and p0[2] != 0 and (p1[1] / p1[2]) > (p0[1] / p0[2]):
        vzh = -vzh
    return Hs, vzh, float(abs(r3 @ tv))


def focal_por_constancia(res, objetivo=OBJETIVO_H_M, rango_ancho=RANGO_ANCHO_M):
    """QUE HACE: elige la focal por CONSTANCIA de la estatura reproyectada, y despues fija la escala
    para que la mediana valga 1,75 m.

    ## POR QUE LA CONSTANCIA Y NO LA MEDIANA

    Medido sobre 11 clips con homografia, con TODAS las alturas y sin el filtro 0,3-5,0 m que
    `control_estatura` aplica antes de publicar su mediana:

        AbandonedAntiqueGalapagostortoise   p10 -11,18   p90  16,27   supervivencia 0,14
        AbandonedBiodegradableCoqui         p10 -11,18   p90  28,39   supervivencia 0,28
        AbandonedUnderstatedGalago          p10   1,66   p90   2,23   supervivencia 1,00

    Hay tiradores con estatura **negativa**. Eso no es una escala mal puesta -- una escala mal
    puesta multiplica, no cambia el signo -- es el punto de fuga vertical del lado equivocado, o sea
    la estructura proyectiva rota. Y una mediana no lo ve: la de Coqui salia 1,76 m y pasaba la
    puerta `1,60-2,00` con dos tercios de las observaciones tiradas a la basura.

    Lo que si lo ve es la DISPERSION. Un tirador mide lo mismo este donde este en la pista, asi que
    con la homografia correcta su estatura reproyectada es **constante** mientras camina de un
    extremo al otro. Con la focal equivocada varia con la posicion. O sea:

        la DISPERSION mide la FORMA proyectiva   ->   elige la focal
        la MEDIANA mide la ESCALA                ->   elige el ancho de pista

    Dos cosas distintas, dos parametros distintos, y separarlas es lo que hace que ninguna de las
    dos tape a la otra. Ademas la dispersion es **insensible a la escala**, asi que elegir la focal
    con ella no es circular con el paso siguiente.

    ## EL CONTROL QUE SOBREVIVE

    La estatura ya no puede validar nada: es el objetivo. Queda el **ancho de pista deducido**, que
    no se ha usado para ajustar nada y que el reglamento FIE acota a 1,50-2,00 m. Si sale 0,35 m,
    la solucion es mala por mucho que la estatura salga clavada en 1,75.

    QUE DEVUELVE: (hg_nueva, info) o (None, info).
    """
    hg, r = res.get("_hg"), res.get("_r")
    if not (hg and hg.get("ok") and r):
        return None, {"ok": False, "motivo": "el clip no tiene homografia"}
    W, Himg = r["W"], r["H"]
    pp = PI._pp(W, Himg)
    Hm = np.asarray(hg["H"], "float64")
    ancho0 = float(hg.get("ancho_m", PI.ANCHO_NOMINAL))
    pares = _pares(res, nmax=120)
    if len(pares) < MIN_APOYOS:
        return None, {"ok": False, "motivo": "menos de %d apoyos" % MIN_APOYOS}

    cand = []
    for f in np.exp(np.linspace(np.log(PI.F_REL_RANGO[0] * W),
                                np.log(PI.F_REL_RANGO[1] * W), 200)):
        Hs, vzh, hc0 = _camara_de_H(Hm, f, pp, ancho0)
        if Hs is None:
            continue
        try:
            Hi = np.linalg.inv(Hs)
        except np.linalg.LinAlgError:
            continue
        hgl = {"H": Hs, "Hi": Hi, "vz_h": vzh}
        v = []
        for pie, cab in pares:
            try:
                h = PI.altura_m(hgl, pie, cab)
            except Exception:
                h = None
            if h is not None and np.isfinite(h):
                v.append(float(h))
        if len(v) < MIN_APOYOS:
            continue
        v = np.asarray(v)
        pos = float((v > 0).mean())
        if pos < MIN_POSITIVAS:
            continue
        """SIN ESTE FILTRO NO HAY NADA QUE HACER: si el 5 % de los tiradores mide menos de cero, la
        vertical esta del lado equivocado y la homografia no describe esa camara."""
        p10, p50, p90 = (float(np.percentile(v, 10)), float(np.median(v)),
                         float(np.percentile(v, 90)))
        if p50 <= 1e-6:
            continue
        disp = (p90 - p10) / p50
        s = objetivo / p50
        ancho = ancho0 * s
        hc = hc0 * s
        cand.append({"f": float(f), "f_rel": float(f / W), "disp": disp, "frac_pos": pos,
                     "p50": p50, "ancho_deducido_m": ancho, "hc_m": hc,
                     "Hs": Hs, "vzh": vzh, "s": s})
    if not cand:
        return None, {"ok": False, "motivo": "ninguna focal da estaturas positivas en el %.0f %% "
                                             "de los cuadros" % (MIN_POSITIVAS * 100)}
    fisicas = [c for c in cand
               if rango_ancho[0] <= c["ancho_deducido_m"] <= rango_ancho[1]
               and PI.ALTURA_CAMARA_M[0] <= c["hc_m"] <= PI.ALTURA_CAMARA_M[1]]
    usadas, filtrado = (fisicas, True) if fisicas else (cand, False)
    m = min(usadas, key=lambda c: c["disp"])
    Hn = m["Hs"] @ np.diag([1.0 / m["s"], 1.0 / m["s"], 1.0])
    try:
        Hin = np.linalg.inv(Hn)
    except np.linalg.LinAlgError:
        return None, {"ok": False, "motivo": "homografia escalada singular"}
    nuevo = dict(hg)
    nuevo.update({"H": Hn, "Hi": Hin, "vz_h": np.asarray(m["vzh"], "float64") / m["s"],
                  "f": m["f"], "hc_m": m["hc_m"], "ancho_m": m["ancho_deducido_m"],
                  "fuente_f": "constancia de la estatura"})
    info = {"ok": True, "f_rel": m["f_rel"], "f_rel_antes": float(hg["f"]) / W,
            "dispersion": m["disp"], "frac_positivas": m["frac_pos"],
            "ancho_deducido_m": m["ancho_deducido_m"], "hc_m": m["hc_m"],
            "ancho_en_rango_fie": bool(rango_ancho[0] <= m["ancho_deducido_m"] <= rango_ancho[1]),
            "n_focales": len(cand), "n_fisicas": len(fisicas), "paso_fisica": filtrado,
            "estable": bool(m["disp"] <= DISP_MAX), "n_apoyos": len(pares)}
    """`estable` ES EL VEREDICTO. Con la dispersion por encima de %.2f la estatura reproyectada
    varia mas de un tercio segun donde este el tirador, y entonces la posicion que salga de esa
    homografia no vale aunque la mediana este clavada.""" % DISP_MAX
    return nuevo, info
