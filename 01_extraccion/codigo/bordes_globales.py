"""Los dos bordes de la pista elegidos del haz de V1 y decididos por FISICA, no por intensidad.

## QUE SUSTITUYE Y POR QUE

`pista.region_y_bordes` barre perpendicularmente al corredor de los pies, hasta `ALCANCE_BARRIDO`
altos de caja, y se queda con el primer corte de color a cada lado. Ese diseno tiene una razon
medida y buena -- la componente conexa se derramaba a las cuatro pistas del pabellon en
`UglyNiceAplomadofalcon` -- pero paga un precio que nadie habia medido: **descarta todas las lineas
del recinto antes de mirarlas**, y con ellas el unico sitio donde hay evidencia de sobra.

Medido sobre 14 clips:

    clips con punto de fuga longitudinal, metodo de hoy         4 de 14   (29 %)
    clips con V1 por lineas globales del pabellon               9 de 14   (64 %)

y de esos 9, **6 fallan hoy**. El haz encontrado apunta ademas en **0,1 a 5,3 grados** de la
direccion por la que caminan los tiradores, que es una confirmacion independiente: esa direccion no
se usa para encontrar el haz, solo para elegir entre candidatos.

## COMO SE ELIGE LA PAREJA DE BORDES, y por que no por intensidad

Del haz de V1 salen entre 5 y 15 rectas distintas (agrupando las casi coincidentes). Los bordes de
la pista son **dos de ellas**. Cual par, lo deciden tres hechos fisicos que se cierran a la vez:

    1. ENCIERRAN LOS APOYOS      los tiradores estan sobre la pista mientras tiran; no es una
                                 heuristica, es la definicion del deporte
    2. ESTATURA CONSTANTE        con la homografia que sale de ese par, un tirador tiene que medir
                                 lo mismo este donde este. Mide la FORMA proyectiva y es insensible
                                 a la escala
    3. ANCHO REGLAMENTARIO       el ancho deducido -- no impuesto -- cae en 1,40-2,10 m

Ninguna de las tres se elige a ojo y ninguna es un umbral de intensidad. La 2 y la 3 son las de
`focal_fisica`, aplicadas un nivel mas arriba: alli elegian la focal con los bordes dados, aqui
eligen ademas los bordes.

## LO QUE ESTE MODULO NO ARREGLA

Planos muy cerrados con pocas lineas. `AbleHilariousAurochs` tiene **9** segmentos largos en todo el
cuadro y ahi no hay haz que sacar; el barrido local si funciona. Por eso esto **no sustituye** a
`region_y_bordes`: las dos vias se prueban y gana la que pase los controles fisicos, y si pasan las
dos, la de mas evidencia. Sustituir un metodo que funciona en un subconjunto por otro que funciona
en otro subconjunto perderia clips en silencio.
"""
from __future__ import annotations

import numpy as np

import focal_fisica as FF
import lineas_globales as LG
import pista as PI

TOL_AGRUPA_PX = 12.0
"""Dos rectas del haz son la misma si sus cortes con la perpendicular al corredor distan menos de
esto. Sin agrupar, un borde con 20 segmentos detectados genera 20 candidatas identicas y el
producto cartesiano explota sin anadir ninguna hipotesis nueva."""

RAZON_SEP = (1.2, 6.0)
"""Separacion admisible entre los dos bordes, en anchos de la banda que recorren los apoyos. Los
pies de los dos tiradores barren del orden de 0,6 m a lo ancho y la pista mide 1,5-2,0 m, o sea unas
2,5-3,3 veces. La ventana 1,2-6,0 es deliberadamente ancha: aqui solo hay que descartar lo absurdo,
porque quien decide de verdad son los dos controles fisicos de despues."""

N_FOCALES = 48
MAX_PARES = 24


def _perp(co):
    """La coordenada perpendicular al corredor: `(c0, n)` tales que `d = (p - c0) . n`."""
    return np.asarray(co["c0"], "float64"), np.asarray(co["n"], "float64")


def candidatas(S, inl, co, W, H, V1=None):
    """QUE DEVUELVE: lista de `{L, d, n}` -- recta homogenea, offset perpendicular al corredor y
    cuantos segmentos la sostienen -- agrupando las del haz que son la misma recta."""
    c0, n = _perp(co)
    grupos = []
    for k in np.nonzero(inl)[0]:
        x0, y0, x1, y1 = S[k]
        L = np.cross([x0, y0, 1.0], [x1, y1, 1.0])
        if abs(L[0]) < 1e-12 and abs(L[1]) < 1e-12:
            continue
        p = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        d = float((p - c0) @ n)
        for g in grupos:
            if abs(float(np.median(g["ds"])) - d) <= TOL_AGRUPA_PX:
                g["Ls"].append(L)
                g["ds"].append(d)
                break
        else:
            grupos.append({"Ls": [L], "ds": [d]})
    out = []
    for g in grupos:
        d = float(np.median(g["ds"]))
        if V1 is not None:
            Lm = np.cross(np.asarray(V1, "float64"), np.array([c0[0] + d * n[0],
                                                               c0[1] + d * n[1], 1.0]))
            """LA RECTA SE CONSTRUYE POR V1, NO SE PROMEDIA. Promediando las rectas del grupo sale
            una que NO pasa por V1, y entonces `_construye_H` recibe una terna (V1, borde A, borde
            B) INCONSISTENTE -- porque el modulo calcula V1 como `cross(LAc, LBc)` y ahi eso ya no
            se cumple. Medido: con las rectas promediadas, las 48 focales daban estaturas negativas
            de 0,005 anchos de pista en las 24 parejas de los 14 clips. Cero de cero.

            Construyendo `L = V1 x p`, los dos bordes se cortan en V1 por construccion y la
            identidad que `_construye_H` da por supuesta se cumple exactamente."""
        else:
            Lm = np.mean([L / (np.linalg.norm(L[:2]) + 1e-12) for L in g["Ls"]], axis=0)
        nn = np.linalg.norm(Lm[:2])
        if nn < 1e-12:
            continue
        out.append({"L": Lm / nn, "d": d, "n": len(g["Ls"])})
    out.sort(key=lambda g: g["d"])
    return out


def pares_que_encierran(cands, co):
    """QUE DEVUELVE: pares (A, B) de candidatas que dejan la banda de los apoyos ENTRE las dos, con
    separacion plausible, ordenados por evidencia (segmentos que las sostienen)."""
    t0, t1 = co["t"]
    ancho_pies = max(abs(t1 - t0), 1.0)
    pares = []
    for i, a in enumerate(cands):
        if a["d"] > t0:
            continue
        for b in cands[i + 1:]:
            if b["d"] < t1:
                continue
            sep = b["d"] - a["d"]
            r = sep / ancho_pies
            if not (RAZON_SEP[0] <= r <= RAZON_SEP[1]):
                continue
            pares.append({"A": a, "B": b, "sep_px": float(sep), "razon": float(r),
                          "evid": a["n"] + b["n"]})
    pares.sort(key=lambda p: -p["evid"])
    return pares[:MAX_PARES]


def _evalua_par(par, res, V1, Vz, W, H, pares_alt):
    """QUE DEVUELVE: (dispersion de la estatura, ancho deducido, focal) del mejor ajuste para este
    par de bordes, o (inf, nan, nan).

    Es el criterio de `focal_fisica.focal_por_constancia` aplicado a una pareja de bordes
    CANDIDATA: se barre la focal, se exige que las estaturas salgan positivas, y se mide lo
    constante que es la estatura. La dispersion mide la forma proyectiva y no depende de la escala,
    asi que sirve para comparar parejas distintas sin que la escala las confunda.
    """
    pp = PI._pp(W, H)
    LAc = PI._a_centrada(par["A"]["L"], pp)
    LBc = PI._a_centrada(par["B"]["L"], pp)
    V1c = np.asarray(V1, "float64")
    Vzc = np.asarray(Vz, "float64")
    Oc = np.cross(LAc, np.cross(V1c, np.array([0.0, 0.0, 1.0])))
    if abs(Oc[2]) < 1e-12:
        Oc = np.array([0.0, 0.0, 1.0])
    else:
        Oc = Oc / Oc[2]
    """EL ORIGEN DEL MUNDO DA IGUAL AQUI: solo desplaza `q` y no cambia ni la dispersion de la
    estatura ni el ancho deducido, que son las dos cosas que deciden. Se toma un punto cualquiera
    del borde A para que `_construye_H` tenga algo con que anclar."""
    mejor = (float("inf"), float("nan"), float("nan"))
    for f in np.exp(np.linspace(np.log(PI.F_REL_RANGO[0] * W),
                                np.log(PI.F_REL_RANGO[1] * W), N_FOCALES)):
        w = np.array([1.0 / (f * f), 1.0 / (f * f), 1.0])
        V2 = np.cross(w * V1c, w * Vzc)
        nv = np.linalg.norm(V2)
        if nv < 1e-14:
            continue
        V2 = V2 / nv
        con = PI._construye_H(V1c, V2, f, LBc, Oc, 1.0, pp)
        if not con.get("ok"):
            continue
        try:
            Hi = np.linalg.inv(con["H"])
        except np.linalg.LinAlgError:
            continue
        hgl = {"H": con["H"], "Hi": Hi, "vz_h": con["vz_h"]}
        v = []
        for pie, cab in pares_alt:
            try:
                h = PI.altura_m(hgl, pie, cab)
            except Exception:
                h = None
            if h is not None and np.isfinite(h):
                v.append(float(h))
        if len(v) < FF.MIN_APOYOS:
            continue
        v = np.asarray(v)
        if (v > 0).mean() < FF.MIN_POSITIVAS:
            continue
        p10, p50, p90 = (float(np.percentile(v, 10)), float(np.median(v)),
                         float(np.percentile(v, 90)))
        if p50 <= 1e-6:
            continue
        disp = (p90 - p10) / p50
        ancho = FF.OBJETIVO_H_M / p50
        if not (FF.RANGO_ANCHO_M[0] <= ancho <= FF.RANGO_ANCHO_M[1]):
            continue
        if disp < mejor[0]:
            mejor = (disp, ancho, float(f))
    return mejor


def bordes(res, verboso=False):
    """QUE DEVUELVE: dict con la MISMA forma que `pista.region_y_bordes` -- `ok`, `n_col`,
    `centro_pista`, `recta_A/B`, `res_A/B`, `pts_A/B` -- mas `V1`, `Vz`, `dispersion`,
    `ancho_deducido_m` y `f_rel`, o `motivo`.

    Misma forma a proposito: asi se puede probar contra la via de hoy sobre el mismo clip sin tocar
    `pista.py`, y la diferencia que se mida sera de los bordes y de nada mas.
    """
    pl = res.get("_pl")
    co = res.get("_co")
    if pl is None:
        return {"ok": False, "motivo": "sin placa limpia"}
    if co is None or "e" not in co:
        return {"ok": False, "motivo": "sin corredor de apoyos"}
    H, W = pl.shape[:2]
    S = LG.segmentos(pl, mascara=res.get("_rot"))
    if len(S) < 8:
        return {"ok": False, "motivo": "solo %d segmentos largos en el cuadro" % len(S)}
    vps = LG.puntos_de_fuga(S)
    cen = np.array([W / 2.0, H / 2.0])
    v1 = LG.elige_v1(vps, co["e"], cen)
    if v1 is None:
        return {"ok": False, "motivo": "ningun haz compatible con la marcha de los tiradores",
                "n_vps": len(vps), "n_segm": len(S)}
    vz = None
    for v in vps:
        if v is v1 or np.allclose(v["P"], v1["P"]):
            continue
        d = np.asarray(v["P"], "float64") - cen
        if abs(d[1]) > abs(d[0]):
            vz = v
            break
    """EL HAZ VERTICAL es el que se va hacia arriba o hacia abajo del cuadro, no hacia los lados.
    Si no hay ninguno, se cae a la vertical de imagen, que es lo que asume una camara sin balanceo
    -- y eso queda declarado en la salida para que nadie lo lea como medido."""
    Vz = (np.array([vz["P"][0], vz["P"][1], 1.0]) if vz is not None
          else np.array([0.0, 1.0, 0.0]))
    V1 = np.array([v1["P"][0], v1["P"][1], 1.0])
    pp = PI._pp(W, H)
    V1c = np.array([V1[0] - pp[0], V1[1] - pp[1], 1.0])
    Vzc = (np.array([Vz[0] - pp[0] * Vz[2], Vz[1] - pp[1] * Vz[2], Vz[2]])
           if abs(Vz[2]) > 1e-12 else Vz.copy())

    cands = candidatas(S, v1["inl"], co, W, H, V1=V1)
    if len(cands) < 2:
        return {"ok": False, "motivo": "el haz de V1 solo da %d recta(s) distintas" % len(cands)}
    pares = pares_que_encierran(cands, co)
    if not pares:
        return {"ok": False, "motivo": "ninguna pareja del haz encierra los apoyos con separacion "
                                       "plausible", "n_candidatas": len(cands)}
    pares_alt = FF._pares(res, nmax=60)
    if len(pares_alt) < FF.MIN_APOYOS:
        return {"ok": False, "motivo": "menos de %d apoyos" % FF.MIN_APOYOS}

    mejor = None
    for p in pares:
        disp, ancho, f = _evalua_par(p, res, V1c, Vzc, W, H, pares_alt)
        inv = {"A": p["B"], "B": p["A"], "sep_px": p["sep_px"], "razon": p["razon"],
               "evid": p["evid"]}
        d2, a2, f2 = _evalua_par(inv, res, V1c, Vzc, W, H, pares_alt)
        if d2 < disp:
            disp, ancho, f, p = d2, a2, f2, inv
        """SE PRUEBAN LOS DOS ORDENES. Cual de los dos bordes se llama A es arbitrario, pero
        `_construye_H` ancla el origen en A y pone B en `y = ancho_m`, asi que el orden fija el
        signo de la normal del plano y con el, el de las alturas. Probar los dos y quedarse con el
        que da estaturas positivas no es elegir el resultado que gusta: el otro describe la misma
        camara mirando la pista desde debajo del suelo."""
        if verboso:
            print("   par sep=%.0f px razon=%.2f evid=%d -> disp %s ancho %s"
                  % (p["sep_px"], p["razon"], p["evid"],
                     ("%.3f" % disp) if np.isfinite(disp) else "-",
                     ("%.2f" % ancho) if np.isfinite(ancho) else "-"))
        if np.isfinite(disp) and (mejor is None or disp < mejor[0]):
            mejor = (disp, ancho, f, p)
    if mejor is None:
        return {"ok": False, "motivo": "ninguna pareja da estaturas positivas con ancho FIE",
                "n_pares": len(pares), "n_candidatas": len(cands)}
    disp, ancho, f, p = mejor
    LA, LB = p["A"]["L"], p["B"]["L"]
    c0, n = _perp(co)
    centro = c0 + 0.5 * (p["A"]["d"] + p["B"]["d"]) * n
    return {"ok": True, "n_col": int(p["evid"]),
            "centro_pista": centro,
            "recta_A": LA, "recta_B": LB,
            "res_A": 0.0, "res_B": 0.0,
            "pts_A": np.zeros((0, 2)), "pts_B": np.zeros((0, 2)),
            "V1": V1, "Vz": Vz, "vz_supuesta": bool(vz is None),
            "dispersion": float(disp), "ancho_deducido_m": float(ancho),
            "f_rel": float(f / W), "sep_px": p["sep_px"],
            "n_segm": int(len(S)), "n_candidatas": len(cands), "n_pares": len(pares),
            "v1_inl": int(v1["n"]), "v1_ang_pies_deg": float(v1["ang_con_pies_deg"]),
            "via_bordes": "haz de V1 global + estatura constante + ancho FIE"}


def refina_con_v1_global(res, verboso=False):
    """QUE HACE: deja los bordes DONDE el barrido local los encontro y les impone la DIRECCION que
    dan las lineas globales del pabellon.

    ## POR QUE ESTE HIBRIDO Y NO SUSTITUIR EL BARRIDO

    Intente elegir los dos bordes de entre las lineas globales y no funciono en ninguno de 14 clips.
    La razon, medida: **los cantos de la pista no estan entre los segmentos largos**. En
    `AbandonedUnderstatedGalago` la pista mide **26 px de ancho** y sus bordes son cortos y de bajo
    contraste; el haz global lo dominan las vallas y los paneles, que estan lejos. El barrido local
    de `region_y_bordes` si los encuentra -- para eso se escribio, y su docstring cuenta el fallo
    medido que lo puso ahi.

    Lo que el barrido hace MAL es otra cosa: ajusta las dos rectas **por separado**, y dos rectas
    cortas y casi paralelas determinan pesimamente su punto de corte. Medido:

        clip                           V1 de los bordes de hoy      V1 del haz global
        AbandonedBiodegradableCoqui            x =  39.523          x =   8.282   (75 lineas)
        AbandonedUnderstatedGalago             x = -40.926          x = -237.574  (48 lineas)

    Factores de 5 y 6. Y V1 es lo que fija la familia de focales en `_f_y_v2`, asi que un V1
    inestable es exactamente la causa de que `f_rel` acabe pegada a los topes 0,30 y 6,00 -- el
    discriminador que separaba los 10 clips buenos de los 33 malos.

    ## LO QUE SE HACE

    Cada borde se REAJUSTA como la recta que pasa por el V1 global y minimiza la distancia a los
    puntos que el barrido encontro. Un grado de libertad en vez de dos, y el que se quita es
    justamente el mal determinado. Los dos bordes se cortan entonces en V1 por construccion.

    QUE DEVUELVE: (bordes nuevos, info) con la misma forma que `pista.region_y_bordes`, o
    (None, info).
    """
    bo, co, pl = res.get("_bo"), res.get("_co"), res.get("_pl")
    if not (bo and bo.get("ok")) or pl is None or co is None:
        return None, {"ok": False, "motivo": "sin bordes locales de partida"}
    H, W = pl.shape[:2]
    pp = PI._pp(W, H)
    S = LG.segmentos(pl, mascara=res.get("_rot"))
    if len(S) < 8:
        return None, {"ok": False, "motivo": "solo %d segmentos largos" % len(S)}
    vps = LG.puntos_de_fuga(S)
    v1 = LG.elige_v1(vps, co["e"], np.array([W / 2.0, H / 2.0]))
    if v1 is None:
        return None, {"ok": False, "motivo": "ningun haz compatible con la marcha",
                      "n_vps": len(vps)}
    Vimg = np.array([v1["P"][0], v1["P"][1], 1.0])
    nuevo = dict(bo)
    resid = {}
    for nom in ("A", "B"):
        pts = np.asarray(bo.get("pts_" + nom))
        if pts is None or len(pts) < 4:
            return None, {"ok": False, "motivo": "el barrido no dejo puntos del borde " + nom}
        L = _recta_por_punto_y_vp(pts, Vimg)
        if L is None:
            return None, {"ok": False, "motivo": "no se puede anclar el borde " + nom + " a V1"}
        nuevo["recta_" + nom] = L
        d = np.abs(pts @ L[:2] + L[2]) / (np.linalg.norm(L[:2]) + 1e-12)
        resid[nom] = float(np.median(d))
        nuevo["res_" + nom] = resid[nom]
    LAc = PI._a_centrada(nuevo["recta_A"], pp)
    LBc = PI._a_centrada(nuevo["recta_B"], pp)
    info = {"ok": True, "v1_inl": int(v1["n"]),
            "v1_ang_pies_deg": float(v1["ang_con_pies_deg"]),
            "res_A_px": resid["A"], "res_B_px": resid["B"],
            "res_A_antes_px": float(bo.get("res_A") or float("nan")),
            "res_B_antes_px": float(bo.get("res_B") or float("nan")),
            "V1_antes": np.cross(PI._a_centrada(bo["recta_A"], pp),
                                 PI._a_centrada(bo["recta_B"], pp)),
            "V1_nuevo": np.cross(LAc, LBc), "n_segm": len(S)}
    """SE DEVUELVE EL RESIDUO ANTES Y DESPUES. Imponer que las dos rectas pasen por un punto solo
    puede EMPEORAR el ajuste a los puntos del barrido: si empeora mucho, el V1 global no es el de
    esta pista y hay que quedarse con el de hoy. Es la cifra que decide, y por eso sale."""
    if verboso:
        print("   V1 antes %s  ->  ahora %s" % (
            np.round(info["V1_antes"][:2] / info["V1_antes"][2], 0),
            np.round(info["V1_nuevo"][:2] / info["V1_nuevo"][2], 0)))
    return nuevo, info


def _recta_por_punto_y_vp(pts, V):
    """QUE DEVUELVE: la recta que pasa por el punto de fuga `V` (homogeneo, en coordenadas de
    IMAGEN) y minimiza la distancia cuadratica a `pts`, o None.

    Con `V` fijo la recta tiene un solo grado de libertad: el angulo. Se resuelve cerrado
    parametrizando por un segundo punto `p = V + t * u` y buscando el `u` que mejor ajusta, que
    equivale a la direccion principal de los puntos vistos desde `V`.
    """
    P = np.asarray(pts, "float64")
    if abs(V[2]) < 1e-12:
        d = np.array([V[0], V[1]], "float64")
        nn = np.linalg.norm(d)
        if nn < 1e-12:
            return None
        d = d / nn
        c = P.mean(axis=0)
        nrm = np.array([-d[1], d[0]])
        return np.array([nrm[0], nrm[1], -float(nrm @ c)])
    v = np.array([V[0] / V[2], V[1] / V[2]])
    D = P - v[None, :]
    nn = np.linalg.norm(D, axis=1, keepdims=True)
    ok = (nn[:, 0] > 1e-9)
    if ok.sum() < 3:
        return None
    U = D[ok] / nn[ok]
    ang = np.arctan2(U[:, 1], U[:, 0])
    ang = np.mod(ang, np.pi)
    a = float(np.median(ang))
    """MEDIANA DEL ANGULO Y NO MINIMOS CUADRADOS: los puntos del barrido traen atipicos cuando el
    borde se pierde detras de un tirador, y un ajuste cuadratico los persigue. El angulo se toma
    modulo pi porque una recta no distingue sus dos sentidos."""
    d = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-d[1], d[0]])
    return np.array([nrm[0], nrm[1], -float(nrm @ v)])


def mejores_bordes(res, verboso=False):
    """QUE DEVUELVE: (bordes, info) quedandose con la via que MEJOR explique los puntos que el
    barrido ya encontro. Es la combinacion de las dos, no la sustitucion de una por otra.

    ## EL CRITERIO, y no tiene ningun umbral elegido a mano

    Los puntos del borde los encuentra el barrido local, y eso no se toca: es lo que sabe hallar una
    pista de 26 px de ancho entre cuatro pistas iguales. Lo que se pone en duda es la DIRECCION, que
    hoy sale de ajustar dos rectas cortas por separado y queda pesimamente determinada.

    Asi que se prueban las dos direcciones -- la local y la del haz global del pabellon -- contra
    **los mismos puntos**, y gana la que deje menos residuo. Medido sobre 14 clips:

        AbandonedAbsoluteGoldfinch    (falla hoy)   14,60 -> 2,29 px    global GANA
        AbleGregariousCollie          (falla hoy)    2,74 -> 0,64 px    global GANA
        AbsoluteAmusedGermanpinscher  (falla hoy)    1,92 -> 1,85 px    global gana justo
        AbandonedUnderstatedGalago    (OK hoy)       0,49 -> 0,96 px    local gana
        AbandonedBigheartedAmericanc. (falla hoy)    1,57 -> 4,87 px    local gana

    Cuando el haz global ES el eje de la pista, explica los puntos del barrido MEJOR que la recta
    que se ajusto a esos mismos puntos -- que es justo lo que tiene que pasar si la direccion es la
    verdadera y el ajuste local estaba capturando ruido. Cuando no lo es, empeora y se descarta.

    No hay que elegir ninguna constante: el criterio es una comparacion de residuos sobre el mismo
    dato.
    """
    bo = res.get("_bo")
    if not (bo and bo.get("ok")):
        return None, {"ok": False, "motivo": "sin bordes de partida"}
    nb, info = refina_con_v1_global(res, verboso=verboso)
    if nb is None:
        return bo, {"ok": True, "via": "barrido local", "motivo_global": info.get("motivo")}
    antes = np.nanmean([info["res_A_antes_px"], info["res_B_antes_px"]])
    ahora = np.nanmean([info["res_A_px"], info["res_B_px"]])
    if np.isfinite(antes) and np.isfinite(ahora) and ahora < antes:
        info.update({"via": "barrido local + direccion del haz global",
                     "res_antes_px": float(antes), "res_ahora_px": float(ahora),
                     "mejora_px": float(antes - ahora)})
        return nb, info
    return bo, {"ok": True, "via": "barrido local", "res_antes_px": float(antes),
                "res_global_px": float(ahora), "v1_inl": info.get("v1_inl")}


def homografia_global(res, verboso=False):
    """QUE HACE: construye la homografia con V1 y Vz sacados de los haces GLOBALES del pabellon,
    sin pasar por `pista.verticales` ni por el minimo de 3 marcas afinadas.

    ## POR QUE HACE FALTA PUENTEAR ESAS DOS PUERTAS

    Medido: con la direccion del haz global adoptada en 5 de 20 clips, solo **1** llegaba a producir
    homografia. Los otros 4 seguian cayendo por donde caen hoy, que no es V1:

        sin punto de fuga vertical: solo 3 verticales concurren        17 de 78
        sin punto de fuga vertical: Vz fuera de la columna principal   16 de 78
        menos de 3 marcas afinadas                                     20 de 78

    Las dos primeras son `verticales()`, que busca estructuras verticales **dentro del corredor** de
    los tiradores. Es el mismo estrechamiento que en los bordes: la sala esta llena de postes, patas
    de mesa y montantes de grada, y se descartan por estar lejos de los pies.

    La tercera, el minimo de 3 marcas, existe porque hoy las marcas eligen la focal. Aqui la focal la
    elige la CONSTANCIA DE LA ESTATURA, asi que ese minimo deja de hacer falta: las marcas pasan a
    servir solo para el origen, que es el papel que ya cumplen en la tira.

    ## LO QUE SIGUE SIN SER GRATIS

    La estatura deja de ser un control independiente, igual que en `focal_fisica`. Lo que queda para
    juzgar esto es el ancho de pista DEDUCIDO contra el rango FIE -- que no se impone -- y
    `strip_location`.

    QUE DEVUELVE: (hg, info) con la forma de `pista.homografia`, o (None, info).
    """
    bo, info_b = mejores_bordes(res, verboso=verboso)
    if bo is None or not bo.get("ok"):
        return None, {"ok": False, "motivo": "sin bordes"}
    pl, co = res.get("_pl"), res.get("_co")
    if pl is None or co is None:
        return None, {"ok": False, "motivo": "sin placa o sin corredor"}
    H, W = pl.shape[:2]
    pp = PI._pp(W, H)
    S = LG.segmentos(pl, mascara=res.get("_rot"))
    vps = LG.puntos_de_fuga(S)
    cen = np.array([W / 2.0, H / 2.0])
    v1 = LG.elige_v1(vps, co["e"], cen)
    if v1 is None:
        return None, {"ok": False, "motivo": "ningun haz compatible con la marcha"}
    LAc = PI._a_centrada(bo["recta_A"], pp)
    LBc = PI._a_centrada(bo["recta_B"], pp)
    V1 = np.cross(LAc, LBc)
    vz = None
    for v in vps:
        if v is v1:
            continue
        d = np.asarray(v["P"], "float64") - cen
        if abs(d[1]) > 2.0 * abs(d[0]):
            vz = v
            break
    """EL HAZ VERTICAL es el que se aleja MAS EN VERTICAL que en horizontal, con factor 2 para no
    confundirlo con uno oblicuo. Si no hay ninguno se cae a la vertical de imagen, que es lo que
    supone una camara sin balanceo, y sale declarado en `vz_supuesta` para que nadie lo lea como
    medido."""
    Vz = (np.array([vz["P"][0] - pp[0], vz["P"][1] - pp[1], 1.0]) if vz is not None
          else np.array([0.0, 1.0, 0.0]))
    Oc = np.cross(LAc, np.cross(V1, np.array([0.0, 0.0, 1.0])))
    Oc = Oc / Oc[2] if abs(Oc[2]) > 1e-12 else np.array([0.0, 0.0, 1.0])
    pares = FF._pares(res, nmax=60)
    if len(pares) < FF.MIN_APOYOS:
        return None, {"ok": False, "motivo": "menos de %d apoyos" % FF.MIN_APOYOS}

    mejor = None
    for f in np.exp(np.linspace(np.log(PI.F_REL_RANGO[0] * W),
                                np.log(PI.F_REL_RANGO[1] * W), 120)):
        w = np.array([1.0 / (f * f), 1.0 / (f * f), 1.0])
        V2 = np.cross(w * V1, w * Vz)
        nv = np.linalg.norm(V2)
        if nv < 1e-14:
            continue
        V2 = V2 / nv
        con = PI._construye_H(V1, V2, f, LBc, Oc, 1.0, pp)
        if not con.get("ok"):
            continue
        try:
            Hi = np.linalg.inv(con["H"])
        except np.linalg.LinAlgError:
            continue
        hgl = {"H": con["H"], "Hi": Hi, "vz_h": con["vz_h"]}
        v = [PI.altura_m(hgl, a, b) for a, b in pares]
        v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
        if len(v) < FF.MIN_APOYOS or (v > 0).mean() < FF.MIN_POSITIVAS:
            continue
        p10 = float(np.percentile(v, 10))
        p50 = float(np.median(v))
        p90 = float(np.percentile(v, 90))
        if p50 <= 1e-6:
            continue
        disp = (p90 - p10) / p50
        ancho = FF.OBJETIVO_H_M / p50
        if not (FF.RANGO_ANCHO_M[0] <= ancho <= FF.RANGO_ANCHO_M[1]):
            continue
        hc = con["hc_m"] * ancho
        if not (PI.ALTURA_CAMARA_M[0] <= hc <= PI.ALTURA_CAMARA_M[1]):
            continue
        if mejor is None or disp < mejor[0]:
            mejor = (disp, ancho, float(f), V2, con, hc, p50)
    if mejor is None:
        return None, {"ok": False, "motivo": "ninguna focal da estatura positiva con ancho FIE y "
                                             "camara a altura fisica"}
    disp, ancho, f, V2, con, hc, p50 = mejor
    s = FF.OBJETIVO_H_M / p50
    Hn = con["H"] @ np.diag([1.0 / s, 1.0 / s, 1.0])
    hg = {"ok": True, "H": Hn, "Hi": np.linalg.inv(Hn),
          "vz_h": np.asarray(con["vz_h"], "float64") / s,
          "V1": V1, "V2": V2, "Vz": Vz, "f": f, "K": con["K"], "mu": con["mu"],
          "hc_m": hc, "ancho_m": ancho, "pp": pp,
          "via": "haces globales", "fuente_f": "constancia de la estatura",
          "marca_usada": None, "n_marcas": 0,
          "ortog_deg": 0.0, "sen_bordes": float("nan"),
          "ang_fuga_marcas_deg": float("nan"), "n_marcas_fuga": 0,
          "incl_horizonte_deg": float("nan"), "ang_fuga_vert_deg": float("nan"),
          "n_vert": int(vz["n"]) if vz is not None else 0, "n_vert_total": len(S),
          "desv_columna_vz_px": float("nan")}
    info = {"ok": True, "dispersion": float(disp), "ancho_deducido_m": float(ancho),
            "hc_m": float(hc), "f_rel": float(f / W), "v1_inl": int(v1["n"]),
            "vz_inl": int(vz["n"]) if vz is not None else 0,
            "vz_supuesta": bool(vz is None),
            "estable": bool(disp <= FF.DISP_MAX),
            "ancho_en_rango_fie": True,
            "via_bordes": info_b.get("via", "?")}
    return hg, info
