"""ETAPA E, paso 20: EL ESTADO DEL CONO ANTES DE TOCARLO, y el careo de los dos ejes.

El usuario dijo tres cosas: *"le pones un florete pero no es el cono que dijimos, es recto, un palo.
Revisalo. Ademas es muy corto, no mide como deberia medir un florete. Y luego el cono tiene que ser
mas estrecho, que es demasiado gordo"*. Los tres son defectos distintos y **solo uno se arregla
dibujando**. Este paso mide los tres antes de cambiar nada, porque sin la linea base no se sabe si
lo que viene despues mejora.

LOS TRES CRITERIOS, DECLARADOS ANTES DE EJECUTAR:

  C-EJE  El eje se cambia de `hombro -> raiz` a `codo -> raiz` si y solo si baja la fraccion de
         CANDIDATOS DE CONTACTO que caen por debajo de las dos manos. La cifra a batir viene dada:
         **63,6 %**. Mejora declarada: **bajar a la mitad o menos, o sea <= 31,8 %**. Se mide ademas
         una segunda cosa que NO puede salir de la definicion del candidato: la fraccion de cuadros
         en que la punta del eje (`vertice + L * u`) cae por debajo del TOBILLO del propio tirador,
         que es «el cono acaba en el suelo» sin depender de donde este el rival.

  C-APER La apertura baja si y solo si baja el error MEDIDO contra la hoja de verdad. Se mide
         `|angulo de la hoja detectada - angulo del eje|` con los dos ejes sobre las MISMAS
         detecciones cacheadas, y con el margen declarado de `e14`: **5 grados**. Si los dos ejes se
         separan menos de 5 grados, EMPATAN y la apertura NO se toca. Y si el eje nuevo sale PEOR,
         la apertura tendria que subir, no bajar, y eso tambien se dice.

  C-LARGO El largo pintado se compara con lo que manda la escala. No hay criterio de mejora que
         inventar aqui: `L = HOJA_M * pxm` es aritmetica. Lo que si se mide es **de que depende
         `pxm`**, porque `escala_px_m` divide el alto de la caja por una estatura de 1,75 m, y un
         esgrimista EN GUARDIA no mide su estatura. Se da la razon `p95/p50` del alto de caja dentro
         del clip, que es cuanto se encoge la referencia, y de ahi cuanto se acorta `L`.

Y UNA ADVERTENCIA SOBRE LAS DOS METRICAS DEL EJE, que es la razon de que hagan falta las dos: el
angulo contra la hoja se mide **modulo 180 grados** (un segmento detectado no tiene sentido, solo
direccion), asi que un eje que apunte al SUELO en vez de al rival da exactamente el mismo numero que
su opuesto. El defecto de «el cono se va al suelo» es invisible para C-APER por construccion. Son
preguntas distintas y no se pueden mezclar.

TODO SALE DE LOS NPZ Y DE LOS JSON DE HOJA YA CACHEADOS. Cero GPU, cero modelos, cero decodificacion
de video: el PC del usuario esta saturado y este paso no necesita ni un fotograma.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
APERT = 23.1
APERT_CODO_P90 = 33.0
"""p90 de |hoja - eje del antebrazo| medido abajo. Se escribe aqui para poder correr la tercera
configuracion en la misma pasada; el valor se vuelve a imprimir medido y si no coincide, se dice."""
BASE_CANDIDATO = 63.6
META_CANDIDATO = 0.5 * BASE_CANDIDATO
MARGEN_EJE = 5.0
TOBILLOS = (13, 14)


def corpus():
    return set(x.strip() for x in
               (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
               if x.strip())


def clips_cono():
    """Los clips con `cono_poses_*.npz`, que son los que la etapa E ya midio, filtrados al corpus.

    SE MUESTREA DE LA LISTA DEL CORPUS Y NUNCA DEL ARBOL DE VIDEOS: en `D:` hay espada y sable de
    otros estudios y eso ya estropeo dos tiradas del proyecto.
    """
    c = corpus()
    return [f.name[len("cono_poses_"):-4] for f in sorted(PRUEBAS.glob("cono_poses_*.npz"))
            if not f.name.endswith("_512x384.npz") and f.name[len("cono_poses_"):-4] in c]


def carga(clip):
    """Lee el NPZ admitiendo los DOS convenios de nombre que hay en `pruebas/`.

    `cono_poses_*` trae `presente`/`marca` y `poses_*` trae `pres`/`mask`. Un lector que solo
    entienda uno de los dos se calla los clips del otro, y eso es un hueco silencioso.
    """
    f = PRUEBAS / ("cono_poses_%s.npz" % clip)
    if not f.is_file():
        f = PRUEBAS / ("poses_%s.npz" % clip)
    with np.load(f, allow_pickle=True) as z:
        d = {"kp": z["KP"] if "KP" in z.files else z["kp"],
             "sc": z["SC"] if "SC" in z.files else z["sc"],
             "B": z["B"], "W": int(z["W"]), "H": int(z["H"]), "T": int(z["T"]),
             "video": str(z["video"])}
        d["pres"] = z["pres"] if "pres" in z.files else z["presente"]
        d["mask"] = (z["mask"] if "mask" in z.files
                     else (z["marca"] if "marca" in z.files else np.ones(d["T"], bool)))
    if d["mask"].dtype != bool or d["mask"].shape != (d["T"],):
        d["mask"] = np.ones(d["T"], bool)
    return d


def geometria(d, cual):
    """Vertice, direccion, validez, escala y largo pintado, por lado, con el ancla pedida."""
    kp, sc, B, W, H = d["kp"], d["sc"], d["B"], d["W"], d["H"]
    g = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            return None
        vert, u, ok, largo = C3.eje_brazo_2d(kp, sc, lado, mano, cual)
        pk, pxm = CL.escala_px_m(B, lado, H)
        M = CL.MANOS[mano]
        g[lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "largo": largo,
                   "pxm": pxm, "pxm_k": pk, "L": CL.HOJA_M * pxm,
                   "alto": (B[:, lado, 3] - B[:, lado, 1]) * H,
                   "root": kp[:, lado, M["root"]].astype("float64"),
                   "hombro": kp[:, lado, M["hombro"]].astype("float64"),
                   "codo": kp[:, lado, M["codo"]].astype("float64"),
                   "tobillo": np.nanmax(kp[:, lado, list(TOBILLOS), 1], axis=1).astype("float64")}
    return g


def mide_clip(clip):
    d = carga(clip)
    dos = np.array([bool(d["mask"][k] and d["pres"][k].sum() == 2) for k in range(d["T"])])
    fila = {"clip": clip, "W": d["W"], "H": d["H"], "T": d["T"], "n_dos": int(dos.sum()),
            "ejes": {}}
    for cual in ("hombro", "codo"):
        g = geometria(d, cual)
        if g is None:
            return None
        e = {"lados": {}}
        for lado in (0, 1):
            gl = g[lado]
            ok = gl["ok"] & dos
            u = gl["u"]
            punta = gl["vert"] + gl["L"] * u
            """`u[:, 1] > 0` es apuntar hacia ABAJO: en imagen la fila crece hacia el suelo."""
            abajo = ok & (u[:, 1] > 0)
            elev = -np.degrees(np.arcsin(np.clip(u[:, 1], -1, 1)))
            en_suelo = ok & np.isfinite(gl["tobillo"]) & (punta[:, 1] > gl["tobillo"])
            brazo = np.linalg.norm(gl["root"] - gl["hombro"], axis=1)
            ante = np.linalg.norm(gl["root"] - gl["codo"], axis=1)
            alto_ok = gl["alto"][np.isfinite(gl["alto"])]
            e["lados"][lado] = {
                "mano": gl["mano"], "n_ok": int(ok.sum()),
                "pxm": gl["pxm"], "L_px": gl["L"],
                "alto_p50": float(np.nanmedian(alto_ok)) if alto_ok.size else float("nan"),
                "alto_p95": float(np.percentile(alto_ok, 95)) if alto_ok.size else float("nan"),
                "L_sobre_brazo": float(gl["L"] / np.nanmedian(brazo[ok])) if ok.any() else
                float("nan"),
                "L_sobre_antebrazo": float(gl["L"] / np.nanmedian(ante[ok])) if ok.any() else
                float("nan"),
                "frac_abajo": float(abajo.sum() / max(ok.sum(), 1)),
                "elev_mediana": float(np.nanmedian(elev[ok])) if ok.any() else float("nan"),
                "frac_punta_bajo_tobillo": float(en_suelo.sum() / max(ok.sum(), 1))}
        """EL CANDIDATO DE CONTACTO: el punto de peso maximo del producto de los dos conos, que es
        exactamente el que `e15`/`e19` marcan con la cruz. Se cuenta cuantos caen por debajo de las
        DOS manos, que es el defecto que el usuario ve como «el cono se mete en el suelo».

        Y de la misma pasada sale el PARPADEO (el booleano de solape cambiando de un cuadro al
        siguiente), que es la unica medida de calidad del cono que no necesita verdad-terreno:
        un contacto de hojas real dura varios cuadros. Referencia previa de `e18`: 3,5-4,9 %."""
        for cfg, apert, alcance in (("base", APERT, CL.HOJA_M), ("largo", APERT, C3.ALCANCE_M),
                                    ("nuevo", APERT_CODO_P90, C3.ALCANCE_M)):
            n_c = n_bajo = n_sol = n_val = 0
            cru, val = np.zeros(d["T"], bool), np.zeros(d["T"], bool)
            for k in range(d["T"]):
                if not (dos[k] and g[0]["ok"][k] and g[1]["ok"][k]):
                    continue
                n_val += 1
                val[k] = True
                L0 = alcance * g[0]["pxm"]
                L1 = alcance * g[1]["pxm"]
                r = CL.cruce(g[0]["vert"][k], g[0]["u"][k], L0, apert,
                             g[1]["vert"][k], g[1]["u"][k], L1, apert)
                if r["n_inter"] > 0:
                    n_sol += 1
                    cru[k] = True
                if not np.isfinite(r["py"]):
                    continue
                ym = max(g[0]["root"][k, 1], g[1]["root"][k, 1])
                if not np.isfinite(ym):
                    continue
                n_c += 1
                if r["py"] > ym:
                    n_bajo += 1
            p = val[:-1] & val[1:]
            e["candidatos" if cfg == "base" else "candidatos_%s" % cfg] = {
                "apert": apert, "alcance_m": alcance,
                "n_val": n_val, "n_solapan": n_sol, "n_cand": n_c, "n_bajo": n_bajo,
                "frac_bajo": n_bajo / max(n_c, 1), "frac_solape": n_sol / max(n_val, 1),
                "parpadeo": float((cru[1:][p] != cru[:-1][p]).mean()) if p.any() else float("nan"),
                "n_pares": int(p.sum())}
        fila["ejes"][cual] = e
    return fila


def careo_hoja():
    """|angulo hoja - eje| con los DOS ejes, sobre TODAS las detecciones de hoja ya cacheadas.

    POR QUE ESTO NO ES REPETIR `e14`: `e14` uso solo las 6 rectas de la ventana de accion de tres
    clips. Aqui entran ademas las 26 + 1 rectas de `cono_hoja_*.json`, que `e5` detecto repartidas
    por el clip y no en la accion. Es el mismo criterio con **mas n**, que es lo unico que podia
    romper el empate de 1,2-2,7 grados que `e14` no pudo resolver.

    Y VA CON TRES SELECCIONES, no una, porque el filtro «delante de la mano» SE CALCULA CON UN EJE y
    por tanto puede favorecer a ese eje: exige que el extremo mas lejano del segmento tenga
    proyeccion positiva sobre el eje, o sea que caiga a menos de 90 grados de el. Si el ganador
    cambia al cambiar el filtro, entonces lo que decide es el filtro y no los datos, y se dice.
    """
    fuentes = []
    for f in sorted(PRUEBAS.glob("cono_hoja3d_*_accion.json")):
        clip = f.name[len("cono_hoja3d_"):-len("_accion.json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            ln = r.get("linea")
            if ln and not r.get("contaminado"):
                fuentes.append((clip, "e13_accion", r["cuadro"], r["lado"], ln,
                                ln.get("dif_codo"), ln.get("dif_hombro")))
    for f in sorted(PRUEBAS.glob("cono_hoja_*.json")):
        clip = f.name[len("cono_hoja_"):-len(".json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            ln = r["det"].get("p99.5", {}).get("linea") if not r.get("contaminado") else None
            if ln:
                fuentes.append((clip, "e5_clip", r["cuadro"], r["lado"], ln,
                                ln.get("dif_eje"), None))
    """TODO SE RECALCULA DEL NPZ, incluidos los angulos que el JSON ya trae. Asi el careo no depende
    de con que eje se escribio cada fichero, que es distinto en `e5` y en `e13`. Los valores
    guardados se usan solo como CONTROL: si el recalculo no los reproduce, el careo no vale."""
    gc_cache, reg, ctrl = {}, [], []
    for clip, fuente, k, lado, ln, dif_guardado_codo, dif_guardado_hombro in fuentes:
        if clip not in gc_cache:
            try:
                d = carga(clip)
            except FileNotFoundError:
                gc_cache[clip] = None
                continue
            gc_cache[clip] = {c: geometria(d, c) for c in ("codo", "hombro")}
        gg = gc_cache[clip]
        if gg is None or gg["codo"] is None or gg["hombro"] is None:
            continue
        vert = gg["codo"][lado]["vert"][k]
        uc, uh = gg["codo"][lado]["u"][k], gg["hombro"][lado]["u"][k]
        if not (np.isfinite(uc).all() and np.isfinite(uh).all() and np.isfinite(vert).all()):
            continue
        p1 = np.asarray(ln["p1"], "float64")
        p2 = np.asarray(ln["p2"], "float64")
        a_c = float(np.degrees(np.arctan2(uc[1], uc[0])))
        a_h = float(np.degrees(np.arctan2(uh[1], uh[0])))
        d_c = float(abs((ln["ang"] - a_c + 90.0) % 180.0 - 90.0))
        d_h = float(abs((ln["ang"] - a_h + 90.0) % 180.0 - 90.0))
        if dif_guardado_codo is not None:
            ctrl.append(abs(d_c - dif_guardado_codo))
        if dif_guardado_hombro is not None:
            ctrl.append(abs(d_h - dif_guardado_hombro))
        reg.append({"clip": clip, "fuente": fuente, "cuadro": k, "lado": lado,
                    "largo": ln["largo"], "ang": ln["ang"], "dif_codo": d_c, "dif_hombro": d_h,
                    "delante_codo": bool(max(float((p1 - vert) @ uc),
                                             float((p2 - vert) @ uc)) > 0),
                    "delante_hombro": bool(max(float((p1 - vert) @ uh),
                                               float((p2 - vert) @ uh)) > 0)})
    print("    control de reproducibilidad del angulo recalculado contra el guardado: "
          "maximo %.4f grados sobre %d comparaciones (limite declarado 0,5)" % (
              max(ctrl) if ctrl else float("nan"), len(ctrl)))
    return reg


def tabla_careo(reg, sel, nom):
    d = [r for r in reg if sel(r)]
    if len(d) < 3:
        print("    %-28s n=%d: no llega para decidir" % (nom, len(d)))
        return None
    out = {"n": len(d)}
    for cual in ("codo", "hombro"):
        v = np.array([r["dif_%s" % cual] for r in d])
        out[cual] = {"n": int(v.size), "mediana": float(np.median(v)),
                     "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
                     "f20": float((v <= 20).mean())}
    dm = out["hombro"]["mediana"] - out["codo"]["mediana"]
    out["gana"] = ("EMPATE" if abs(dm) < MARGEN_EJE else ("codo" if dm > 0 else "hombro"))
    out["separacion"] = float(abs(dm))
    print("    %-28s n=%3d | codo med %5.1f p90 %5.1f | hombro med %5.1f p90 %5.1f | "
          "separacion %4.1f -> %s" % (
              nom, len(d), out["codo"]["mediana"], out["codo"]["p90"],
              out["hombro"]["mediana"], out["hombro"]["p90"], out["separacion"], out["gana"]))
    return out


def main():
    print("=" * 108)
    print("ESTADO DEL CONO ANTES DE TOCARLO  ·  criterios declarados arriba  ·  sin GPU")
    print("=" * 108)
    print()
    print("  QUE USA CADA SCRIPT DE DIBUJO HOY (leido del codigo, no medido):")
    print("    e7_pinta_cono.py        eje `codo -> raiz` (cono_lib.eje_antebrazo)  ·  "
          "L = 0,90 m * pxm  ·  DOS aperturas: p95 del temblor y p90 de la hoja (25,4)")
    print("    e11_figuras_cono3d.py   eje `hombro -> raiz` (cono3d_lib.EJE_POR_DEFECTO)  ·  "
          "L = 0,90 m * pxm  ·  apertura de construye_cono (23,1 / 27,3 de reserva)")
    print("    e19_figuras_cono_cono   eje `hombro -> raiz`  ·  L = 0,90 m * pxm  ·  "
          "apertura CONSTANTE 23,1")
    print("    -> NINGUNO pinta un largo fijo en pixeles: los tres usan la escala. El largo corto")
    print("       no viene de una constante, viene de `pxm`. Se mide abajo.")
    print()
    res = {}
    print("  %-32s %5s %9s | %7s %7s %7s | %7s %7s" % (
        "clip", "res", "cuadros", "pxm", "L px", "L/brazo", "alto95", "/alto50"))
    for clip in clips_cono():
        f = mide_clip(clip)
        if f is None:
            print("  %-32s sin mano armada decidible" % clip[:32])
            continue
        res[clip] = f
        e = f["ejes"]["hombro"]["lados"]
        print("  %-32s %5s %9d | %7.1f %7.1f %7.2f | %7.1f %7.2f" % (
            clip[:32], "1080" if f["H"] >= 1000 else "720", f["n_dos"],
            0.5 * (e[0]["pxm"] + e[1]["pxm"]), 0.5 * (e[0]["L_px"] + e[1]["L_px"]),
            0.5 * (e[0]["L_sobre_brazo"] + e[1]["L_sobre_brazo"]),
            0.5 * (e[0]["alto_p95"] + e[1]["alto_p95"]),
            0.5 * (e[0]["alto_p95"] / e[0]["alto_p50"] + e[1]["alto_p95"] / e[1]["alto_p50"])))

    print()
    print("  C-EJE · EL CONO SE VA AL SUELO. Dos medidas independientes, por eje")
    print("  %-32s | %-27s | %-27s" % ("clip", "HOMBRO -> RAIZ (actual)", "CODO -> RAIZ (propuesto)"))
    print("  %-32s | %6s %7s %7s | %6s %7s %7s" % (
        "", "cand.", "punta", "elev", "cand.", "punta", "elev"))
    ag = {c: {"n_cand": 0, "n_bajo": 0, "suelo": [], "elev": [], "sol": [], "parp": [],
              "sol_largo": [], "parp_largo": [], "sol_nuevo": [], "parp_nuevo": []}
          for c in ("hombro", "codo")}
    for clip, f in res.items():
        lin = []
        for cual in ("hombro", "codo"):
            e = f["ejes"][cual]
            a = ag[cual]
            a["n_cand"] += e["candidatos"]["n_cand"]
            a["n_bajo"] += e["candidatos"]["n_bajo"]
            a["sol"].append(e["candidatos"]["frac_solape"])
            a["parp"].append(e["candidatos"]["parpadeo"])
            a["sol_largo"].append(e["candidatos_largo"]["frac_solape"])
            a["parp_largo"].append(e["candidatos_largo"]["parpadeo"])
            a["sol_nuevo"].append(e["candidatos_nuevo"]["frac_solape"])
            a["parp_nuevo"].append(e["candidatos_nuevo"]["parpadeo"])
            for lado in (0, 1):
                a["suelo"].append(e["lados"][lado]["frac_punta_bajo_tobillo"])
                a["elev"].append(e["lados"][lado]["elev_mediana"])
            lin.append((100 * e["candidatos"]["frac_bajo"],
                        100 * 0.5 * (e["lados"][0]["frac_punta_bajo_tobillo"]
                                     + e["lados"][1]["frac_punta_bajo_tobillo"]),
                        0.5 * (e["lados"][0]["elev_mediana"] + e["lados"][1]["elev_mediana"])))
        print("  %-32s | %5.1f %% %6.1f %% %6.1f | %5.1f %% %6.1f %% %6.1f" % (
            clip[:32], lin[0][0], lin[0][1], lin[0][2], lin[1][0], lin[1][1], lin[1][2]))
    print()
    tot = {}
    for cual in ("hombro", "codo"):
        a = ag[cual]
        tot[cual] = {"frac_bajo": a["n_bajo"] / max(a["n_cand"], 1), "n_cand": a["n_cand"],
                     "punta_bajo_tobillo": float(np.mean(a["suelo"])),
                     "elev_mediana": float(np.median(a["elev"])),
                     "frac_solape": float(np.median(a["sol"])),
                     "parpadeo": float(np.nanmedian(a["parp"])),
                     "frac_solape_L110": float(np.median(a["sol_largo"])),
                     "parpadeo_L110": float(np.nanmedian(a["parp_largo"])),
                     "frac_solape_nuevo": float(np.median(a["sol_nuevo"])),
                     "parpadeo_nuevo": float(np.nanmedian(a["parp_nuevo"]))}
        print("  TOTAL %-8s candidatos por debajo de las dos manos **%.1f %%** (%d de %d)  ·  "
              "punta bajo el tobillo %.1f %%  ·  elevacion mediana %.1f gr  ·  solape %.1f %%" % (
                  cual, 100 * tot[cual]["frac_bajo"], a["n_bajo"], a["n_cand"],
                  100 * tot[cual]["punta_bajo_tobillo"], tot[cual]["elev_mediana"],
                  100 * tot[cual]["frac_solape"]))
    print()
    print("  VEREDICTO C-EJE: linea base dada %.1f %%  ·  meta declarada <= %.1f %%  ·  "
          "medido con el codo %.1f %%  ->  %s" % (
              BASE_CANDIDATO, META_CANDIDATO, 100 * tot["codo"]["frac_bajo"],
              "MEJORA" if 100 * tot["codo"]["frac_bajo"] <= META_CANDIDATO else "NO LLEGA"))
    print()
    print("  EFECTO SECUNDARIO SIN VERDAD-TERRENO: el PARPADEO del booleano de solape, mediana de "
          "clips (referencia de `e18` con el codo: 3,5-4,9 %%)")
    print("  %-16s %11s %11s | %11s %11s | %11s %11s" % (
        "eje", "sol 23/0,90", "parp", "sol 23/1,10", "parp", "sol 33/1,10", "parp"))
    for cual in ("hombro", "codo"):
        t = tot[cual]
        print("  %-16s %10.1f %% %10.1f %% | %10.1f %% %10.1f %% | %10.1f %% %10.1f %%" % (
            cual, 100 * t["frac_solape"], 100 * t["parpadeo"],
            100 * t["frac_solape_L110"], 100 * t["parpadeo_L110"],
            100 * t["frac_solape_nuevo"], 100 * t["parpadeo_nuevo"]))
    print("  CRITERIO 2 de `e18` (el solape sigue informando si esta en 5-80 %%): la configuracion "
          "nueva (codo, %.1f gr, %.2f m) da %.1f %%  ->  %s" % (
              APERT_CODO_P90, C3.ALCANCE_M, 100 * tot["codo"]["frac_solape_nuevo"],
              "DENTRO" if 0.05 <= tot["codo"]["frac_solape_nuevo"] <= 0.80 else "SATURADO"))

    print()
    print("  C-APER · |ANGULO DE LA HOJA DETECTADA - EJE|, con las detecciones ya cacheadas")
    reg = careo_hoja()
    print("    detecciones reunidas: %d  (%d de la ventana de accion de `e13`, %d repartidas por "
          "el clip de `e5`)" % (
              len(reg), sum(1 for r in reg if r["fuente"] == "e13_accion"),
              sum(1 for r in reg if r["fuente"] == "e5_clip")))
    car = {"todas": tabla_careo(reg, lambda r: True, "todas, sin filtro"),
           "delante_codo": tabla_careo(reg, lambda r: r["delante_codo"],
                                       "delante segun el CODO"),
           "delante_hombro": tabla_careo(reg, lambda r: r["delante_hombro"],
                                         "delante segun el HOMBRO")}
    print()
    print("    Y DESGLOSADO, porque un resultado que descansa en una fuente o en un clip no vale:")
    for fu in sorted(set(r["fuente"] for r in reg)):
        car["fuente_%s" % fu] = tabla_careo(
            reg, lambda r, fu=fu: r["delante_codo"] and r["fuente"] == fu, "solo %s" % fu)
    for cl in sorted(set(r["clip"] for r in reg)):
        car["clip_%s" % cl] = tabla_careo(
            reg, lambda r, cl=cl: r["delante_codo"] and r["clip"] == cl, "solo %s" % cl[:20])
    print()
    d31 = np.array([r["dif_codo"] for r in reg if r["delante_codo"]])
    print("    DISTRIBUCION de |hoja - codo| con el filtro de delante (n=%d): %s" % (
        d31.size, "  ".join("%.0f" % v for v in np.sort(d31))))
    print("    percentiles: p50 %.1f  p75 %.1f  p90 %.1f  p95 %.1f" % tuple(
        np.percentile(d31, [50, 75, 90, 95])))
    print()
    print("  LA TABLA QUE DECIDE SI SE PUEDE ESTRECHAR: cuantas de las hojas MEDIDAS caben dentro")
    print("  del cono. Estrechar sin bajar el error no es mejorar el dibujo, es perder amenazas.")
    print("  %-34s %10s %10s %14s" % ("cono", "contiene", "pierde", "que es"))
    cont = {}
    for nom, cual, ap, que in (
            ("hombro 23,1 gr (lo que se pinta hoy)", "hombro", APERT, "mediana de n=6 de `e14`"),
            ("hombro 66,3 gr", "hombro", 66.3, "p90 honrado de ese eje"),
            ("codo 14,4 gr", "codo", 14.4, "p50 medido: el NUCLEO"),
            ("codo 23,1 gr", "codo", APERT, "la de hoy, con el eje nuevo"),
            ("codo 33,0 gr", "codo", APERT_CODO_P90, "p90 medido: la ENVOLVENTE")):
        v = np.array([r["dif_%s" % cual] for r in reg if r["delante_codo"]])
        f = float((v <= ap).mean())
        cont[nom] = {"eje": cual, "apert": ap, "contiene": f, "n": int(v.size)}
        print("  %-34s %9.1f %% %9.1f %% %14s" % (nom, 100 * f, 100 * (1 - f), que))
    print()
    votos = [v["gana"] for v in (car["todas"], car["delante_codo"], car["delante_hombro"]) if v]
    print("    VEREDICTO C-APER con margen declarado de %.0f grados: %s" % (
        MARGEN_EJE, " / ".join(votos)))
    print("    -> %s" % (
        "los tres filtros coinciden en %s" % votos[0] if len(set(votos)) == 1 else
        "el ganador CAMBIA con el filtro: lo decide el filtro, no los datos. EMPATE operativo"))

    (PRUEBAS / "cono_e20_estado.json").write_text(json.dumps(
        {"apert_usada": APERT, "base_candidato": BASE_CANDIDATO, "meta": META_CANDIDATO,
         "margen_eje": MARGEN_EJE, "total": tot, "careo_hoja": car, "contencion": cont,
         "apert_codo_p90": APERT_CODO_P90, "alcance_m": C3.ALCANCE_M,
         "detecciones_hoja": reg, "n_detecciones_hoja": len(reg), "clips": res},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("guardado %s" % (PRUEBAS / "cono_e20_estado.json"))


if __name__ == "__main__":
    main()
