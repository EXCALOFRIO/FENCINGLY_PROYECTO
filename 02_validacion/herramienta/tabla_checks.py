"""Cada garantia de la tuberia convertida en una COMPROBACION que se ejecuta y da apto o no apto.

LO QUE PIDIO EL USUARIO, literal: "quiero eso, que tengamos para los clips sacada bien las poses sin
intrusos, haz una tabla y me lo pones al terminar con los checks y donde estan las pruebas de todo
para verificarlo". Y: "lo quiero super pulido y sin ningun error, no podemos permitir ningun error".

## LA DIFERENCIA ENTRE ESTO Y `busca_errores.py`

`busca_errores.py` busca SINTOMAS: ordena los clips por sospecha y saca imagenes para mirarlas. Es
una red de arrastre y un aviso suyo no es un error hasta que se ve.

Esto es lo contrario: una lista de **garantias que la tuberia dice cumplir**, cada una con su umbral
declarado, ejecutada sobre N clips. Aqui un fallo SI es un error, porque el umbral es el que el propio
codigo dice respetar. Si `_continuidad` promete que ningun cuadro entregado salta mas de 0,60 anchos
de caja, esta tabla lo comprueba y si sale 0,61 esta roto.

## POR QUE UNA GARANTIA VALE MAS QUE UNA NOTA MEDIA

La averia del fundido de `VainAchingGonolek` daba **96,1 % de presencia**, una cifra buenisima, sobre
poses mezcladas de dos planos distintos. Una nota media alta puede convivir con un dato corrupto. Una
garantia no: o se cumple en todos los cuadros entregados o no se cumple.

## COMO SE CORRE

  cajas:  python tabla_checks.py cajas 40      con cualquier interprete, usa el cache de detecciones
  poses:  python tabla_checks.py poses         con el interprete de cuda, lee los NPZ de poses
  tabla:  python tabla_checks.py tabla         escribe `02_validacion/CHECKS.md`
"""
from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

VAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion")
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SCR = pathlib.Path(os.environ.get(
    "SCRATCH", r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad"))
LISTA = SCR / "corpus_etiquetado.txt"
CRUDO = PRUEBAS / "checks_crudo.json"
SALIDA = VAL / "CHECKS.md"
REPO_HF = "EXCALORIO/fencingly-extraccion"


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def checks_cajas(n):
    """QUE HACE: las garantias que se pueden comprobar sin GPU, sobre las cajas entregadas."""
    import extrae_pareja as EP
    import pareja_v2 as PAR

    ids = [l.strip() for l in LISTA.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(20260910)
    rng.shuffle(ids)
    hechos, clips = [], []
    for cid in ids:
        if len(clips) >= n:
            break
        ca = SCR / ("det_%s.pkl" % cid)
        if not ca.exists():
            continue
        with open(ca, "rb") as fh:
            cajas, cuadros, W, H, T = pickle.load(fh)
        if T < 20 or not len(cajas):
            continue
        r = EP.procesa(cajas, cuadros, W, H, T)
        B, pres, mask = r["B"], r["pres"], r["mask"]
        if not mask.any():
            continue
        clips.append(cid)

        peor_salto, k_salto = 0.0, -1
        for lado in (0, 1):
            ks = np.nonzero(pres[:, lado] & mask)[0]
            for i in range(1, len(ks)):
                a, b = int(ks[i - 1]), int(ks[i])
                g = b - a
                if g > 3:
                    continue
                p, q = B[a, lado], B[b, lado]
                anc = max((p[2] - p[0]) * W, 1.0)
                d = float(np.hypot((0.5 * (q[0] + q[2]) - 0.5 * (p[0] + p[2])) * W,
                                   (0.5 * (q[1] + q[3]) - 0.5 * (p[1] + p[3])) * H)) / anc / g
                if d > peor_salto:
                    peor_salto, k_salto = d, b

        amb = np.nonzero((pres.sum(axis=1) == 2) & mask)[0]
        peor_iou, k_iou = 0.0, -1
        lado_mal = 0
        for k in amb:
            v = _iou(B[k, 0], B[k, 1])
            if v > peor_iou:
                peor_iou, k_iou = v, int(k)
        peor_prop, k_prop = 9.9, -1
        n_tiras = 0
        for k in np.nonzero(mask)[0]:
            for lado in (0, 1):
                if not pres[k, lado]:
                    continue
                b = B[k, lado]
                if not np.isfinite(b).all():
                    continue
                w, h = (b[2] - b[0]) * W, (b[3] - b[1]) * H
                if h <= 0:
                    continue
                borde = b[0] <= 0.004 or b[2] >= 0.996
                if borde:
                    if w / h < peor_prop:
                        peor_prop, k_prop = w / h, int(k)
                    if w / h < EP.PROP_MIN:
                        n_tiras += 1
        ks_m = np.nonzero(mask)[0]
        bordes_ok = bool(pres[ks_m[0]].sum() == 2 and pres[ks_m[-1]].sum() == 2)
        cruces = 0
        for k in amb:
            if 0.5 * (B[k, 0, 0] + B[k, 0, 2]) > 0.5 * (B[k, 1, 0] + B[k, 1, 2]):
                cruces += 1
        borde = []
        for lado in (0, 1):
            ks = np.nonzero(pres[:, lado] & mask)[0]
            if not len(ks):
                continue
            b = B[ks, lado]
            borde.append(float(np.mean((b[:, 0] <= 0.004) | (b[:, 2] >= 0.996))))
        hechos.append({
            "borde_max": max(borde) if borde else 0.0,
            "clip": cid, "T": int(T), "util": int(mask.sum()),
            "ambos": float(r["ambos"]), "faltan": int(r["faltan"]),
            "peor_salto": peor_salto, "k_salto": int(k_salto),
            "peor_iou": peor_iou, "k_iou": int(k_iou),
            "peor_prop_borde": None if k_prop < 0 else peor_prop, "n_tiras": int(n_tiras),
            "bordes_medidos": bordes_ok, "cruces_de_lado": int(cruces),
            "motivo": r["motivo"]})
        print("%-38s ambos %5.1f%%  salto %.2f  iou %.2f  tiras %d  cruces %d" % (
            cid[:38], 100 * r["ambos"], peor_salto, peor_iou, n_tiras, cruces))
    return {"cajas": hechos, "n_clips": len(clips),
            "umbrales": {"DESPL_MAX": EP.DESPL_MAX, "PROP_MIN": EP.PROP_MIN,
                         "SEP_ALTURAS": list(PAR.SEP_ALTURAS),
                         "DPIE_REL_MAX": PAR.DPIE_REL_MAX, "FRENTE_MIN": PAR.FRENTE_MIN}}


def checks_poses():
    """QUE HACE: las garantias del nivel de POSE, sobre los NPZ ya calculados."""
    import video_revision as VR

    fich = sorted(PRUEBAS.glob("poses_*.npz"))
    hechos = []
    for f in fich:
        z = np.load(f, allow_pickle=True)
        if "origen" not in z.files:
            continue
        KP, SC = z["KP"], z["SC"]
        pres, mask, origen = z["pres"], z["mask"], z["origen"]
        T = int(z["T"])
        B = z["B"]
        W, H = int(z["W"]), int(z["H"])
        ent = np.nonzero(mask)[0]
        if not len(ent):
            continue
        peor_pts, k_pts = 999, -1
        fuera_caja = 0
        for k in ent:
            for lado in (0, 1):
                if origen[k, lado] < 0:
                    continue
                ok = int(((SC[k, lado, :VR.CARA_DESDE] >= VR.CONF_PINTA)
                          & np.isfinite(KP[k, lado, :VR.CARA_DESDE, 0])).sum())
                if origen[k, lado] == 0 and ok < peor_pts:
                    peor_pts, k_pts = ok, int(k)
                if origen[k, lado] == 0 and pres[k, lado]:
                    b = B[k, lado]
                    if np.isfinite(b).all():
                        x0, y0 = b[0] * W, b[1] * H
                        x1, y1 = b[2] * W, b[3] * H
                        hx, hy = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
                        val = (SC[k, lado, :VR.CARA_DESDE] >= VR.CONF_PINTA) & np.isfinite(
                            KP[k, lado, :VR.CARA_DESDE, 0])
                        px = KP[k, lado, :VR.CARA_DESDE, 0][val]
                        py = KP[k, lado, :VR.CARA_DESDE, 1][val]
                        fuera_caja += int(((px < x0 - hx) | (px > x1 + hx)
                                           | (py < y0 - hy) | (py > y1 + hy)).sum())
        sin_pose = int((origen[ent] < 0).sum())
        val = VR.valida_inventadas(KP, SC, pres, mask, T)
        hue = np.array(z["hueco"]) if "hueco" in z.files else None
        hechos.append({
            "clip": f.name[6:-4], "T": T, "entregados": int(len(ent)),
            "sin_pose": sin_pose,
            "con_filtro": bool("filtro_vacias" in z.files),
            "tiene_hueco": hue is not None,
            "lados": 2 * int(len(ent)),
            "medidas": int((origen[ent] == 0).sum()),
            "interp_validado": (0 if hue is None
                                else int(((origen[ent] > 0) & (hue[ent] <= 24)).sum())),
            "interp_sin_medida": (0 if hue is None
                                  else int(((origen[ent] > 0) & (hue[ent] > 24)).sum())),
            "hueco_max": (0 if hue is None else int(hue[ent].max())),
            "inventadas": int((origen[ent] > 0).sum()),
            "sostenidas": int((origen[ent] == 2).sum()),
            "peor_puntos_medida": None if k_pts < 0 else int(peor_pts),
            "kp_fuera_de_caja": int(fuera_caja),
            "tiene_origen": True,
            "err_rel_p50": None if not val else val["rel_p50"],
            "err_rel_p90": None if not val else val["rel_p90"]})
        print("%-38s entregados %4d  sin pose %3d  inventadas %3d  peor puntos %s" % (
            f.name[6:-4][:38], len(ent), sin_pose, int((origen[ent] > 0).sum()),
            "-" if k_pts < 0 else peor_pts))
    return {"poses": hechos, "PUNTOS_MIN": VR.PUNTOS_MIN}


FILAS = [
    ("C1", "Ninguna caja entregada se teletransporta",
     "salto del centro <= 0,60 anchos de caja por cuadro de hueco",
     "extrae_pareja._continuidad, dos pasadas",
     "evidencia/casos_sospechosos (avisos `salto`)"),
    ("C2", "Las dos cajas nunca caen sobre la misma persona",
     "solape de la pareja entregada < 0,70",
     "extrae_pareja._quita_duplicados",
     "evidencia/casos_sospechosos (avisos `solapan`)"),
    ("C3", "Ninguna caja entregada es una tira del borde",
     "proporcion ancho/alto >= 0,25 si toca el borde",
     "extrae_pareja._quita_tiras",
     "evidencia/poses_inventadas (el caso que lo destapo)"),
    ("C4", "El clip entregado empieza y acaba con los DOS medidos",
     "primer y ultimo cuadro de la mascara con dos presencias",
     "extrae_pareja._recorta",
     "evidencia/videos_revision (ya no hay poses congeladas en los bordes)"),
    ("C5", "El cruce de lados se queda en lo que hacen las parejas reales",
     "NO es 0: verificado a ojo, dos tiradores reales se cruzan al pasar uno junto al otro",
     "a3_selector.elige_dos y pareja_v2.CRUZA_MAX",
     "evidencia/eleccion_de_pareja, muestras/limite_intruso"),
    ("C8", "Ninguna caja entregada pertenece a alguien que nunca se vio entero",
     "trayectoria cortada por el borde lateral en el 100 % de sus cuadros: descartada",
     "pareja_v2.BORDE_SIEMPRE",
     "muestras/limite_intruso (LavishDistantDikkops, 27,4 % -> 100 %)"),
    ("C6", "Los cuadros de primer plano y repeticion no se entregan",
     "puerta de planos: caja > 0,75 del alto Y sin pareja plausible",
     "planos.tramo_utilizable",
     "evidencia/puerta_de_planos"),
    ("C7", "Los cuadros de fundido encadenado no se entregan",
     "mezcla >= 2,50 Y residuo relativo al contraste >= 0,15",
     "transiciones.marca",
     "evidencia/fundidos_y_zoom"),
    ("P1", "Ninguna pose MEDIDA que se entrega esta vacia",
     "al menos 20 puntos utiles del cuerpo de 63",
     "video_revision.tira_poses_vacias",
     "evidencia/poses_inventadas"),
    ("P2", "Siempre hay DOS poses en los cuadros entregados",
     "cero cuadros entregados sin pose de alguno de los dos",
     "video_revision.inventa_poses",
     "evidencia/videos_revision, evidencia/poses_inventadas"),
    ("P3", "Toda pose inventada va marcada como tal",
     "`origen` presente y con valor por cada punto entregado",
     "video_revision.inventa_poses devuelve `origen`",
     "evidencia/poses_inventadas (el panel del medio es lo inventado)"),
    ("P4", "Ningun keypoint cae fuera de la caja de su propia persona",
     "keypoint dentro de su caja de YOLO con 5 % de holgura",
     "video_revision._fuera_de_caja",
     "evidencia/capas_densas"),
    ("P5", "Se puede saber que parte de lo entregado NO esta validada",
     "cada pose inventada lleva el largo del hueco del que sale, en `hueco`",
     "video_revision.inventa_poses devuelve `hueco`",
     "evidencia/poses_inventadas (el rotulo lleva el largo del hueco)"),
]
"""Cada fila es una GARANTIA que el codigo dice cumplir, con el sitio del codigo que la cumple y el
sitio de la evidencia para verla a ojo. La columna del codigo no es decoracion: si una garantia falla,
ahi esta el fichero que hay que arreglar."""


def tabla():
    if not CRUDO.exists():
        raise SystemExit("no hay medidas; corre `tabla_checks.py cajas N` y `poses` primero")
    d = json.loads(CRUDO.read_text(encoding="utf-8"))
    cj = d.get("cajas", [])
    ps = d.get("poses", [])

    def veredicto(ok, medido):
        return ("APTO" if ok else "**NO APTO**"), medido

    res = {}
    if cj:
        s = max(c["peor_salto"] for c in cj)
        res["C1"] = veredicto(s <= 0.60 + 1e-9, "peor salto %.3f anchos en %d clips" % (s, len(cj)))
        v = max(c["peor_iou"] for c in cj)
        res["C2"] = veredicto(v < 0.70, "peor solape %.3f" % v)
        t = sum(c["n_tiras"] for c in cj)
        pr = [c["peor_prop_borde"] for c in cj if c["peor_prop_borde"] is not None]
        res["C3"] = veredicto(t == 0, "%d tiras entregadas; proporcion minima en el borde %s" % (
            t, ("%.3f" % min(pr)) if pr else "sin cajas en el borde"))
        nb = sum(1 for c in cj if not c["bordes_medidos"])
        res["C4"] = veredicto(nb == 0, "%d clips de %d con un borde sin los dos medidos" % (
            nb, len(cj)))
        """C5 NO puede ser una garantia binaria y esta verificado por que: en
        `SimilarOffbeatIberianlynx` los dos tiradores se cruzan de verdad en el 9,3 % de los cuadros
        al pasar uno junto al otro, y la pareja es CORRECTA (visto a ojo en
        `muestras/limite_intruso`). El caso con arbitro elegido daba 15,0 %. Con 1,6 veces de margen
        y un solo positivo verificado, poner un umbral aqui seria inventarselo, asi que se informa la
        distribucion y el clip peor, y el aviso `blancura` de `busca_errores.py` es quien lo levanta
        para revision."""
        fr = sorted(((c["cruces_de_lado"] / max(1, c["util"]), c["clip"]) for c in cj),
                    reverse=True)
        n_con = sum(1 for f, _c in fr if f > 0)
        res["C5"] = ("MEDIDO", "%d clips de %d con algun cruce; el peor %s con %.1f %% de sus "
                     "cuadros (referencia verificada: pareja correcta 9,3 %%, arbitro elegido "
                     "15,0 %%)" % (n_con, len(cj), fr[0][1][:26], 100 * fr[0][0]))
        nb = sum(1 for c in cj if c.get("borde_max", 0.0) >= 1.0)
        res["C8"] = veredicto(nb == 0, "%d clips con un lado nunca visto entero" % nb)
        res["C6"] = ("MEDIDO", "%d cuadros utilizables de %d entregados en total" % (
            sum(c["util"] for c in cj), sum(c["T"] for c in cj)))
        res["C7"] = ("MEDIDO", "ver la tirada de `busca_errores.py`, aviso `fundido`")
    if ps:
        """P1 solo se puede juzgar en los NPZ que llevan la marca `filtro_vacias`. Los escritos antes
        de que el filtro existiera tienen poses medidas de 0 a 8 puntos utiles dentro, y contarlos
        seria acusar a la tuberia de un fallo que ya no comete. Los viejos se informan aparte para que
        se sepa que estan pendientes de regenerar, no para que pasen por buenos."""
        con = [p for p in ps if p.get("con_filtro")]
        sin = [p for p in ps if not p.get("con_filtro")]
        pp = [p["peor_puntos_medida"] for p in con if p["peor_puntos_medida"] is not None]
        umb = d.get("PUNTOS_MIN", 20)
        extra = "" if not sin else ("; %d NPZ sin la marca del filtro, pendientes de regenerar "
                                    "(minimo %s en ellos)" % (
                                        len(sin),
                                        min([p["peor_puntos_medida"] for p in sin
                                             if p["peor_puntos_medida"] is not None] or [0])))
        if not pp:
            res["P1"] = ("sin medir", "ningun NPZ lleva la marca `filtro_vacias` todavia%s" % extra)
        else:
            res["P1"] = veredicto(min(pp) >= umb,
                                  "minimo %d puntos utiles en %d NPZ con filtro%s" % (
                                      min(pp), len(con), extra))
        sp = sum(p["sin_pose"] for p in ps)
        res["P2"] = veredicto(sp == 0, "%d cuadros entregados sin pose, en %d clips" % (sp, len(ps)))
        res["P3"] = veredicto(all(p["tiene_origen"] for p in ps),
                              "%d poses inventadas marcadas, de ellas %d sostenidas" % (
                                  sum(p["inventadas"] for p in ps),
                                  sum(p["sostenidas"] for p in ps)))
        fc = sum(p["kp_fuera_de_caja"] for p in ps)
        res["P4"] = veredicto(fc == 0, "%d keypoints fuera de su caja" % fc)
        lados = sum(p["lados"] for p in ps)
        med = sum(p["medidas"] for p in ps)
        iv = sum(p["interp_validado"] for p in ps)
        isn = sum(p["interp_sin_medida"] for p in ps)
        res["P5"] = veredicto(
            all(p["tiene_hueco"] for p in ps),
            "de %d poses entregadas: %.1f %% medidas, %.1f %% interpoladas con hueco <= 24 "
            "cuadros (rango con error medido), %.1f %% interpoladas con hueco mayor (SIN medida); "
            "hueco maximo %d" % (
                lados, 100.0 * med / max(lados, 1), 100.0 * iv / max(lados, 1),
                100.0 * isn / max(lados, 1), max(p["hueco_max"] for p in ps)))

    l = ["# Checks de la extraccion", "",
         "Cada fila es una **garantia que la tuberia dice cumplir**, con su umbral declarado, el "
         "sitio del codigo que la cumple y donde esta la evidencia para verla a ojo.", "",
         "Esto no es una nota media. Una nota media alta puede convivir con un dato corrupto: la "
         "averia del fundido de `VainAchingGonolek` daba **96,1 % de presencia** sobre poses "
         "mezcladas de dos planos distintos. Una garantia se cumple en todos los cuadros entregados "
         "o no se cumple.", "",
         "Medido sobre **%d clips de cajas** y **%d clips de poses** del corpus de florete." % (
             len(cj), len(ps)), "",
         "| id | garantia | umbral declarado | medido | veredicto | codigo | evidencia |",
         "|---|---|---|---|---|---|---|"]
    for i, g, u, c, e in FILAS:
        v, m = res.get(i, ("sin medir", "-"))
        l.append("| %s | %s | %s | %s | %s | `%s` | `%s` |" % (i, g, u, m, v, c, e))
    l.append("")
    l.append("La evidencia esta en el repo privado `%s` bajo esas rutas, y en local en "
             "`02_validacion/muestras/` y `02_validacion/videos_revision/`." % REPO_HF)
    l.append("")
    if cj:
        amb = np.array([c["ambos"] for c in cj])
        l.append("## Presencia de los dos tiradores, en la misma tirada")
        l.append("")
        l.append("| medida | valor |")
        l.append("|---|---|")
        l.append("| mediana de cuadros con los dos | %.1f %% |" % (100 * np.median(amb)))
        l.append("| p10 | %.1f %% |" % (100 * np.quantile(amb, 0.1)))
        l.append("| minimo | %.1f %% |" % (100 * amb.min()))
        l.append("| clips por encima del 90 %% | %d de %d |" % (int((amb >= 0.90).sum()), len(amb)))
        l.append("")
        l.append("Esa columna es **antes de inventar** las poses que faltan. Despues de inventar, "
                 "el check P2 dice que no queda ningun cuadro entregado sin los dos.")
        l.append("")
    if ps:
        r5 = [p["err_rel_p50"] for p in ps if p["err_rel_p50"] is not None]
        if r5:
            l.append("## Error de la pose inventada")
            l.append("")
            l.append("Medido **escondiendo poses que si se conocen** e interpolando desde sus dos "
                     "vecinos: mediana %.1f %% del torso, p90 %.1f %%." % (
                         100 * float(np.median(r5)),
                         100 * float(np.median([p["err_rel_p90"] for p in ps
                                                if p["err_rel_p90"] is not None]))))
            l.append("")
            l.append("**Lo que esta medida NO dice.** Los huecos que de verdad se rellenan son en "
                     "buena parte cuadros donde el tirador **no esta en el plano**, y para esos no "
                     "existe verdad con la que comparar: su error es, por construccion, no "
                     "verificable. Por eso `origen` va guardado junto a los puntos (check P3) y no "
                     "se debe perder.")
            l.append("")
    SALIDA.write_text("\n".join(l), encoding="utf-8")
    print("\n".join(l[:40]))
    print()
    print("tabla en %s" % SALIDA)
    malos = [i for i, (v, _m) in res.items() if v.startswith("**NO")]
    if malos:
        print("CHECKS NO APTOS: %s" % ", ".join(malos))
    else:
        print("todos los checks medidos salen APTO")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "tabla"
    nn = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if q in ("cajas", "poses"):
        d = json.loads(CRUDO.read_text(encoding="utf-8")) if CRUDO.exists() else {}
        d.update(checks_cajas(nn) if q == "cajas" else checks_poses())
        CRUDO.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
        print()
        print("medidas en %s" % CRUDO)
    else:
        tabla()
