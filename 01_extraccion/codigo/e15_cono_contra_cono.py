"""ETAPA E, paso 15: CONO CONTRA CONO. Y la primera pregunta es si el cono informa de algo.

EL PLANTEAMIENTO CAMBIA, y el cambio es del usuario. Sus palabras: *"lo de los conos no es con el
torso, es con el OTRO CONO, sabes? Pero bueno, me sirve. E intenta mejorarlo, que no esta demasiado
bien, igual lo puedes hacer mejor."*

Y tiene razon en algo que no es un detalle: los pasos 10 y 11 midieron el cono contra el TORSO del
rival porque asi venia el encargo, y eso mide otra cosa. Las dos preguntas son distintas y las dos
valen, pero no son la misma:

    cono contra TORSO  ->  donde puede haber AMENAZA a la superficie valida
    cono contra CONO   ->  donde puede haber CONTACTO ENTRE LAS DOS HOJAS

Y el contacto entre hojas es la senal que le falta al proyecto entero: el detector de prioridad se
cerro con **0 paradas anotadas** en 15,4 s de accion porque nadie consigue ver el choque de hojas, y
el reglamento de florete hace de la parada la bisagra que pasa la prioridad a la respuesta (t.85 y
siguientes), asi que cinco de las doce reglas se quedan sin medir. Si el solape de conos fuera un
proxy utilizable de contacto, reabre esa via.

--------------------------------------------------------------------------------------------------
LA PRIMERA MEDIDA ES SI ESTO PUEDE INFORMAR DE ALGO, Y VA ANTES QUE CUALQUIER OTRA

Un cono con semiapertura de 23-27 grados y 90 cm de radio es una region ENORME. Si los conos de los
dos tiradores se solapan practicamente siempre, entonces «los conos se solapan» no distingue un
cuadro de otro y **no informa de nada**, por bien construido que este. Asi que lo primero que se mide
es la FRECUENCIA de solape, y su criterio va declarado aqui:

  CRITERIO 1, DECLARADO ANTES DE MIRAR. El solape informa si su frecuencia esta **entre el 5 % y el
  80 %** de los cuadros con los dos tiradores presentes. Por encima del 80 % la senal esta saturada
  (casi siempre «si») y por debajo del 5 % esta muerta (casi siempre «no»); en los dos casos la
  columna es casi constante y no puede predecir nada. El contacto de hojas real es un suceso poco
  frecuente, asi que lo esperable de un buen proxy es la parte baja de ese rango.

  Y SI SALE SATURADO, el defecto principal es la APERTURA y hay que decirlo con su cifra. Para eso
  se BARRE la apertura de 4 a 30 grados y se da la frecuencia de solape en cada una: eso convierte
  «el cono esta demasiado abierto» en un numero y dice cual seria la apertura que informa.

  CRITERIO 2, LA LINEA BASE GRATIS QUE HAY QUE BATIR. Si el solape de conos no dice nada que no diga
  ya la **distancia entre las dos manos**, no aporta. Se mide la correlacion entre las dos y, sobre
  todo, si el solape aniade separacion cuando la distancia de manos se mantiene fija. La distancia
  entre manos cuesta 0 ms y sale de dos puntos de la pose.

--------------------------------------------------------------------------------------------------
QUE SE REUTILIZA Y QUE NO SE REESCRIBE

`cono_lib.cruce` **ya calcula la interseccion de los dos conos** en 2D: la escribio `e3` y su
reproducibilidad ya esta medida (el booleano «se cruzan» coincide en el 97,3-97,4 % al cambiar la
resolucion de la pose, `peso_max` da pearson 0,942-0,962). Aqui se llama, no se reimplementa. Lo
unico que cambia respecto a `e3` es el EJE: `e3` usaba `codo -> raiz` y aqui se usa
`hombro -> raiz`, que es la variable operativa medida (cv del largo proyectado 0,108 para el codo,
que es un hueso, contra 0,226 para el hombro, que es el que se extiende).

De `cruce` se usan las tres salidas y cada una dice algo distinto:
    `area_rel`  area del solape sobre el area del cono menor: CUANTO se solapan
    `peso_max`  maximo del producto de los dos pesos; vale 1 solo si un punto esta en los DOS ejes
                a la vez, o sea que es la senal de «las dos hojas apuntan al mismo sitio»
    `px, py`    DONDE se alcanza ese maximo, que es la posicion candidata del contacto

--------------------------------------------------------------------------------------------------
LO QUE ESTE PASO NO HACE

No mide 3D, y no hace falta reabrirlo: ya esta cerrado que la z de `pointmap` no da profundidad
utilizable (C3 pasa 14 de 40, el escorzo 0 de 10, y las discrepancias 2D/3D son indistinguibles de
barajar la z al azar, 1,00x). Pero si se puede afirmar una asimetria SIN medir profundidad, y es
util: **el solape 2D es COTA SUPERIOR del solape 3D**, porque dos conos que no se cruzan en la
proyeccion tampoco se cruzan en el espacio. O sea que un «no se solapan» en 2D es un «no hay
contacto» firme, y un «si se solapan» es solo un candidato. Es la simetrica de lo que ya se dijo de
la distancia lateral, que es cota INFERIOR de la 3D.
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
"""Semiapertura operativa, en grados. Es la mediana de |angulo hoja - eje hombro-mano| que midio
`e14` agrupando los tres clips con hoja detectada, y **su n es 6**. No es una constante solida: se
usa porque es la unica cifra medida contra el eje que se usa, y por eso el barrido de abajo es la
parte que de verdad manda."""
BARRIDO = [4, 6, 8, 10, 12, 15, 18, 21, 23.1, 26, 30]
FREC_MIN, FREC_MAX = 0.05, 0.80


def clips_con_poses():
    """Clips del CORPUS con poses de 308 puntos ya guardadas. Sin GPU y sin `os.walk` del arbol."""
    corpus = set(x.strip() for x in
                 (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    out = []
    for f in sorted(PRUEBAS.glob("cono_poses_*.npz")):
        nom = f.name[len("cono_poses_"):-4]
        if nom.endswith("_512x384") or nom not in corpus:
            continue
        out.append(nom)
    return out


def geometria(clip):
    """Los dos conos por cuadro, con el eje anclado en el HOMBRO. Todo de los NPZ, sin GPU."""
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        pres = z["presente"]
        mask = z["mask"] if "mask" in z.files else np.ones(T, dtype=bool)
        video = str(z["video"])
    g = {"W": W, "H": H, "T": T, "video": video, "lados": {}}
    for lado in (0, 1):
        mano, det = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            return None
        vert, u, ok, largo = C3.eje_brazo_2d(kp, sc, lado, mano, "hombro")
        _pk, pxm = CL.escala_px_m(B, lado, H)
        g["lados"][lado] = {"mano": mano, "vert": vert, "u": u, "ok": ok, "pxm": pxm,
                            "L": CL.HOJA_M * pxm, "margen_mano": det.get("margen_x_px", 0.0)}
    g["dos"] = np.array([bool(mask[k] and pres[k].sum() == 2) for k in range(T)])
    g["kp"], g["sc"], g["B"] = kp, sc, B
    return g


def mide(g, apert):
    """Recorre el clip y devuelve las columnas del solape y las dos lineas base gratis."""
    T = g["T"]
    d0, d1 = g["lados"][0], g["lados"][1]
    col = {n: np.full(T, np.nan) for n in ("area_rel", "peso_max", "px", "py", "d_ejes_cm",
                                           "d_manos_cm", "d_tiradores_cm", "ang_ejes")}
    cru = np.zeros(T, dtype=bool)
    val = np.zeros(T, dtype=bool)
    pxm_med = 0.5 * (d0["pxm"] + d1["pxm"])
    for k in range(T):
        if not (g["dos"][k] and d0["ok"][k] and d1["ok"][k]):
            continue
        val[k] = True
        r = CL.cruce(d0["vert"][k], d0["u"][k], d0["L"], apert,
                     d1["vert"][k], d1["u"][k], d1["L"], apert)
        col["area_rel"][k] = r["area_rel"]
        col["peso_max"][k] = r["peso_max"]
        col["px"][k], col["py"][k] = r["px"], r["py"]
        cru[k] = r["n_inter"] > 0
        dd = CL.dist_segmentos(d0["vert"][k], d0["vert"][k] + d0["L"] * d0["u"][k],
                               d1["vert"][k], d1["vert"][k] + d1["L"] * d1["u"][k])
        col["d_ejes_cm"][k] = 100.0 * dd / pxm_med
        col["d_manos_cm"][k] = 100.0 * float(np.linalg.norm(
            d0["vert"][k] - d1["vert"][k])) / pxm_med
        b0, b1 = g["B"][k, 0], g["B"][k, 1]
        if np.isfinite(b0).all() and np.isfinite(b1).all():
            c0 = np.array([0.5 * (b0[0] + b0[2]) * g["W"], 0.5 * (b0[1] + b0[3]) * g["H"]])
            c1 = np.array([0.5 * (b1[0] + b1[2]) * g["W"], 0.5 * (b1[1] + b1[3]) * g["H"]])
            col["d_tiradores_cm"][k] = 100.0 * float(np.linalg.norm(c0 - c1)) / pxm_med
        col["ang_ejes"][k] = abs(CL.dif_ang(
            float(np.degrees(np.arctan2(d0["u"][k][1], d0["u"][k][0]))),
            float(np.degrees(np.arctan2(d1["u"][k][1], d1["u"][k][0])))))
    return col, cru, val


def spearman(a, b):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 15:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype("float64")
    rb = np.argsort(np.argsort(b[m])).astype("float64")
    return float(np.corrcoef(ra, rb)[0, 1])


CLIPS = clips_con_poses()
print("=" * 104)
print("CONO CONTRA CONO  ·  %d clips del corpus con poses guardadas  ·  eje HOMBRO -> raiz  ·  "
      "sin GPU" % len(CLIPS))
print("=" * 104)

resumen, barr = {}, {a: [0, 0] for a in BARRIDO}
for clip in CLIPS:
    g = geometria(clip)
    if g is None:
        print("  %-34s sin mano armada decidible: saltado" % clip[:34])
        continue
    col, cru, val = mide(g, APERT)
    n = int(val.sum())
    if n < 30:
        print("  %-34s solo %d cuadros validos: saltado" % (clip[:34], n))
        continue
    frec = float(cru[val].mean())
    print()
    print("  %s  ·  %dx%d  ·  %d cuadros validos de %d" % (clip, g["W"], g["H"], n, g["T"]))
    print("    SE SOLAPAN los dos conos en %d de %d = **%.1f %%**  ->  %s" % (
        int(cru[val].sum()), n, 100 * frec,
        "informa" if FREC_MIN <= frec <= FREC_MAX else
        ("SATURADO: el cono es demasiado ancho" if frec > FREC_MAX else "MUERTO: casi nunca")))
    for nom, uni in (("area_rel", ""), ("peso_max", ""), ("d_ejes_cm", " cm"),
                     ("d_manos_cm", " cm"), ("d_tiradores_cm", " cm"), ("ang_ejes", " gr")):
        v = col[nom][val]
        v = v[np.isfinite(v)]
        if v.size:
            print("    %-16s mediana %7.1f%s  p10 %7.1f  p90 %7.1f" % (
                nom, np.median(v), uni, np.percentile(v, 10), np.percentile(v, 90)))
    rho_m = spearman(col["peso_max"][val], -col["d_manos_cm"][val])
    rho_t = spearman(col["peso_max"][val], -col["d_tiradores_cm"][val])
    print("    LINEA BASE GRATIS: spearman(peso_max, -dist manos) = %+.3f  ·  "
          "(peso_max, -dist tiradores) = %+.3f" % (rho_m, rho_t))
    for a in BARRIDO:
        c2, v2 = mide(g, a)[1:]
        barr[a][0] += int(c2[v2].sum())
        barr[a][1] += int(v2.sum())
    resumen[clip] = {"n": n, "frec_solape": frec, "W": g["W"], "H": g["H"],
                     "area_rel_med": float(np.nanmedian(col["area_rel"][val])),
                     "peso_max_med": float(np.nanmedian(col["peso_max"][val])),
                     "d_manos_med_cm": float(np.nanmedian(col["d_manos_cm"][val])),
                     "d_ejes_med_cm": float(np.nanmedian(col["d_ejes_cm"][val])),
                     "rho_manos": rho_m, "rho_tiradores": rho_t,
                     "margen_mano_armada": [g["lados"][l]["margen_mano"] for l in (0, 1)]}
    f = PRUEBAS / ("cono_cono_%s.csv" % clip)
    with f.open("w", encoding="utf-8") as fh:
        fh.write("cuadro,valido,se_solapan,area_rel,peso_max,px,py,d_ejes_cm,d_manos_cm,"
                 "d_tiradores_cm,ang_ejes\n")
        for k in range(g["T"]):
            fh.write("%d,%d,%d,%s\n" % (k, int(val[k]), int(cru[k]), ",".join(
                "" if not np.isfinite(col[nn][k]) else "%.4f" % col[nn][k]
                for nn in ("area_rel", "peso_max", "px", "py", "d_ejes_cm", "d_manos_cm",
                           "d_tiradores_cm", "ang_ejes"))))

print()
print("=" * 104)
print("EL BARRIDO DE APERTURA: convierte «el cono esta demasiado abierto» en un numero")
print("=" * 104)
print("  %-10s %-14s %s" % ("apertura", "solape", "veredicto contra el criterio declarado 5-80 %"))
for a in BARRIDO:
    s_, t_ = barr[a]
    if t_ == 0:
        continue
    fr = s_ / t_
    print("  %-10.1f %6.1f %% (%5d/%5d)  %s" % (
        a, 100 * fr, s_, t_,
        "informa" if FREC_MIN <= fr <= FREC_MAX else
        ("SATURADO" if fr > FREC_MAX else "muerto")))
if resumen:
    fr = [v["frec_solape"] for v in resumen.values()]
    print()
    print("  A LA APERTURA OPERATIVA de %.1f gr: solape mediana %.1f %% (rango %.1f-%.1f) sobre "
          "%d clips" % (APERT, 100 * float(np.median(fr)), 100 * min(fr), 100 * max(fr),
                        len(resumen)))
(PRUEBAS / "cono_e15_cono_cono.json").write_text(
    json.dumps({"apert_operativa": APERT, "criterio": [FREC_MIN, FREC_MAX],
                "barrido": {str(a): barr[a] for a in BARRIDO}, "clips": resumen},
               indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e15_cono_cono.json"))
