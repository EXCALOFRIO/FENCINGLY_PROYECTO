"""ETAPA E, paso 10: EL EJE 3D DEL ARMA, EL CONO 3D Y SU INTERSECCION CON EL TORSO DEL RIVAL.

LO QUE PIDIO EL USUARIO, en sus palabras: *"hay que ver que sea 3D, porque si no no tiene sentido en
2D"*, y antes *"con el alfa 0,0006 se puede ver el florete, el filo super fino; esto nos puede ayudar
a detectarlo para lo del cono"*. Su razonamiento es correcto y este paso lo respeta: dos floretes que
se cruzan en la imagen pueden estar a un metro en profundidad, asi que una interseccion medida en 2D
puede ser falsa.

--------------------------------------------------------------------------------------------------
1. EL EJE 3D: QUE SE MIDE Y QUE SE SUPONE, SEPARADO

  MEDIDO    la DIRECCION PROYECTADA del arma. Sale del filo visto en el alfa de `matting`, con el
            metodo de `e5` sin cambiar un parametro. Son dos numeros sacados de pixeles reales.
  MEDIDO    la ELEVACION del BRAZO, o sea cuanto sale del plano de la imagen. Sale de la z de
            `pointmap` en **hombro y mano**, y su fiabilidad es lo que mide `e9`.
  SUPUESTO  **que la hoja saca del plano lo mismo que el brazo.** Y esto NO se puede medir con lo
            que hay: la hoja tiene 0,64-0,96 px de ancho, asi que `pointmap` en un pixel de hoja lee
            el FONDO y no la hoja. Es la unica suposicion de la construccion.

  EL ANCLA DEL EJE ES EL HOMBRO, NO EL CODO, y es una correccion al encargo de este paso. `codo ->
  raiz` es un HUESO: no crece al extender el brazo, asi que no mide extension. La variable operativa
  medida del proyecto es `hombro -> raiz` (`ext_hr` en `p1_variables.py`), y la tabla C3 de `e9` lo
  corrobora por su cuenta: el cv del largo proyectado vale 0,065-0,121 para `codo -> raiz` contra
  0,163-0,336 para `hombro -> raiz`, o sea 2,5-3,4 veces mas. El rigido es el primero.
  Aun asi, **cual de los dos predice la DIRECCION de la hoja se decide midiendo**, no por argumento:
  ver el careo del apartado 2.

  Con la proyeccion medida `(dx, dy)` y la elevacion `phi`, el eje queda determinado:

      u3d = (cos(phi) * dx,  cos(phi) * dy,  sin(phi))

  EL SIGNO se resuelve en `cono3d_lib.hoja_en_alfa`, orientando el segmento de Hough hacia el lado
  que se ALEJA DEL CODO, porque la punta esta delante de la mano y el brazo detras. **No se usa
  «apunta al rival»**, y eso es deliberado: si el signo se eligiera para que el arma mire al rival,
  la interseccion con el rival quedaria garantizada por construccion y la medida no valdria nada.

2. LAS DOS APERTURAS, cada una medida y emparejada con su origen

  `e6` midio la apertura como el p90 de |angulo hoja - eje del antebrazo| y salio 24,8-29,7 grados,
  o sea 2,4-2,8 veces el techo de utilidad: **el cono no localiza la punta si la hoja se SUPONE
  desde el brazo.** Pero esa cifra es el error de PREDECIR la hoja desde el brazo, y cuando la hoja
  se VE ese error no aplica. Asi que aqui hay dos aperturas y cada cono usa la suya:

    apertura SUPUESTA  = p90 de |angulo hoja - angulo del eje GANADOR|    <- regla de `e6`, sin tocar
    apertura MEDIDA    = p90 de |cambio del angulo de la HOJA entre cuadros consecutivos donde se
                         detecta en los dos|                              <- el temblor propio de la
                         medida de la hoja, que es la analoga del «suelo» de `e3` pero para la hoja

  Y ANTES DE ESO, EL CAREO DE LOS TRES EJES CANDIDATOS. Los cuadros con hoja VISTA son los unicos
  con verdad-terreno de la direccion del arma, asi que se usan para decidir que eje la predice:
  `hombro -> raiz`, `codo -> raiz` y `raiz -> pulgar`, con el azar declarado en 45 grados. El eje
  ganador es el que se usa en los cuadros SIN hoja, y la `apertura SUPUESTA` se mide contra ese
  mismo eje: la apertura ES el error del eje, asi que emparejarla con otro eje seria incoherente.

  A 720p ESTO NO SE PUEDE HACER, y condiciona el alcance de todo el paso: la hoja se ve en el 1,6 %
  de los recortes a 720p contra el 42,6 % a 1080p, y el corpus es 81,2 % de 720p. O sea que el cono
  con hoja VISTA solo es construible en unos 1.540 de los 8.559 clips. `e13` lo mide por separado.

  Las dos se comparan con el TECHO DE UTILIDAD que ya se definio en `e3`: la apertura para la que el
  cono a 90 cm es tan ancho como los hombros del rival. Por encima del techo, «la punta esta en el
  cono» equivale a «la punta esta en algun sitio del rival», que no es informacion.

3. LA SUPERFICIE VALIDA, que en FLORETE es el TORSO y solo el torso

  Ni brazos, ni piernas, ni careta. Se toma el cuadrilatero hombros 5-6 / caderas 9-10 del mapa
  MEDIDO, se lee la z de sus cuatro vertices en el `pointmap` DEL CUADRO COMPLETO (que es la unica
  via que puede dar z comparable entre los dos tiradores) y se muestrea con una rejilla de 7x7.

4. LA COMPARACION 2D CONTRA 3D, montada para que solo cambie UNA cosa

  El mismo vertice, el mismo eje proyectado, la misma apertura, el mismo L, el mismo torso y **la
  misma forma de peso** `1 - angulo/apertura`. Lo unico que cambia es si la tercera coordenada
  entra o no. Y la conversion de unidades degrada limpiamente: si la z fuera constante, el 3D
  coincidiria EXACTAMENTE con el 2D y no habria ni una discrepancia (demostrado en el docstring de
  `cono3d_lib.p3d_exacto`). Asi que **toda discrepancia que aparezca viene de la z y de nada mas.**

  LO QUE DE VERDAD JUSTIFICA ESTE PASO son los cuadros donde el 2D dice que amenaza y el 3D dice que
  no, o al contrario. Se buscan a proposito, se cuentan, y se saca la imagen de los mas extremos. Y
  si no hubiera ninguno, esa tambien es una respuesta valida y hay que decirla: significaria que el
  3D no cambia la decision y solo anade coste.

  Los dos mecanismos por los que el 3D puede cambiar la decision, y conviene separarlos porque no
  dicen lo mismo:
    ALCANCE     el torso esta a mas de 90 cm en 3D aunque en la imagen parezca cerca. Es el
                argumento literal del usuario y el que mas peso fisico tiene.
    DIRECCION   el eje sale del plano, asi que el torso que en la imagen cae dentro del sector en 3D
                se queda fuera del cono (o al contrario).

5. EL CONO COMO DATO GUARDADO, con la trazabilidad que el proyecto ya exige a `origen` y `tipo`

  Se guarda por cuadro y por tirador: vertice 3D, eje 3D, apertura, L, la marca de amenaza en 2D y
  en 3D, y **`origen_cono`**: 0 = hoja VISTA en el alfa, 1 = hoja SUPUESTA desde el brazo, -1 = no
  hay cono. Sin esa columna nadie podria volver a distinguir un cono medido de uno inventado, y
  entonces el dato inventado envenena el entrenamiento. Es la misma regla que ya se aplico a
  `origen` (medido/interpolado/sostenido) y a `tipo`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL
import e7_pinta_cono as E7

import cv2

PRUEBAS = CL.PRUEBAS
MUESTRAS = pathlib.Path(
    r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\muestras\cono3d")
MUESTRAS.mkdir(parents=True, exist_ok=True)
CLIPS = ["AromaticShockedGermanspitz", "ImpeccablePertinentBasilisk", "WelloffForcefulIndri",
         "SeparateHoarseFeline", "IdealFrigidJerboa"]
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
"""Sufijo de VENTANA, el mismo que usa `e9`: vacio = ventana quieta, `_accion` = ventana de
maxima cercania. Los dos juegos de datos conviven y no se pisan."""
N_MALLA = 7
N_FASE_A = 50
"""CUANTOS CUADROS DE LA VENTANA PASAN POR `matting`, y por que hay tope.

MEDIDO en esta maquina y en estas condiciones: `matting` a 1024x768 mas `seg` cuesta **~10 s por
recorte** con la GPU compartida con otros nueve procesos, contra los 767 + 245 ms que mide el
proyecto con la GPU libre. A ese precio los 100 cuadros x 2 tiradores x 5 clips serian **2,8 horas**
solo de fase A.

Se recortan a los **50 cuadros CENTRALES y CONSECUTIVOS** de la ventana de accion, que son los mas
cercanos al tocado. Consecutivos y no salteados porque la `apertura MEDIDA` es el temblor de la hoja
entre cuadros CONTIGUOS y una muestra salteada no la puede medir.

Y LA FASE B SE RESTRINGE A ESOS MISMOS CUADROS. Si la geometria se evaluara sobre los 100 mientras la
hoja solo se busco en 50, la mitad de los conos tendria `origen = 1` **por construccion** y la
fraccion de «hoja vista» saldria dividida por dos sin que nada lo dijera."""
COL = {0: (255, 170, 40), 1: (60, 210, 255)}
GRIS = (150, 150, 150)
SOLO_CACHE = os.environ.get("CONO3D_SOLO_CACHE", "") == "1"
"""MODO SIN GPU: solo se usan los clips cuya fase A ya esta cacheada, y los demas se SALTAN.

POR QUE EXISTE. La GPU de esta maquina la comparten varios agentes y se midio llena a **11,9 de
12,3 GB**, con lo que `matting-1b` (6 GB) no cabe y el paso muere sin traza. Con este interruptor la
geometria, la comparacion 2D/3D y las figuras se completan **con los clips que ya estan medidos**, en
CPU, en vez de perderlo todo por un recurso que no controlo.

Y LO IMPORTANTE: los clips que se saltan **se dicen por su nombre y se cuentan**. Un resultado sobre
3 de 5 clips no puede parecerse a un resultado sobre 5."""
HUMO = int(os.environ.get("CONO3D_MAX_A", "0")) > 0
"""EN MODO PRUEBA DE HUMO NO SE ESCRIBE NADA. Una prueba de humo con 4 cuadros produce aperturas sin
`n` suficiente y conteos de 8 conos; si eso se guardara con el mismo nombre que la medida de verdad,
mas tarde nadie sabria cual es cual. El proyecto ya tiene el escarmiento de dar por buena una cifra
mal medida, asi que aqui la escritura esta cerrada por construccion y no por acordarse."""


def fase_a(clip, lect, ks, mano, pxm, W, H, video, KP, SC, B):
    """FASE A, la unica que gasta GPU: detectar la hoja en el alfa en cada cuadro y cada tirador.

    Se cachea en JSON porque toda la geometria de despues es de CPU y hay que poder reiterarla sin
    volver a pagar `matting` (767 ms) mas `seg` (245 ms) por recorte.
    """
    f = PRUEBAS / ("cono_hoja3d_%s%s.json" % (clip, SUF))
    if f.is_file():
        print("  fase A: reutilizada de %s" % f.name)
        fl = json.loads(f.read_text(encoding="utf-8"))
        return fl, sorted(set(int(r["cuadro"]) for r in fl)) or list(ks)
    if SOLO_CACHE:
        print("  fase A: NO cacheada y CONO3D_SOLO_CACHE=1  ->  clip SALTADO")
        return None, None
    cap = cv2.VideoCapture(video)
    filas, t0 = [], time.perf_counter()
    n_a = min(N_FASE_A, len(ks))
    a = (len(ks) - n_a) // 2
    ks = ks[a:a + n_a]
    tope = int(os.environ.get("CONO3D_MAX_A", "0")) or len(ks)
    """`CONO3D_MAX_A` acorta la fase A para una prueba de humo. En 0 (por defecto) no acorta nada:
    una prueba de humo NUNCA debe poder pasar por una medida, asi que si esta puesto se avisa."""
    if tope < len(ks):
        print("  AVISO: PRUEBA DE HUMO, solo %d de %d cuadros. Estos numeros NO son una medida."
              % (tope, len(ks)))
    for k in ks[:tope]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = cap.read()
        if not okf:
            continue
        for lado in (0, 1):
            M = CL.MANOS[mano[lado]]
            pc, pr = KP[k, lado, M["codo"]], KP[k, lado, M["root"]]
            if not (np.isfinite(pc).all() and np.isfinite(pr).all()
                    and SC[k, lado, M["codo"]] >= CL.CONF_MIN
                    and SC[k, lado, M["root"]] >= CL.CONF_MIN):
                continue
            v = pr - pc
            nv = float(np.linalg.norm(v))
            if nv < 1.0:
                continue
            u = (v / nv).astype("float64")
            vert = pr.astype("float64")
            """LOS TRES CANDIDATOS DE EJE, para poder decidir por MEDIDA cual predice la hoja."""
            ph = KP[k, lado, M["hombro"]]
            u_hom = None
            if np.isfinite(ph).all() and SC[k, lado, M["hombro"]] >= CL.CONF_MIN:
                vh = pr - ph
                if float(np.linalg.norm(vh)) > 1.0:
                    u_hom = (vh / float(np.linalg.norm(vh))).astype("float64")
            u_pul = C3.eje_pulgar_2d(KP[k, lado], SC[k, lado], mano[lado])
            L = CL.HOJA_M * pxm[lado]
            (cx0, cy0, cx1, cy1), R = C3.recorte_hoja(B[k, lado], B[k, 1 - lado], vert, u, L, W, H)
            if cx1 - cx0 < 64 or cy1 - cy0 < 64 or R < 24:
                continue
            sub = np.ascontiguousarray(im[cy0:cy1, cx0:cx1, ::-1])
            alfa, seg = E7.infiere(sub)
            am = float(alfa.mean())
            ln = C3.hoja_en_alfa(cv2, alfa, seg, vert, u, R, (cx0, cy0))
            filas.append({
                "cuadro": int(k), "lado": lado, "R": R, "alfa_recorte": am,
                "contaminado": bool(am < 0.05), "recorte": [cx0, cy0, cx1, cy1],
                "ang_eje": float(np.degrees(np.arctan2(u[1], u[0]))),
                "ang_hombro": (float(np.degrees(np.arctan2(u_hom[1], u_hom[0])))
                               if u_hom is not None else None),
                "ang_pulgar": (float(np.degrees(np.arctan2(u_pul[1], u_pul[0])))
                               if u_pul is not None else None),
                "linea": None if ln is None else {
                    "p1": ln["p1"].tolist(), "p2": ln["p2"].tolist(), "largo": ln["largo"],
                    "ang": ln["ang"], "t_min": ln["t_min"], "t_max": ln["t_max"],
                    "delante": ln["delante"], "n_px": ln["n_px"],
                    "dif_codo": C3.dif90(ln["ang"], float(np.degrees(np.arctan2(u[1], u[0])))),
                    "dif_hombro": (C3.dif90(ln["ang"], float(np.degrees(
                        np.arctan2(u_hom[1], u_hom[0])))) if u_hom is not None else None),
                    "dif_pulgar": (C3.dif90(ln["ang"], float(np.degrees(
                        np.arctan2(u_pul[1], u_pul[0])))) if u_pul is not None else None)}})
    cap.release()
    dt = time.perf_counter() - t0
    print("  fase A: %d recortes en %.1f s  ·  %.0f ms/recorte (GPU COMPARTIDA: indicativo)" % (
        len(filas), dt, 1000.0 * dt / max(len(filas), 1)))
    if tope >= len(ks):
        f.write_text(json.dumps(filas, indent=1, default=float), encoding="utf-8")
    return filas, list(ks[:tope])


def calibra(filas, kp, sc, B, W, H, pxm, mano, eje=C3.EJE_POR_DEFECTO):
    """Las dos aperturas, el techo de utilidad y **el careo de los tres ejes candidatos**.

    EL CAREO ES LO NUEVO Y ES UNA CORRECCION AL ENCARGO. Los cuadros donde la hoja SE VE en el alfa
    son los unicos con verdad-terreno de la direccion del arma. Contra ella se comparan:

        `hombro -> raiz de la mano`   la variable operativa medida del proyecto (`ext_hr`)
        `codo -> raiz de la mano`     lo que decia el encargo, y que es un HUESO
        `raiz -> centroide del pulgar` lo que se propone como sustituto a 720p

    con el AZAR DECLARADO en 45 grados: si un candidato no baja claramente de 45, no predice nada.
    La `apert_supuesta` se calcula contra el eje GANADOR y no contra uno elegido de antemano, porque
    esa apertura es precisamente el error de ese eje y mezclarlos daria una apertura que no
    corresponde al eje que se usa.
    """
    out = {}
    for lado in (0, 1):
        fs = [f for f in filas if f["lado"] == lado and not f["contaminado"]]
        val = [f for f in fs if f["linea"] and f["linea"]["delante"]]

        def col(nom):
            return np.array([f["linea"][nom] for f in val
                             if f["linea"].get(nom) is not None])
        d_hom, d_cod, d_pul = col("dif_hombro"), col("dif_codo"), col("dif_pulgar")
        cand = {"hombro": d_hom, "codo": d_cod, "pulgar": d_pul}
        med = {k: (float(np.median(v)) if v.size >= 5 else None) for k, v in cand.items()}
        elegibles = {k: v for k, v in med.items() if v is not None and k != "pulgar"}
        ganador = min(elegibles, key=elegibles.get) if elegibles else eje
        dif = cand[ganador] if cand[ganador].size else d_cod

        por_k = {f["cuadro"]: f["linea"]["ang"] for f in val}
        d1 = np.array([C3.dif90(por_k[k], por_k[k - 1]) for k in sorted(por_k)
                       if (k - 1) in por_k])
        rival = 1 - lado
        oh = CL.bueno(sc[:, rival], list(CL.HOMBROS))
        anch = float(np.median(np.linalg.norm(
            kp[oh, rival, CL.HOMBROS[1]] - kp[oh, rival, CL.HOMBROS[0]], axis=1))) \
            if oh.sum() >= 10 else float("nan")
        L = CL.HOJA_M * pxm[lado]
        out[lado] = {
            "n_recortes": len(fs), "n_con_recta": sum(1 for f in fs if f["linea"]),
            "n_delante": len(val),
            "frac_delante": len(val) / max(sum(1 for f in fs if f["linea"]), 1),
            "dif_mediana": {k: med[k] for k in med},
            "n_por_eje": {k: int(v.size) for k, v in cand.items()},
            "eje_ganador": ganador, "azar": C3.AZAR,
            "dif_eje_mediana": float(np.median(dif)) if dif.size else None,
            "apert_supuesta": float(np.percentile(dif, 90)) if dif.size >= 5 else None,
            "apert_medida": float(np.percentile(d1, 90)) if d1.size >= 5 else None,
            "n_pares_hoja": int(d1.size), "anch_hombros_rival": anch, "L_px": L,
            "techo": float(np.degrees(np.arctan(0.5 * anch / max(L, 1e-9))))}
    return out


resumen = {}
discrepantes = []
saltados = []
for clip in CLIPS:
    fz = PRUEBAS / ("cono_z3d_%s%s.npz" % (clip, SUF))
    fp = PRUEBAS / ("cono_poses_%s.npz" % clip)
    if not (fz.is_file() and fp.is_file()):
        print("faltan datos de %s (corre e9 primero)" % clip)
        continue
    lect, ks, mano, pxm, W, H, video = C3.carga_z(clip, SUF)
    with np.load(fp, allow_pickle=True) as z:
        KP, SC, B, T = z["kp"], z["sc"], z["B"], int(z["T"])
        tipo_pose = z["tipo"] if "tipo" in z.files else np.full((T, 2), -1, dtype="int8")
        mask_pose = z["mask"] if "mask" in z.files else np.ones(T, dtype=bool)
        pres_pose = z["presente"] if "presente" in z.files else np.zeros((T, 2), dtype=bool)
        fps_pose = float(z["fps"]) if "fps" in z.files else 0.0
    f_med, f_disp, f_det = C3.focal(lect, W, H)
    print("=" * 104)
    print("%s   %dx%d   ventana %d-%d (%d cuadros)" % (clip, W, H, ks[0], ks[-1], len(ks)))
    print("=" * 104)
    print("  C4 COHERENCIA DE CAMARA de `pointmap`, por tres caminos independientes")
    for nom in ("desde_X", "desde_Y", "desde_s"):
        if nom in f_det:
            v, d, nn = f_det[nom]
            print("    focal %-8s = %8.1f px   dispersion (p75-p25)/p50 = %.3f   n=%d  %s" % (
                nom, v, d, nn, "" if d < 0.25 else "<- FALLA el limite declarado de 0,25"))
    fz = f_det["frac_Z_negativa"][0]
    print("    Z NEGATIVA en el %.1f %% de las lecturas  %s" % (
        100 * fz, "" if fz < 0.01 else "<- una profundidad de un punto delante de la camara NO "
                                       "puede ser negativa"))
    if not np.isfinite(f_med):
        print("  sin focal medible: `pointmap` no da geometria de camara. Clip descartado.")
        continue

    filas, ks = fase_a(clip, lect, ks, mano, pxm, W, H, video, KP, SC, B)
    if filas is None:
        saltados.append(clip)
        continue
    print("  fase B se restringe a los %d cuadros donde SI se busco la hoja (%d-%d)" % (
        len(ks), ks[0], ks[-1]))
    cal = calibra(filas, KP, SC, B, W, H, pxm, mano)
    eje_clip = cal[0]["eje_ganador"] if cal[0]["dif_mediana"]["hombro"] else C3.EJE_POR_DEFECTO
    """El eje que se usa para los cuadros SIN hoja lo decide el careo del propio clip. Si no hay
    bastantes hojas vistas para decidir, se usa el hombro, que es la variable operativa medida."""
    print()
    print("  LAS DOS APERTURAS, medidas en este clip, y el techo de utilidad de `e3`")
    for lado in (0, 1):
        c = cal[lado]
        print("    lado %d: recortes %d  ·  con recta %d (%.1f %%)  ·  de ellas delante de la mano "
              "%d (%.1f %%)" % (
                  lado, c["n_recortes"], c["n_con_recta"],
                  100.0 * c["n_con_recta"] / max(c["n_recortes"], 1), c["n_delante"],
                  100.0 * c["frac_delante"]))
        print("             CAREO de ejes contra la hoja VISTA (azar declarado %.0f gr): "
              "|hoja-hombro| %s (n=%d)  ·  |hoja-codo| %s (n=%d)  ·  |hoja-pulgar| %s (n=%d)  ->  "
              "gana %s" % (
                  c["azar"],
                  "%.1f" % c["dif_mediana"]["hombro"] if c["dif_mediana"]["hombro"] else "sin n",
                  c["n_por_eje"]["hombro"],
                  "%.1f" % c["dif_mediana"]["codo"] if c["dif_mediana"]["codo"] else "sin n",
                  c["n_por_eje"]["codo"],
                  "%.1f" % c["dif_mediana"]["pulgar"] if c["dif_mediana"]["pulgar"] else "sin n",
                  c["n_por_eje"]["pulgar"], c["eje_ganador"].upper()))
        print("             apertura SUPUESTA (regla e6, p90 |hoja-eje ganador|) = %s  ·  apertura MEDIDA "
              "(p90 del temblor de la hoja, n=%d) = %s  ·  techo de utilidad = %.1f gr" % (
                  "%.1f gr" % c["apert_supuesta"] if c["apert_supuesta"] else "sin n",
                  c["n_pares_hoja"],
                  "%.1f gr" % c["apert_medida"] if c["apert_medida"] else "sin n", c["techo"]))
        for nom in ("apert_supuesta", "apert_medida"):
            if c[nom]:
                print("               %-15s -> %s el techo (%.2fx)" % (
                    nom, "CABE bajo" if c[nom] <= c["techo"] else "PASA de", c[nom] / c["techo"]))

    """EL CONTROL QUE DECIDE, y es de CPU y por tanto gratis: la z PERMUTADA entre cuadros.

    Si al mezclar al azar las profundidades entre cuadros el numero de discrepancias 2D/3D se queda
    igual, entonces esas discrepancias **no las produce la profundidad de la escena, las produce el
    ruido de `pointmap`**, y el cono 3D no esta midiendo nada. Es el mismo tipo de control que el
    proyecto ya uso con el cono girado en `e5` y con el senuelo de SAM2.

    Se permutan solo las Z, dejando (x, y) y todo lo demas en su sitio: asi el control cambia
    EXACTAMENTE la variable que se esta juzgando y nada mas.

    LIMITE DECLARADO ANTES DE MIRAR: el 3D solo tiene senal si las discrepancias con la z real son
    al menos **2 veces** las del control permutado. Por debajo de eso, lo que hay es ruido.
    """
    def permuta_z(lect, ks, rng):
        cp = {kl: {i: dict(d) for i, d in m.items()} for kl, m in lect.items()}
        for lado in (0, 1):
            kk = [k for k in ks if (k, lado) in cp]
            for i in set(i for k in kk for i in cp[(k, lado)]):
                dis = [k for k in kk if i in cp[(k, lado)]]
                vals = [lect[(k, lado)][i]["Z"] for k in dis]
                for k, v in zip(dis, rng.permutation(vals)):
                    cp[(k, lado)][i]["Z"] = float(v)
        return cp

    def compara(lc, guardar):
        O = {"vert3": np.full((T, 2, 3), np.nan), "eje3": np.full((T, 2, 3), np.nan),
             "eje2": np.full((T, 2, 2), np.nan), "apert": np.full((T, 2), np.nan),
             "L": np.full((T, 2), np.nan), "origen": np.full((T, 2), -1, dtype="int8"),
             "apert_origen": np.full((T, 2), -1, dtype="int8"),
             "am2": np.zeros((T, 2), dtype="int8"), "am3": np.zeros((T, 2), dtype="int8"),
             "d2_cm": np.full((T, 2), np.nan), "d3_cm": np.full((T, 2), np.nan),
             "ang2": np.full((T, 2), np.nan), "ang3": np.full((T, 2), np.nan),
             "elev": np.full((T, 2), np.nan), "dz_cm": np.full((T, 2), np.nan),
             "razon_Z": np.full((T, 2), np.nan), "w2": np.full((T, 2), np.nan),
             "w3": np.full((T, 2), np.nan)}
        cta = {"n": 0, "hoja": 0, "brazo": 0, "am2": 0, "am3": 0, "solo2": 0, "solo3": 0,
               "solo2_alcance": 0, "solo2_direccion": 0, "apert_medida": 0, "apert_e6": 0,
               "apert_e6_pool": 0}
        disc, resid = [], []
        for k in ks:
            for lado in (0, 1):
                fl = por.get((k, lado))
                ln = fl["linea"] if (fl and not fl["contaminado"]) else None
                g = C3.construye_cono(lc.get((k, lado), {}), lc.get((k, 1 - lado), {}),
                                      mano[lado], W, H, f_med, pxm[lado], ln,
                                      cal[lado]["apert_medida"], cal[lado]["apert_supuesta"],
                                      N_MALLA, eje_clip, caja_rival=B[k, 1 - lado])
                """SE PASA LA CAJA DEL RIVAL para que la puerta D2 -- la mano de un tirador dentro
                del cuerpo del otro -- se pruebe contra la caja y no contra la envolvente del
                torso. Medido sobre 1.684 cuadros-lado: con la envolvente salta el 0,9 % y con la
                caja el 2,4 %, o sea que sin pasarla la puerta filtraria un tercio de lo que
                debe."""
                if g is None:
                    continue
                O["vert3"][k, lado] = g["V"]
                O["eje3"][k, lado] = g["u3"]
                O["eje2"][k, lado] = g["u2"]
                for nom in ("apert", "L", "origen", "apert_origen", "d2_cm", "d3_cm",
                            "ang2", "ang3", "elev", "dz_cm", "razon_Z", "w2", "w3"):
                    O[nom][k, lado] = g[nom]
                O["am2"][k, lado] = int(g["am2"])
                O["am3"][k, lado] = int(g["am3"])
                """COMPROBACION POR PASO del eje 3D, con su cifra: la PROYECCION del eje 3D sobre
                el plano de la imagen tiene que devolver EXACTAMENTE la direccion medida de la hoja.
                Es la identidad que define la construccion, asi que su residuo mide si el codigo
                hace lo que dice el docstring, y no si el modelo acierta. Limite declarado: 1e-9."""
                pr2 = g["u3"][:2] / max(float(np.linalg.norm(g["u3"][:2])), 1e-12)
                resid.append(float(np.abs(pr2 - g["u2"]).max()))
                cta["n"] += 1
                cta["hoja" if g["origen"] == 0 else "brazo"] += 1
                cta[("apert_medida", "apert_e6", "apert_e6_pool")[g["apert_origen"]]] += 1
                cta["am2"] += int(g["am2"])
                cta["am3"] += int(g["am3"])
                if g["am2"] != g["am3"]:
                    caso = "solo2D" if g["am2"] else "solo3D"
                    mot = "alcance" if (g["am2"] and g["fuera_de_alcance_3d"]) else "direccion"
                    if g["am2"]:
                        cta["solo2"] += 1
                        cta["solo2_alcance" if mot == "alcance" else "solo2_direccion"] += 1
                    else:
                        cta["solo3"] += 1
                    if guardar:
                        disc.append({
                            "clip": clip, "cuadro": int(k), "lado": lado, "caso": caso,
                            "motivo": mot,
                            **{nom: g[nom] for nom in ("d2_cm", "d3_cm", "ang2", "ang3", "apert",
                                                       "origen", "apert_origen", "dz_cm", "elev",
                                                       "L_cm", "w2", "w3")}})
        cta["resid_proyeccion"] = float(max(resid)) if resid else float("nan")
        return O, cta, disc

    """FASE B, sin GPU: los conos, la interseccion y la comparacion."""
    por = {(f["cuadro"], f["lado"]): f for f in filas}

    O, cuenta, disc_clip = compara(lect, True)
    discrepantes.extend(disc_clip)
    ctl = [compara(permuta_z(lect, ks, np.random.default_rng(s)), False)[1] for s in range(5)]
    c = cuenta
    print()
    print("  LA COMPARACION 2D CONTRA 3D  (n = %d conos, de %d tiradores-cuadro de la ventana)" % (
        c["n"], 2 * len(ks)))
    if c["n"] == 0:
        print("    sin conos construibles. Clip descartado.")
        continue
    print("    COMPROBACION del eje 3D: |proyeccion(eje3D) - direccion medida| maximo = %.2e  ->  "
          "%s (limite declarado 1e-9)" % (
              c["resid_proyeccion"], "PASA" if c["resid_proyeccion"] < 1e-9 else "FALLA"))
    print("    origen del cono: hoja VISTA %d (%.1f %%)  ·  hoja SUPUESTA desde el brazo %d "
          "(%.1f %%)" % (c["hoja"], 100.0 * c["hoja"] / c["n"], c["brazo"],
                         100.0 * c["brazo"] / c["n"]))
    print("    origen de la APERTURA: temblor MEDIDO de este clip %d  ·  regla de `e6` en este clip "
          "%d  ·  regla de `e6` traida de OTROS clips %d  ->  las tres van separadas en el NPZ" % (
              c["apert_medida"], c["apert_e6"], c["apert_e6_pool"]))
    print("    amenaza en 2D: %d (%.1f %%)  ·  amenaza en 3D: %d (%.1f %%)" % (
        c["am2"], 100.0 * c["am2"] / c["n"], c["am3"], 100.0 * c["am3"] / c["n"]))
    print("    DISCREPAN: el 2D dice SI y el 3D dice NO -> %d (%.1f %%)   [por ALCANCE %d, por "
          "DIRECCION %d]" % (c["solo2"], 100.0 * c["solo2"] / c["n"], c["solo2_alcance"],
                             c["solo2_direccion"]))
    print("               el 3D dice SI y el 2D dice NO -> %d (%.1f %%)" % (
        c["solo3"], 100.0 * c["solo3"] / c["n"]))
    print("               total discrepancias %d de %d = %.1f %%" % (
        c["solo2"] + c["solo3"], c["n"], 100.0 * (c["solo2"] + c["solo3"]) / c["n"]))
    tot_real = c["solo2"] + c["solo3"]
    tot_ctl = float(np.median([x["solo2"] + x["solo3"] for x in ctl]))
    print("    CONTROL DE LA z PERMUTADA (5 permutaciones, mediana): %.0f discrepancias contra %d "
          "con la z real  ->  razon %.2fx  ·  %s (limite declarado 2x)" % (
              tot_ctl, tot_real, tot_real / max(tot_ctl, 1e-9),
              "el 3D lleva SENAL" if tot_real >= 2 * max(tot_ctl, 0.5) else
              "las discrepancias son RUIDO de `pointmap`, no profundidad de la escena"))
    v = np.isfinite(O["d2_cm"])
    print("    distancia mano -> torso mas cercano: en 2D mediana %.0f cm  ·  en 3D mediana %.0f cm"
          "  ·  |3D - 2D| mediana %.0f cm  p90 %.0f cm" % (
              float(np.nanmedian(O["d2_cm"][v])), float(np.nanmedian(O["d3_cm"][v])),
              float(np.nanmedian(np.abs(O["d3_cm"][v] - O["d2_cm"][v]))),
              float(np.nanpercentile(np.abs(O["d3_cm"][v] - O["d2_cm"][v]), 90))))
    rz = O["razon_Z"][v]
    print("    LINEALIZACION: razon Z_torso_rival / Z_mano_propia  mediana %.3f  p5 %.3f  p95 %.3f"
          "  ->  desviacion tipica de la conversion %.1f %%" % (
              float(np.nanmedian(rz)), float(np.nanpercentile(rz, 5)),
              float(np.nanpercentile(rz, 95)),
              100.0 * float(np.nanmedian(np.abs(rz - 1.0)))))
    print("    ELEVACION del eje 3D: mediana %+.1f gr  ·  p5 %+.1f  p95 %+.1f  ·  |dz| mano-torso "
          "mediana %.0f cm" % (
              float(np.nanmedian(O["elev"][v])), float(np.nanpercentile(O["elev"][v], 5)),
              float(np.nanpercentile(O["elev"][v], 95)),
              float(np.nanmedian(np.abs(O["dz_cm"][v])))))

    if HUMO:
        print("    PRUEBA DE HUMO: no se escribe ningun NPZ ni JSON")
        continue
    out = PRUEBAS / ("cono3d_%s%s.npz" % (clip, SUF))
    np.savez_compressed(out, ks=np.asarray(ks, "int32"), W=W, H=H, T=T, video=video,
                        mano=np.asarray([mano[0], mano[1]]), pxm=np.asarray([pxm[0], pxm[1]]),
                        focal=f_med, focal_disp=f_disp,
                        focal_det=np.asarray([[v for v in f_det[n]] for n in sorted(f_det)]),
                        focal_det_nom=np.asarray(sorted(f_det)),
                        apert_medida=np.asarray([cal[0]["apert_medida"] or np.nan,
                                                 cal[1]["apert_medida"] or np.nan]),
                        apert_supuesta=np.asarray([cal[0]["apert_supuesta"] or np.nan,
                                                   cal[1]["apert_supuesta"] or np.nan]),
                        techo=np.asarray([cal[0]["techo"], cal[1]["techo"]]),
                        leyenda_origen=np.asarray(["-1 sin cono", "0 hoja VISTA en el alfa",
                                                   "1 hoja SUPUESTA desde el brazo"]),
                        leyenda_apert_origen=np.asarray([
                            "-1 sin cono", "0 temblor MEDIDO de la hoja de este clip",
                            "1 regla de e6 (p90 |hoja-eje|) medida en ESTE clip",
                            "2 regla de e6 traida de OTROS clips (27,3 gr): en este clip no se vio "
                            "la hoja ni una vez"]),
                        **{kk: vv for kk, vv in O.items()})
    print("    cono guardado como dato -> %s" % out.name)

    """EL CONO **JUNTO A LAS POSES**, en un solo fichero, que es lo que pidio el usuario cuando dijo
    que quiere el cono «como dato».

    Y VA CON UNA LEYENDA DENTRO DEL PROPIO NPZ, no en un documento aparte. El motivo es la regla que
    el proyecto ya aplica a `origen` y a `tipo`: **un dato inventado que no se distingue de uno
    medido envenena el entrenamiento.** Aqui hay CUATRO columnas de procedencia y las cuatro son
    distintas, asi que las cuatro van:

        `tipo`         la CAJA: 0 medida, 1 interpolada, 2 sostenida, 3 redetectada
        `origen_pose`  la POSE: se hereda de `tipo` (la pose se mide sobre la caja que haya)
        `origen`       la DIRECCION del eje del cono: 0 hoja VISTA en el alfa, 1 SUPUESTA del brazo
        `apert_origen` la APERTURA del cono: 0 temblor medido de la hoja, 1 regla de `e6`

    Quien entrene con esto puede pesar distinto un cono cuya direccion se vio de uno cuya direccion
    se dedujo del antebrazo, que es una diferencia de 24-30 grados de apertura, o sea enorme. Sin
    estas columnas los dos serian el mismo numero.
    """
    comp = PRUEBAS / ("cono3d_completo_%s%s.npz" % (clip, SUF))
    np.savez_compressed(
        comp, kp=KP, sc=SC, B=B, tipo=tipo_pose, mask=mask_pose, presente=pres_pose,
        ks=np.asarray(ks, "int32"), W=W, H=H, T=T, video=video, fps=fps_pose,
        mano=np.asarray([mano[0], mano[1]]), pxm=np.asarray([pxm[0], pxm[1]]), focal=f_med,
        techo=np.asarray([cal[0]["techo"], cal[1]["techo"]]),
        leyenda=np.asarray([
            "kp (T,2,308,2) puntos de pose · sc (T,2,308) confianza · B (T,2,4) caja xyxy norm",
            "tipo (T,2): 0 caja medida · 1 interpolada · 2 sostenida · 3 redetectada · -1 nada",
            "mask (T,): cuadros utilizables · presente (T,2): tirador presente",
            "vert3 (T,2,3): vertice del cono en pixeles a la profundidad de referencia",
            "eje3 (T,2,3): eje 3D unitario del arma · eje2 (T,2,2): su proyeccion medida",
            "apert (T,2): semiapertura en grados · L (T,2): alcance de la hoja en pixeles",
            "origen (T,2): 0 direccion de HOJA VISTA en el alfa · 1 SUPUESTA del brazo · -1 sin cono",
            "apert_origen (T,2): 0 temblor MEDIDO en este clip · 1 regla de e6 en este clip · "
            "2 regla de e6 de OTROS clips (27,3 gr, la hoja no se vio aqui) · -1 sin cono",
            "am2 / am3 (T,2): amenaza al torso del rival en 2D y en 3D (0/1)",
            "d2_cm / d3_cm: distancia mano->torso mas cercano · ang2 / ang3: angulo al eje",
            "elev (T,2): elevacion del eje 3D en grados · dz_cm: profundidad mano->torso",
            "razon_Z (T,2): Z_torso_rival / Z_mano_propia segun pointmap",
            "AVISO: am3, elev, dz_cm y razon_Z dependen de la z de `pointmap`, que FALLA sus",
            "controles C4 y el careo de `e12`. No usar como profundidad fisica sin releer el",
            "apartado 22 de 04_documentacion/BITACORA_2026-09.md. am2, d2_cm, ang2, eje2, apert y origen NO dependen",
            "de `pointmap` y son los que se sostienen."]),
        **{kk: vv for kk, vv in O.items()})
    print("    cono JUNTO A LAS POSES, con leyenda dentro -> %s" % comp.name)
    resumen[clip] = {"cal": cal, "cuenta": c, "eje_usado": eje_clip, "control_z_permutada": tot_ctl,
                     "razon_senal_ruido": tot_real / max(tot_ctl, 1e-9),
                     "focal": f_med, "focal_disp": f_disp, "focal_det": f_det,
                     "n_ventana": len(ks),
                     "d2_med": float(np.nanmedian(O["d2_cm"][v])),
                     "d3_med": float(np.nanmedian(O["d3_cm"][v])),
                     "razon_Z_med": float(np.nanmedian(rz)),
                     "elev_med": float(np.nanmedian(O["elev"][v]))}
    print()

if not HUMO:
    (PRUEBAS / ("cono_e10_cono3d%s.json" % SUF)).write_text(
            json.dumps({"clips": resumen, "discrepantes": discrepantes, "saltados": saltados},
                   indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("=" * 104)
if saltados:
    print("CLIPS SALTADOS por no tener fase A cacheada: %d  ->  %s" % (
        len(saltados), ", ".join(saltados)))
print("TOTAL: %d discrepancias 2D/3D en %d clips  ->  %s" % (
    len(discrepantes), len(resumen), PRUEBAS / ("cono_e10_cono3d%s.json" % SUF)))
if discrepantes:
    from collections import Counter
    print("  por caso y motivo: %s" % dict(Counter(
        "%s/%s" % (d["caso"], d["motivo"]) for d in discrepantes)))
