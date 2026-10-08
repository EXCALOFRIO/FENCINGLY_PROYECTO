"""ETAPA E, paso 9: QUE 3D SE PUEDE TENER DE VERDAD, Y CON QUE ERROR. La materia prima, medida.

El usuario lo pidio asi: *"hay que ver que sea 3D, porque si no no tiene sentido en 2D. Asi que a ver
como lo haces con la profundidad"*. Su razonamiento es correcto: dos floretes que se cruzan en la
imagen pueden estar a un metro en profundidad, asi que una interseccion medida en 2D puede ser falsa.
Pero antes de construir ningun cono 3D hay que **medir la materia prima**, porque `pointmap` ya
fallo un control en este proyecto y no se puede dar por bueno.

POR QUE ESTO PUEDE FUNCIONAR AUNQUE `pointmap` YA FALLARA SU CONTROL. Fallo comparando la z ENTRE
PERSONAS distintas para decidir quien esta en la pista: eso pide escala global y la profundidad
monocular es ambigua de escala. Aqui se pide otra cosa: la **direccion** de un segmento DENTRO de una
sola persona. Un error de escala GLOBAL no cambia una direccion, porque se cancela al normalizar.

EL PELIGRO REAL ES OTRO y es lo que se mide aqui: si el error de `pointmap` no es una escala global
sino un **sesgo por pixel o una deformacion**, entonces si corrompe la direccion. Nada se construye
encima hasta que esto tenga un numero.

DOS MANERAS DE SACAR LA z, LAS DOS MEDIDAS, porque hay una trampa ya documentada en el proyecto:
*«la z de `pointmap` es inestable ENTRE RECORTES: ampliando el recorte un 30 % cambia hasta un
694 %»*.

  (A) `pointmap` del **CUADRO COMPLETO**, una pasada por cuadro. El recorte es identico en todos los
      cuadros, asi que lo que cambie entre cuadros es inestabilidad DEL MODELO y no mia. Es la buena
      para medir estabilidad temporal, y es la unica que puede dar z comparable entre los DOS
      tiradores, que es lo que el cono 3D necesita. Precio: con 1920x1080 entrando en un lienzo de
      768x1024 el contenido queda a 0,4x, o sea que el antebrazo pasa de 94 px a unos 38.
  (B) `pointmap` del **RECORTE DEL TIRADOR**. Mucha mas resolucion en la mano, pero el recorte cambia
      de tamano conforme el tirador se mueve, y por la trampa de arriba eso mete su propia deriva de
      escala. Se mide en 2 de los 5 clips para CONFIRMAR un hecho ya medido, no para descubrirlo, y
      por eso no se paga en los cinco.

--------------------------------------------------------------------------------------------------
LOS CINCO CRITERIOS, DECLARADOS AQUI ANTES DE MIRAR NINGUN NUMERO

  C1 ESTABILIDAD DE LA DIRECCION. El salto mediano de la direccion 3D entre cuadros consecutivos
     tiene que ser **<= 3 veces el salto mediano en 2D medido sobre los mismos cuadros**. La
     direccion 3D lleva la 2D como proyeccion, asi que no puede salir mas estable que ella; 3x es la
     tolerancia honesta por anadir una coordenada. Con el 2D en 0,9-2,1 grados eso son unos 6 grados.

  C2 QUE LA COORDENADA NUEVA LLEVE SENAL. Lo UNICO que el 3D anade sobre el 2D es el angulo de
     ELEVACION, o sea cuanto sale el brazo del plano de la imagen. Se exige
     **temblor / recorrido < 0,5**, con temblor = mediana del salto de la elevacion entre cuadros y
     recorrido = p95 menos p5 de la elevacion en el clip. Si el ruido es del tamano de la senal, la
     tercera coordenada es ruido disfrazado de profundidad.

  C3 RIGIDEZ, y este es el criterio que de verdad contesta a la pregunta del escorzo. El antebrazo
     tiene **largo 3D constante**: es un hueso. Su proyeccion en la imagen, en cambio, se acorta
     cuando el brazo apunta a la camara. Asi que si la z de `pointmap` es profundidad de verdad,
     anadirla tiene que **estabilizar** el largo del segmento. Se exige
     **cv(largo 3D) < cv(largo 2D)**, con cv el coeficiente de variacion robusto (MAD normalizada
     sobre mediana). Es el criterio mas exigente de los cinco y no necesita verdad-terreno: la
     rigidez del hueso hace de patron. Si falla, la z no es escorzo.

  C4 COHERENCIA DE CAMARA, control gratis. La conversion de unidades usa `s = Z / f` con `f` la
     focal del lienzo de entrada, que es FIJO (1024x768). Asi que `f = Z / s` tiene que salir
     parecido en todos los cuadros y en todos los clips. Se exige **dispersion relativa de `f`
     (p75-p25 sobre mediana) < 0,25** y **`s_col / s_fila` entre 0,8 y 1,25** (pixel cuadrado). Si
     esto falla, `pointmap` no esta produciendo una geometria de camara coherente y pasar su z a
     pixeles no significa nada.

  C5 RUIDO DE LA z CON EL BRAZO QUIETO, que es la pregunta literal del usuario. Se seleccionan los
     pares de cuadros consecutivos donde el eje 2D del brazo se movio **menos de 1 grado** (o sea,
     el brazo esta quieto) y se mide cuanto cambia la z de la mano entre esos dos cuadros, en
     **centimetros**, usando la escala px/m ya medida en `cono_lib.escala_px_m`.

     OJO, BUG CORREGIDO Y VALE COMO ESCARMIENTO: la primera version media
     `(Z_mano - Z_vertice) / s`, y **el vertice ES la mano**, asi que la resta valia CERO por
     construccion. C5 dio 0,0 cm en las 10 celdas y «paso» 10 de 10 sin medir nada. Un criterio que
     sale exactamente 0,000 en todas las celdas no es un exito: es un aviso de que no esta midiendo.
     Ahora se usa la Z **absoluta** de la mano, que es lo que cambia entre cuadros. Limite declarado:
     **por debajo de +-17 cm**, que es la MISMA vara que el proyecto ya usa como techo de utilidad
     del cono (la anchura de hombros del rival a 90 cm de distancia). Por encima de eso la
     profundidad no puede decidir si la punta llega al torso, porque su ruido es mayor que el blanco.

  Y LA PREGUNTA DEL SENTIDO, con su control del azar: cuando el brazo esta extendido HACIA LA CAMARA
  su proyeccion se acorta. Se mide la correlacion de Spearman entre el escorzo
  (`1 - largo2d / p95(largo2d)`) y `|z_mano - z_hombro|`. Si la z lleva escorzo, esa correlacion es
  POSITIVA. Y se compara con el control declarado: la misma correlacion con la z **permutada al
  azar** entre cuadros, que tiene que dar ~0.

  DIAGNOSTICO, el mismo que ya se uso en 2D: razon del salto a lag 2 sobre lag 1. Con ruido blanco
  vale ~1 y con movimiento suave ~2.

SI C1, C3 O C5 FALLAN, SE PARA Y SE REPORTA. No se construye cono 3D encima de una z que no se
sostiene: es lo que ya se hizo con el eje del pulgar en `e2` y con `pointmap` como puerta de pista.

--------------------------------------------------------------------------------------------------
POR QUE ESTE FICHERO SE REESCRIBIO: NO PODIA CORRER

La version anterior importaba `ultralytics` **y** `capas_limpias` (que importa `transformers`) en el
mismo proceso, y en esta maquina los dos interpretes NO coexisten:

    .venv-gpu  -> ultralytics  ·  sin transformers
    cuda/.venv -> transformers, torch, cv2  ·  **sin ultralytics**

Comprobado con `-c "import X"` en los dos. Asi que la deteccion no se puede hacer aqui. Se usa el
CACHE de detecciones del proyecto (`det_<CLIP>.pkl`, 107 clips), que cuesta 0 s, y la redeteccion en
recorte queda APAGADA (`yolo=None`), lo que baja la presencia y se informa por clip en vez de
esconderlo.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import capas_limpias as CL2
import cono3d_lib as C3
import cono_lib as CL

import cv2

PRUEBAS = CL.PRUEBAS
CLIPS = ["AromaticShockedGermanspitz", "ImpeccablePertinentBasilisk", "WelloffForcefulIndri",
         "SeparateHoarseFeline", "IdealFrigidJerboa"]
"""Cinco clips de FLORETE del corpus etiquetado, todos a 1080p y con detecciones cacheadas. Los tres
filtros y su motivo estan en `cono3d_lib.clips_1080_del_corpus`. Los dos clips con los que se midio
`e2`-`e8` (`SentimentalLarge...` y `AdorableFitting...`) **NO estan en el corpus etiquetado**, asi que
no se reutilizan aqui: podrian ser de otra arma."""
MODO = sys.argv[1] if len(sys.argv) > 1 else "quieta"
SUF = "" if MODO == "quieta" else "_%s" % MODO
"""DOS VENTANAS, Y HACEN FALTA LAS DOS, porque las preguntas de este paso no viven en el mismo sitio.

  `quieta`  el tramo consecutivo mas largo con los dos tiradores presentes, centrado. Es el bueno
            para C1, C2, C3 y C5, que miden RUIDO entre cuadros contiguos: cuanto mas quieto este el
            brazo, mas limpiamente se separa el ruido del movimiento real.
  `accion`  la ventana consecutiva que MINIMIZA la distancia mediana de la mano armada al torso del
            rival, o sea donde los dos estan a distancia de tocarse.

  POR QUE HIZO FALTA LA SEGUNDA, y es un error mio corregido con su cifra: con la ventana `quieta`
  del primer clip la distancia lateral mano-torso da una mediana de **200 cm** y el 2D no dice
  «amenaza» en **0 de 200** conos. En un tramo asi la pregunta «2D y 3D discrepan?» no se puede ni
  plantear, porque las dos respuestas son «no». La ventana se elige por un criterio MEDIDO (la
  distancia minima) y no por la etiqueta de tocado, que para estos clips no esta a mano y que ademas
  meteria circularidad."""
CLIPS_VIA_B = 2 if MODO == "quieta" else 0
VENTANA = 100
LEE = [5, 6, 7, 8, 9, 10, 21, 42]
TOL_C1 = 3.0
TOL_C2 = 0.5
QUIETO_GR = 1.0
TOL_C5_CM = 17.0
TOL_F = 0.25
N_PTS = 308


def perfil(serie, ok, es3d):
    """Salto entre cuadros consecutivos y a lag 2. `serie` son vectores 3D o angulos en grados."""
    d1, d2 = [], []
    n = len(serie)
    for k in range(n - 1):
        if ok[k] and ok[k + 1]:
            d1.append(C3.ang3d(serie[k], serie[k + 1]) if es3d
                      else abs(CL.dif_ang(serie[k], serie[k + 1])))
    for k in range(n - 2):
        if ok[k] and ok[k + 2]:
            d2.append(C3.ang3d(serie[k], serie[k + 2]) if es3d
                      else abs(CL.dif_ang(serie[k], serie[k + 2])))
    d1 = np.array([v for v in d1 if np.isfinite(v)])
    d2 = np.array([v for v in d2 if np.isfinite(v)])
    if d1.size < 10:
        return None
    return {"n": int(d1.size), "med": float(np.median(d1)),
            "p90": float(np.percentile(d1, 90)), "p95": float(np.percentile(d1, 95)),
            "max": float(d1.max()), "f10": float((d1 > 10).mean()), "f40": float((d1 > 40).mean()),
            "lag2": float(np.median(d2) / np.median(d1)) if d2.size and np.median(d1) > 0
            else float("nan")}


def cv_robusto(v):
    """MAD normalizada sobre la mediana. Robusto y comparable entre 2D y 3D porque es adimensional."""
    v = np.asarray(v, dtype="float64")
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 10:
        return float("nan")
    m = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - m)) / max(m, 1e-9))


def spearman(a, b):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 15:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype("float64")
    rb = np.argsort(np.argsort(b[m])).astype("float64")
    return float(np.corrcoef(ra, rb)[0, 1])


capas = CL2.Capas(["pose", "pointmap"])
resumen = {}
rng = np.random.default_rng(0)

for ic, clip in enumerate(CLIPS):
    mp4 = C3.localiza(clip)
    if not mp4:
        print("no encuentro el video de %s" % clip)
        continue
    cajas, cuadros, W, H, T = C3.carga_det(clip)
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        okf, im = cap.read()
        if not okf:
            break
        frames.append(im)
    cap.release()
    T = min(T, len(frames))
    gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY) for f in frames[:T]]
    sel = CL2.elige_pareja_del_clip(cajas, cuadros, W, H, T, gris=gris)
    B, pres, mask, tipo = sel["B"], sel["pres"], sel["mask"], sel["tipo"]
    dos = np.array([bool(mask[k] and pres[k].sum() == 2) for k in range(T)])
    print("=" * 102)
    print("%s   %dx%d   %d cuadros   utilizables %d   los dos presentes %.1f %%   (redeteccion "
          "APAGADA: sin ultralytics en este interprete)" % (
              clip, W, H, T, int(mask.sum()),
              100 * float(dos[mask].mean()) if mask.any() else float("nan")))
    print("=" * 102)

    """Los TRAMOS consecutivos con los dos presentes. La ventana es CONSECUTIVA a proposito: C1, C2 y
    C5 miden saltos entre cuadros contiguos y una muestra salteada no puede contestarlas."""
    tramos, ini, cur = [], 0, 0
    for k in range(T + 1):
        if k < T and dos[k]:
            cur += 1
            if cur == 1:
                ini = k
        else:
            if cur >= 40:
                tramos.append((cur, ini))
            cur = 0
    if not tramos:
        print("  ningun tramo consecutivo de 40 cuadros con los dos. No se mide.")
        continue

    KP = np.full((T, 2, N_PTS, 2), np.nan, dtype="float32")
    SC = np.full((T, 2, N_PTS), np.nan, dtype="float32")
    npz = PRUEBAS / ("cono_poses_%s.npz" % clip)
    if npz.is_file():
        with np.load(npz, allow_pickle=True) as z:
            KP, SC = z["kp"], z["sc"]
        print("  poses: reutilizadas de %s" % npz.name)
    else:
        t0 = time.perf_counter()
        for k in range(T):
            cjs, lados = [], []
            for lado in (0, 1):
                b = B[k, lado]
                if np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]:
                    cjs.append(np.array([b[0] * W, b[1] * H, b[2] * W, b[3] * H]))
                    lados.append(lado)
            if not cjs:
                continue
            out = capas.pose(np.ascontiguousarray(frames[k][:, :, ::-1]), cjs)
            for j, lado in enumerate(lados):
                kp, sc = out[j]
                m = min(len(kp), N_PTS)
                KP[k, lado, :m] = kp[:m, :2]
                SC[k, lado, :m] = sc[:m]
        ms = 1000.0 * (time.perf_counter() - t0) / max(T, 1)
        np.savez_compressed(npz, res=np.asarray((384, 288)), kp=KP, sc=SC, B=B, presente=pres,
                            marca=(tipo == 1), tipo=tipo, mask=mask, W=W, H=H, T=T, fps=0.0,
                            video=mp4, ms_pose=ms)
        print("  poses: %.0f ms/cuadro  ->  %s" % (ms, npz.name))

    mano, pxm = {}, {}
    for lado in (0, 1):
        m, det = CL.mano_armada(KP, SC, B, W, H, lado)
        if m is None:
            print("  lado %d: sin mano armada decidible. Clip descartado." % lado)
            mano = None
            break
        mano[lado] = m
        _pk, pxm[lado] = CL.escala_px_m(B, lado, H)
        print("  lado %d: mano armada %s (margen %.0f px, (b') %s)  ·  escala %.1f px/m" % (
            lado, m, det.get("margen_x_px", 0.0),
            "corrobora" if det.get("corrobora_b2") else "NO corrobora", pxm[lado]))
    if mano is None:
        continue

    """Distancia de la mano armada al centro del torso del rival, en centimetros, por cuadro. Se
    toma la MENOR de los dos tiradores: basta con que uno de los dos este a distancia de tocar para
    que la pregunta del cono este viva en ese cuadro."""
    dcm = np.full((T, 2), np.nan)
    for lado in (0, 1):
        M = CL.MANOS[mano[lado]]
        r = 1 - lado
        for k in range(T):
            p = KP[k, lado, M["root"]]
            oh = KP[k, r, list(CL.HOMBROS)]
            oc = KP[k, r, list(C3.CADERAS)]
            if not (np.isfinite(p).all() and np.isfinite(oh).all() and np.isfinite(oc).all()):
                continue
            c = 0.5 * (oh.mean(axis=0) + oc.mean(axis=0))
            dcm[k, lado] = float(np.linalg.norm(p - c)) * 100.0 / pxm[lado]
    with np.errstate(all="ignore"):
        dmin = np.nanmin(dcm, axis=1)
    vivo = np.isfinite(dmin) & (dmin <= 100.0 * CL.HOJA_M)
    """`vivo` = el torso del rival esta dentro del alcance de la hoja (90 cm, FIE m.8.4). Es la
    definicion MEDIDA de «aqui la pregunta del cono tiene sentido», y no depende de ninguna
    etiqueta de tocado: usar la etiqueta para elegir donde medir seria circular."""

    if MODO == "accion":
        cands = []
        for largo, a in tramos:
            n = min(VENTANA, largo)
            for a0 in range(a, a + largo - n + 1):
                cands.append((-int(vivo[a0:a0 + n].sum()),
                              float(np.nanmedian(dmin[a0:a0 + n])), a0, n))
        cands.sort()
        nviv, dmed, a0, n = cands[0]
        print("  ventana ACCION: %d-%d (%d cuadros)  ·  %d cuadros con el torso del rival dentro de "
              "los 90 cm  ·  distancia mediana %.0f cm  (de %d ventanas, la peor tenia %d vivos)"
              % (a0, a0 + n - 1, n, -nviv, dmed, len(cands), -cands[-1][0]))
        """SE MAXIMIZA EL NUMERO DE CUADROS VIVOS, no se minimiza la mediana, y esto es un cambio de
        criterio con su motivo: la mediana la dominan los muchos cuadros lejanos de la preparacion
        (187-334 cm), asi que minimizarla mueve la ventana poco y puede dejar fuera el propio
        tocado. Lo que hace falta son cuadros donde el 2D pueda decir «amenaza», porque si el 2D
        nunca dice si, la comparacion 2D contra 3D no se puede ni plantear."""
    else:
        largo, a = max(tramos)
        n = min(VENTANA, largo)
        a0 = a + (largo - n) // 2
        print("  ventana QUIETA: %d-%d (%d cuadros)  ·  tramo consecutivo de %d  ·  distancia "
              "mediana mano->torso rival %.0f cm  ·  %d cuadros dentro de los 90 cm" % (
                  a0, a0 + n - 1, n, largo, float(np.nanmedian(dmin[a0:a0 + n])),
                  int(vivo[a0:a0 + n].sum())))
    ks = list(range(a0, a0 + n))
    print("  en TODO el clip hay %d cuadros (%.1f %%) con el rival dentro de los 90 cm" % (
        int(vivo.sum()), 100.0 * float(vivo.mean())))

    via_b = ic < CLIPS_VIA_B
    reg = {lado: {"k": [], "lect": [], "s": [], "Zv": [], "u3_ante": [], "u3_brazo": [],
                  "ok3": [], "ang2": [], "ok2": [], "l2_ante": [], "l3_ante": [],
                  "l2_brazo": [], "l3_brazo": [], "z_root_px": [], "z_hom_px": [],
                  "mad_codo": [], "mad_root": [], "f_focal": [], "s_ratio": [],
                  "z_root_abs": [],
                  "u3_ante_B": [], "ok3B": []} for lado in (0, 1)}
    Z3 = {}
    t0 = time.perf_counter()
    for k in ks:
        rgb = np.ascontiguousarray(frames[k][:, :, ::-1])
        pm = capas.pointmap(rgb)
        for lado in (0, 1):
            M = CL.MANOS[mano[lado]]
            alto = (B[k, lado, 3] - B[k, lado, 1]) * H
            lect = C3.lee_articulaciones(pm, KP[k], SC[k], lado, LEE, alto)
            r = reg[lado]
            r["k"].append(k)
            i_c, i_r, i_h = M["codo"], M["root"], M["hombro"]
            v2 = KP[k, lado, i_r] - KP[k, lado, i_c]
            ok2 = bool(np.isfinite(v2).all() and np.linalg.norm(v2) > 1
                       and SC[k, lado, i_c] >= CL.CONF_MIN and SC[k, lado, i_r] >= CL.CONF_MIN)
            r["ang2"].append(float(np.degrees(np.arctan2(v2[1], v2[0]))) if ok2 else np.nan)
            r["ok2"].append(ok2)
            r["l2_ante"].append(float(np.linalg.norm(v2)) if ok2 else np.nan)
            v2b = KP[k, lado, i_r] - KP[k, lado, i_h]
            r["l2_brazo"].append(float(np.linalg.norm(v2b))
                                 if np.isfinite(v2b).all() else np.nan)

            Zv = s0 = np.nan
            if i_r in lect:
                Zv, s0 = lect[i_r]["P"][2], lect[i_r]["s"]
                r["mad_root"].append(lect[i_r]["mad"])
                if np.isfinite(s0) and s0 > 0:
                    r["f_focal"].append(abs(Zv) / s0)
                    r["s_ratio"].append(lect[i_r]["s_col"] / max(lect[i_r]["s_fila"], 1e-12))
            if i_c in lect:
                r["mad_codo"].append(lect[i_c]["mad"])
            r["s"].append(s0)
            r["Zv"].append(Zv)
            r["z_root_abs"].append(Zv)
            """La Z ABSOLUTA de la mano. Es la que cambia entre cuadros; la relativa al vertice vale
            cero por construccion y fue el bug de C5."""
            Z3[(k, lado)] = (lect, float(Zv), float(s0))

            u3 = u3b = None
            l3a = l3b = np.nan
            if np.isfinite(s0) and s0 > 0:
                Pc = C3.a_pixeles(lect, i_c, Zv, s0)
                Pr = C3.a_pixeles(lect, i_r, Zv, s0)
                Ph = C3.a_pixeles(lect, i_h, Zv, s0)
                u3 = C3.dir3d(Pc, Pr)
                u3b = C3.dir3d(Ph, Pr)
                if Pc is not None and Pr is not None:
                    l3a = float(np.linalg.norm(Pr - Pc))
                if Ph is not None and Pr is not None:
                    l3b = float(np.linalg.norm(Pr - Ph))
                    r["z_root_px"].append(Pr[2])
                    r["z_hom_px"].append(Ph[2])
                else:
                    r["z_root_px"].append(np.nan)
                    r["z_hom_px"].append(np.nan)
            else:
                r["z_root_px"].append(np.nan)
                r["z_hom_px"].append(np.nan)
            r["u3_ante"].append(u3 if u3 is not None else np.full(3, np.nan))
            r["u3_brazo"].append(u3b if u3b is not None else np.full(3, np.nan))
            r["ok3"].append(bool(u3 is not None and ok2))
            r["l3_ante"].append(l3a)
            r["l3_brazo"].append(l3b)

            u3B = None
            if via_b:
                c = CL2.recorte(B, k, lado, W, H)
                if c is not None and (c[2] - c[0]) > 48 and (c[3] - c[1]) > 48:
                    pmb = capas.pointmap(rgb[c[1]:c[3], c[0]:c[2]])
                    lb = C3.lee_articulaciones(pmb, KP[k] - np.array([c[0], c[1]]), SC[k],
                                               lado, [i_c, i_r], alto)
                    if i_r in lb and np.isfinite(lb[i_r]["s"]) and lb[i_r]["s"] > 0:
                        Zb, sb = lb[i_r]["P"][2], lb[i_r]["s"]
                        u3B = C3.dir3d(C3.a_pixeles(lb, i_c, Zb, sb),
                                       C3.a_pixeles(lb, i_r, Zb, sb))
            r["u3_ante_B"].append(u3B if u3B is not None else np.full(3, np.nan))
            r["ok3B"].append(bool(u3B is not None and ok2))
    dt = time.perf_counter() - t0
    ms = 1000.0 * dt / max(n, 1)
    print("  %d cuadros x 2 tiradores  ·  %.1f s  ·  %.0f ms/cuadro%s  (GPU COMPARTIDA: indicativo)"
          % (n, dt, ms, "  incluye la via B" if via_b else ""))

    out = {"n": n, "modo": MODO, "ks": [int(ks[0]), int(ks[-1])],
           "n_vivos_ventana": int(vivo[a0:a0 + n].sum()),
           "n_vivos_clip": int(vivo.sum()), "frac_vivos_clip": float(vivo.mean()),
           "dist_mediana_cm": float(np.nanmedian(dmin[a0:a0 + n])), "W": W, "H": H, "via_b": bool(via_b),
           "ms_por_cuadro": ms, "pres_dos": float(dos[mask].mean()) if mask.any() else None,
           "lados": {}}
    print()
    print("  C4 COHERENCIA DE CAMARA (control gratis: `f = Z/s` tiene que ser casi constante)")
    for lado in (0, 1):
        r = reg[lado]
        ff = np.array([v for v in r["f_focal"] if np.isfinite(v)])
        sr = np.array([v for v in r["s_ratio"] if np.isfinite(v)])
        if ff.size < 10:
            print("    lado %d: sin lecturas suficientes" % lado)
            continue
        disp = float((np.percentile(ff, 75) - np.percentile(ff, 25)) / max(np.median(ff), 1e-9))
        pasa = bool(disp < TOL_F and 0.8 <= float(np.median(sr)) <= 1.25)
        print("    lado %d: f = Z/s  mediana %.0f px  ·  dispersion (p75-p25)/p50 = %.3f  ·  "
              "s_col/s_fila mediana %.3f  ->  %s" % (
                  lado, float(np.median(ff)), disp, float(np.median(sr)),
                  "PASA" if pasa else "FALLA"))
        print("             MAD de la z en el disco: codo %.4g  ·  root %.4g  (el root esta en el "
              "borde de la silueta y es el de riesgo)" % (
                  float(np.median(r["mad_codo"])) if r["mad_codo"] else float("nan"),
                  float(np.median(r["mad_root"])) if r["mad_root"] else float("nan")))

    print()
    print("  C1 ESTABILIDAD DE LA DIRECCION (limite declarado: 3D <= %.0f x 2D)" % TOL_C1)
    print("    %-10s %-5s %6s %7s %7s %7s %6s %6s %6s" % (
        "via", "lado", "n", "med", "p90", "max", ">10", ">40", "lag2"))
    for lado in (0, 1):
        r = reg[lado]
        p2 = perfil(r["ang2"], r["ok2"], False)
        if p2:
            print("    %-10s %-5d %6d %7.2f %7.2f %7.1f %6.3f %6.3f %6.2f" % (
                "2D ref", lado, p2["n"], p2["med"], p2["p90"], p2["max"], p2["f10"], p2["f40"],
                p2["lag2"]))
        for nom, key, okk in (("3D A cuadro", "u3_ante", "ok3"), ("3D B recorte", "u3_ante_B",
                                                                  "ok3B")):
            p3 = perfil(r[key], r[okk], True)
            if not p3:
                continue
            print("    %-10s %-5d %6d %7.2f %7.2f %7.1f %6.3f %6.3f %6.2f  ->  %.1fx el 2D  ·  %s"
                  % (nom, lado, p3["n"], p3["med"], p3["p90"], p3["max"], p3["f10"], p3["f40"],
                     p3["lag2"], p3["med"] / max(p2["med"], 1e-9) if p2 else float("nan"),
                     "PASA" if p2 and p3["med"] <= TOL_C1 * p2["med"] else "FALLA"))
        out["lados"][lado] = {"2D": p2, "3D_A": perfil(r["u3_ante"], r["ok3"], True),
                              "3D_B": perfil(r["u3_ante_B"], r["ok3B"], True),
                              "c1_pasa": bool(p2 and perfil(r["u3_ante"], r["ok3"], True)
                                              and perfil(r["u3_ante"], r["ok3"], True)["med"]
                                              <= TOL_C1 * p2["med"])}

    print()
    print("  C2 LA ELEVACION LLEVA SENAL? (limite declarado: temblor/recorrido < %.1f)" % TOL_C2)
    for lado in (0, 1):
        r = reg[lado]
        u = np.asarray(r["u3_ante"], dtype="float64")
        o = np.asarray(r["ok3"])
        if o.sum() < 20:
            continue
        elev = np.array([C3.elevacion(v) if np.isfinite(v).all() else np.nan for v in u])
        d1 = np.array([abs(elev[j + 1] - elev[j]) for j in range(len(elev) - 1)
                       if o[j] and o[j + 1] and np.isfinite(elev[j]) and np.isfinite(elev[j + 1])])
        ev = elev[o & np.isfinite(elev)]
        rec = float(np.percentile(ev, 95) - np.percentile(ev, 5)) if ev.size else float("nan")
        tem = float(np.median(d1)) if d1.size else float("nan")
        raz = tem / rec if rec > 0 else float("inf")
        print("    lado %d: elevacion mediana %+.1f gr  ·  recorrido p5-p95 %.1f gr  ·  temblor "
              "%.1f gr  ->  razon %.2f  %s" % (
                  lado, float(np.median(ev)) if ev.size else float("nan"), rec, tem, raz,
                  "PASA" if raz < TOL_C2 else "FALLA"))
        out["lados"].setdefault(lado, {})["C2"] = {"mediana": float(np.median(ev)) if ev.size
                                                   else None, "recorrido": rec, "temblor": tem,
                                                   "razon": raz, "pasa": bool(raz < TOL_C2)}

    print()
    print("  C3 RIGIDEZ: el hueso hace de patron. cv(largo 3D) tiene que ser < cv(largo 2D)")
    for lado in (0, 1):
        r = reg[lado]
        for nom, k2, k3 in (("antebrazo codo-root", "l2_ante", "l3_ante"),
                            ("brazo hombro-root", "l2_brazo", "l3_brazo")):
            c2 = cv_robusto(r[k2])
            c3 = cv_robusto(r[k3])
            print("    lado %d %-20s cv 2D %.3f  ·  cv 3D %.3f  ->  %s  (%.2fx)" % (
                lado, nom, c2, c3,
                "PASA: la z estabiliza" if np.isfinite(c3) and c3 < c2 else
                "FALLA: la z desestabiliza", c3 / max(c2, 1e-9)))
            out["lados"].setdefault(lado, {}).setdefault("C3", {})[nom] = {
                "cv2d": c2, "cv3d": c3, "pasa": bool(np.isfinite(c3) and c3 < c2)}

    print()
    print("  C5 RUIDO DE LA z CON EL BRAZO QUIETO (< %.0f grados de giro del eje 2D)  ·  limite "
          "declarado +-%.0f cm" % (QUIETO_GR, TOL_C5_CM))
    for lado in (0, 1):
        r = reg[lado]
        zr = np.asarray(r["z_root_abs"], dtype="float64")
        sr = np.asarray(r["s"], dtype="float64")
        a2 = np.asarray(r["ang2"], dtype="float64")
        o2 = np.asarray(r["ok2"])
        dz, dz_all = [], []
        for j in range(len(zr) - 1):
            if not (o2[j] and o2[j + 1] and np.isfinite(zr[j]) and np.isfinite(zr[j + 1])
                    and np.isfinite(sr[j]) and sr[j] > 0):
                continue
            d = abs(zr[j + 1] - zr[j]) / sr[j] * 100.0 / pxm[lado]
            dz_all.append(d)
            if abs(CL.dif_ang(a2[j + 1], a2[j])) < QUIETO_GR:
                dz.append(d)
        if len(dz) < 8:
            print("    lado %d: solo %d pares con el brazo quieto. n insuficiente." % (lado,
                                                                                       len(dz)))
            continue
        dz, dz_all = np.array(dz), np.array(dz_all)
        print("    lado %d: n=%d de %d pares  ·  |dz| mediana %.1f cm  p90 %.1f cm  max %.1f cm  "
              "->  %s" % (lado, len(dz), len(dz_all), np.median(dz), np.percentile(dz, 90),
                          dz.max(), "PASA" if np.median(dz) < TOL_C5_CM else "FALLA"))
        print("             (todos los pares, quieto o no: mediana %.1f cm  p90 %.1f cm)" % (
            np.median(dz_all), np.percentile(dz_all, 90)))
        out["lados"].setdefault(lado, {})["C5"] = {
            "n": len(dz), "n_total": len(dz_all), "med_cm": float(np.median(dz)),
            "p90_cm": float(np.percentile(dz, 90)), "max_cm": float(dz.max()),
            "med_cm_todos": float(np.median(dz_all)),
            "pasa": bool(np.median(dz) < TOL_C5_CM)}

    print()
    print("  EL SENTIDO DEL ESCORZO, con su control del azar")
    for lado in (0, 1):
        r = reg[lado]
        l2 = np.asarray(r["l2_brazo"], dtype="float64")
        dzh = np.abs(np.asarray(r["z_root_px"], dtype="float64")
                     - np.asarray(r["z_hom_px"], dtype="float64"))
        m = np.isfinite(l2) & np.isfinite(dzh)
        if m.sum() < 15:
            print("    lado %d: n insuficiente" % lado)
            continue
        esc = 1.0 - l2 / np.nanpercentile(l2[m], 95)
        rho = spearman(esc[m], dzh[m])
        ctl = float(np.median([spearman(esc[m], rng.permutation(dzh[m])) for _ in range(21)]))
        print("    lado %d: n=%d  ·  spearman(escorzo 2D, |z_mano - z_hombro|) = %+.3f  ·  control "
              "con la z permutada = %+.3f  ->  %s" % (
                  lado, int(m.sum()), rho, ctl,
                  "el sentido es el CORRECTO" if rho >= 0.30 else
                  "NO se ve el escorzo en la z"))
        out["lados"].setdefault(lado, {})["escorzo"] = {"rho": rho, "control": ctl,
                                                        "n": int(m.sum())}

    zk, zl, zP, zs, zZ = [], [], [], [], []
    for (k, lado), (lect, Zv, s0) in sorted(Z3.items()):
        for i, d in lect.items():
            zk.append(k)
            zl.append(lado)
            zP.append([i, d["P"][0], d["P"][1], d["P"][2], d["mad"], d["s"], d["xy"][0],
                       d["xy"][1]])
        zs.append([k, lado, s0, Zv])
    np.savez_compressed(PRUEBAS / ("cono_z3d_%s%s.npz" % (clip, SUF)),
                        k=np.asarray(zk, "int32"), lado=np.asarray(zl, "int8"),
                        dato=np.asarray(zP, "float64"), vert=np.asarray(zs, "float64"),
                        cols=np.asarray(["idx", "X", "Y", "Z", "mad", "s", "x_px", "y_px"]),
                        ks=np.asarray(ks, "int32"), mano=np.asarray([mano[0], mano[1]]),
                        pxm=np.asarray([pxm[0], pxm[1]]), W=W, H=H, T=T, video=mp4)
    print("  lecturas de z guardadas -> cono_z3d_%s%s.npz" % (clip, SUF))
    resumen[clip] = out
    print()

(PRUEBAS / ("cono_e9_eje3d%s.json" % SUF)).write_text(
    json.dumps(resumen, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("guardado %s" % (PRUEBAS / ("cono_e9_eje3d%s.json" % SUF)))
