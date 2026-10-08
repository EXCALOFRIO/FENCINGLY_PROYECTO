"""ETAPA E, paso 24: el campo de la hoja GUARDADO COMO DATO y PINTADO graduado.

`e23` midio la distribucion. Esto la deja en dos formas utilizables: un NPZ con el campo
**parametrizado por cuadro** (que es lo que le sirve a un modelo, no una imagen) y las figuras con el
relleno graduado por probabilidad (que es lo que el usuario quiere ver).

EL DATO. `cono_campo_<clip>.npz`, con la leyenda DENTRO del propio fichero, al lado de las cuatro
procedencias que ya existian. Tres columnas por cuadro y lado:

    campo_mu_gr     direccion central del campo, en grados de imagen. Si la hoja se VIO en ese
                    cuadro es el angulo MEDIDO de la hoja; si no, el del eje del antebrazo.
    campo_sigma_gr  SEMIANGULO del campo: 8,6 con hoja vista, 33,0 con direccion supuesta
    campo_origen    0 hoja VISTA · 1 direccion SUPUESTA del antebrazo
    temblor_hoja_gr 4,2, la incertidumbre de la DIRECCION de una hoja vista. NO es la anchura.

LO QUE HACE FALTA QUE NO SE CONFUNDA, y es el requisito del encargo: **un 1 medido y un 1 supuesto no
valen lo mismo**. Por eso `campo_origen` viaja pegado a `campo_mu_gr` y nunca se fusionan. Y la cifra
que lo cuantifica es **3,8x**: 33,0 contra 8,6, las dos el p90 de un residuo medido usado como
semiangulo, o sea la misma construccion. Dos versiones anteriores de esta cifra estan retiradas y se
dicen: los 14,0 grados de `e23` (8 pares de un solo asalto) y el **7,9x** (33,0/4,2), que comparaba
una anchura de dibujo con una incertidumbre de direccion. La derivacion de los 8,6 esta en
`cono3d_lib.APERT_HOJA_VISTA_SEMI_GR` y el control que la caza en `verifica_cono_campo.py`.

LA FIGURA. Dos paneles del mismo cuadro, con el mismo eje, el mismo L y la misma apertura, para que
**solo cambie la forma del relleno**:

    izquierda  `1 - angulo/apertura`, la forma ELEGIDA que el proyecto usa hoy
    derecha    la forma MEDIDA de `e23`, centrada en la hoja vista cuando la hay

Y el rotulo dice que la forma medida **no es la operativa** y por que: con n=31 solo el pico del
primer tramo sobrevive al ruido de Poisson. Un rotulo que vendiera la forma medida como establecida
seria peor que ninguno.

CPU. 4 fotogramas decodificados, los que tienen hoja vista. Cero GPU.
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
import e21_pinta_cono_arreglado as E21
import e23_campo_de_probabilidad as E23

import cv2

PRUEBAS = CL.PRUEBAS
SAL = E21.SAL
SIGMA_VISTA = C3.APERT_HOJA_VISTA_SEMI_GR
SIGMA_SUPUESTA = 33.0
TEMBLOR_VISTA = C3.TEMBLOR_HOJA_SEMI_GR
"""`SIGMA_VISTA` es la anchura del campo cuando la hoja se ha VISTO y `SIGMA_SUPUESTA` el p90 del
|desvio| de la direccion supuesta. Los dos son p90 de un residuo medido usados como semiangulo, o
sea la MISMA construccion, que es lo que los hace comparables. `TEMBLOR_VISTA` es otra cosa y se
guarda aparte.

CORREGIDO DOS VECES, Y LAS DOS CORRECCIONES SE QUEDAN ESCRITAS:

**2026-09-11, primera.** `e23` midio el temblor con 14,0 grados sobre **8 pares** de cuadros
consecutivos, todos del mismo asalto, y de ahi concluyo que ver la hoja «no estrecha el cono, solo lo
centra». `e27`, con **70 pares de 12 asaltos**, mide **4,2 grados** (p68 del cambio entre cuadros
6,0, dividido por raiz de 2). La conclusion de `e23` M3 quedo retirada: ver la hoja SI estrecha el
campo. Eso sigue en pie.

**2026-09-11, segunda, y era un error mio de CONVENIO.** Puse los 4,2 como anchura del campo, y 4,2
es la incertidumbre de la **direccion**, no la anchura de una cunia con el vertice en la **raiz de la
mano**. La recta de hoja detectada no pasa por la raiz de la mano -- pasa a 7 px de mediana, y el
segmento esta a `r/L = 0,24` del alcance -- asi que un campo de +-4,2 gr **no contiene la hoja que
dice contener**: solo 39 de 60 cuadros. Lo cazo `02_validacion/herramienta/verifica_cono_campo.py`,
que es el control que faltaba, y no un razonamiento.

    anchura publicada    +-8,6 gr   = p90 del residuo de la PUNTA con el centro en la hoja
    contiene la punta    54 de 60 (0,90), por definicion del p90
    razon contra 33,0    **3,8x**, no 7,9x

El 7,9x comparaba una anchura de dibujo con una incertidumbre de direccion. Ver la hoja sigue
estrechando el campo casi cuatro veces; lo que se retira es el factor exacto. La derivacion entera
esta en `cono3d_lib.APERT_HOJA_VISTA_SEMI_GR`, que es donde viven las otras dos aperturas.

El planteamiento del usuario («si lo hemos visto, lo marcamos como 1») era mas correcto que mi
medida: un cuadro con hoja vista si es casi puntual comparado con uno supuesto."""
LEYENDA = (
    "campo_mu_gr: direccion central del campo de la hoja, grados de imagen (atan2(dy,dx)). "
    "campo_sigma_gr: SEMIANGULO del campo, en grados desde campo_mu_gr, o sea que el campo completo "
    "abre el doble; 8,6 para origen 0 y 33,0 para origen 1, los dos p90 de un residuo medido. "
    "temblor_hoja_gr (4,2) NO es esa anchura: es la incertidumbre de la DIRECCION de una hoja vista "
    "entre cuadros consecutivos, y va aparte porque mide otra cosa. campo_origen: 0 hoja VISTA, "
    "1 direccion SUPUESTA del antebrazo `codo -> raiz`. campo_L_px: alcance, 1,10 m a escala. "
    "NaN = cuadro sin cono construible. La forma radial NO esta medida y se declara uniforme. "
    "La forma angular medida esta en perfil_bordes/perfil_valor y NO es operativa: con n=31 solo el "
    "primer tramo se distingue del ruido. La decision de amenaza sigue con cono_lib.peso_sector.")


def campo_por_clip(clip, hojas):
    d = E20.carga(clip)
    kp, sc, B, W, H, T = d["kp"], d["sc"], d["B"], d["W"], d["H"], d["T"]
    mu = np.full((T, 2), np.nan)
    sig = np.full((T, 2), np.nan)
    org = np.full((T, 2), -1, dtype="int8")
    Lpx = np.full((T, 2), np.nan)
    manos = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            return None
        manos[lado] = mano
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, "codo")
        _pk, pxm = C3.escala_px_m_de_pie(B, lado, H)
        ang_eje = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        for k in range(T):
            if not ok[k]:
                continue
            ln = hojas.get((clip, k, lado))
            Lpx[k, lado] = C3.ALCANCE_M * pxm
            if ln is None:
                mu[k, lado], sig[k, lado], org[k, lado] = ang_eje[k], SIGMA_SUPUESTA, 1
                continue
            """LA HOJA VISTA SE ORIENTA HACIA DELANTE sobre el eje antes de tomar su angulo: el
            segmento detectado no tiene sentido propio y un giro de 180 grados invertiria el campo."""
            p1 = np.asarray(ln["p1"], "float64")
            p2 = np.asarray(ln["p2"], "float64")
            dd = p2 - p1 if float((p2 - p1) @ u[k]) >= 0 else p1 - p2
            mu[k, lado] = float(np.degrees(np.arctan2(dd[1], dd[0])))
            sig[k, lado], org[k, lado] = SIGMA_VISTA, 0
    f = PRUEBAS / ("cono_campo_%s.npz" % clip)
    np.savez_compressed(
        f, campo_mu_gr=mu, campo_sigma_gr=sig, campo_origen=org, campo_L_px=Lpx,
        perfil_bordes=np.asarray(C3.PERFIL_MEDIDO_BORDES),
        perfil_valor=np.asarray(C3.PERFIL_MEDIDO_VALOR),
        apert_envolvente_semi_gr=C3.APERT_ENVOLVENTE_SEMI_GR,
        apert_nucleo_semi_gr=C3.APERT_NUCLEO_SEMI_GR,
        sigma_vista_gr=SIGMA_VISTA, sigma_supuesta_gr=SIGMA_SUPUESTA,
        temblor_hoja_gr=TEMBLOR_VISTA,
        alcance_m=C3.ALCANCE_M, eje="codo -> raiz (antebrazo)",
        mano_lado0=manos[0], mano_lado1=manos[1], leyenda=LEYENDA)
    n_vis = int((org == 0).sum())
    n_sup = int((org == 1).sum())
    return {"fichero": f.name, "n_vista": n_vis, "n_supuesta": n_sup,
            "frac_vista": n_vis / max(n_vis + n_sup, 1)}


def rellena_campo(img, vert, mu_gr, L, apert, col, perfil, alfa_max=0.62):
    x0 = max(0, int(vert[0] - L - 2))
    x1 = min(img.shape[1], int(vert[0] + L + 2))
    y0 = max(0, int(vert[1] - L - 2))
    y1 = min(img.shape[0], int(vert[1] + L + 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    GY, GX = np.mgrid[y0:y1, x0:x1]
    P = np.stack([GX.ravel() + 0.5, GY.ravel() + 0.5], axis=1).astype("float64")
    w = C3.campo_hoja(P, vert, mu_gr, L, apert, perfil)
    a = (alfa_max * w).reshape(y1 - y0, x1 - x0)[:, :, None]
    sub = img[y0:y1, x0:x1].astype("float32")
    img[y0:y1, x0:x1] = np.clip(sub * (1.0 - a) + np.asarray(col, "float32") * a,
                                0, 255).astype("uint8")


def panel(im, g, k, hojas, clip, perfil, tit, pie):
    v = im.copy()
    for lado in (0, 1):
        d = g[lado]
        L = C3.ALCANCE_M * d["pxm"]
        vert, u = d["vert"][k], d["u"][k]
        ln = hojas.get((clip, k, lado))
        if ln is None:
            mu = float(np.degrees(np.arctan2(u[1], u[0])))
        else:
            p1 = np.asarray(ln["p1"], "float64")
            p2 = np.asarray(ln["p2"], "float64")
            dd = p2 - p1 if float((p2 - p1) @ u) >= 0 else p1 - p2
            mu = float(np.degrees(np.arctan2(dd[1], dd[0])))
        mv = np.array([np.cos(np.radians(mu)), np.sin(np.radians(mu))])
        rellena_campo(v, vert, mu, L, C3.APERT_ENVOLVENTE_SEMI_GR, E21.COL[lado], perfil)
        E21.arco(v, vert, mv, L, C3.APERT_ENVOLVENTE_SEMI_GR, E21.COL[lado], 1, trazos=True)
        if np.isfinite(d["ancla"][k]).all():
            E21.trazos(v, d["ancla"][k], vert, E21.EJE, 1)
        cv2.circle(v, np.int32(vert), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(v, np.int32(vert), 4, (255, 255, 255), -1, cv2.LINE_AA)
        if ln is not None:
            p1, p2 = np.int32(ln["p1"]), np.int32(ln["p2"])
            cv2.line(v, tuple(p1), tuple(p2), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(v, tuple(p1), tuple(p2), E21.HOJA, 3, cv2.LINE_AA)
            tx = (int(p1[0]), int(p1[1]) + 26) if p1[1] < p2[1] else (int(p2[0]), int(p2[1]) + 26)
            for gr, cc in ((3, (0, 0, 0)), (1, E21.HOJA)):
                cv2.putText(v, "BLADE SEEN s%d" % lado, tx,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, gr, cv2.LINE_AA)
    rx = E21.recorta(g, k, C3.ALCANCE_M, C3.APERT_ENVOLVENTE_SEMI_GR, g["W"], g["H"])
    return E21.celda(v[rx[1]:rx[3], rx[0]:rx[2]], tit, pie, 880, 620)


def main():
    hojas = E21.hojas_cacheadas()
    print("=" * 108)
    print("EL CAMPO DE LA HOJA, COMO DATO Y PINTADO  ·  sin GPU")
    print("=" * 108)
    print("  anchura del campo con hoja VISTA +-%.1f gr (p90 del residuo de la punta, 60 cuadros de "
          "3 clips)" % SIGMA_VISTA)
    print("  anchura con direccion SUPUESTA  +-%.1f gr (p90 de |hoja - antebrazo|)  ->  razon "
          "**%.2fx**" % (SIGMA_SUPUESTA, SIGMA_SUPUESTA / SIGMA_VISTA))
    print("  o sea que ver la hoja estrecha el campo **%.1f veces**, y ademas lo centra."
          % (SIGMA_SUPUESTA / SIGMA_VISTA))
    print("  aparte, el TEMBLOR de la direccion de una hoja vista es %.1f gr (70 pares, 12 asaltos):"
          % TEMBLOR_VISTA)
    print("  es la incertidumbre de la DIRECCION y NO la anchura del dibujo. El 7,9x que se publico")
    print("  comparaba las dos cosas y esta RETIRADO; el 14,0 gr de `e23` (8 pares) tambien.")
    print()
    datos = {}
    for clip in E20.clips_cono():
        r = campo_por_clip(clip, hojas)
        if r is None:
            print("  %-32s sin mano armada decidible" % clip[:32])
            continue
        datos[clip] = r
        print("  %-32s %s  ·  conos con hoja VISTA %d de %d (%.1f %%)" % (
            clip[:32], r["fichero"], r["n_vista"], r["n_vista"] + r["n_supuesta"],
            100 * r["frac_vista"]))
    tot_v = sum(r["n_vista"] for r in datos.values())
    tot_n = sum(r["n_vista"] + r["n_supuesta"] for r in datos.values())
    cru = [r for r in E23.crudas() if not r["contaminado"]]
    n_int = len(cru)
    n_rec = len([1 for r in cru if r["linea"]])
    print()
    print("  TRES DENOMINADORES DISTINTOS, y confundirlos daria tres cifras contradictorias:")
    print("    hoja vista / conos del dato       %4d de %4d = **%.1f %%**  <- lo que hay en el NPZ" % (
        tot_v, tot_n, 100 * tot_v / max(tot_n, 1)))
    print("    hoja vista / recortes INTENTADOS  %4d de %4d = **%.1f %%**  <- la tasa del detector" % (
        n_rec, n_int, 100.0 * n_rec / max(n_int, 1)))
    print("    y en la ventana de accion, `e10` midio 3,0 %% (6 de 198)")
    print("  El %.1f %% del NPZ NO es la tasa del detector: su denominador incluye los conos donde la"
          % (100 * tot_v / max(tot_n, 1)))
    print("  deteccion **no se intento nunca** (solo se paso `matting` en %d recortes de %d conos)."
          % (n_int, tot_n))
    print("  La cifra honrada para «cuanto se ve el florete» es la del detector; la del NPZ es")
    print("  «cuantos conos del dato de hoy llevan direccion medida», y es la que limita el campo.")

    print()
    figs = []
    for clip, k in (("AdorableFittingAsiansmallclawedotter", 64),
                    ("AromaticShockedGermanspitz", 211),
                    ("ImpeccablePertinentBasilisk", 175),
                    ("WelloffForcefulIndri", 199)):
        g, d = E21.geometria(clip, "codo")
        if g is None:
            continue
        g["W"], g["H"] = d["W"], d["H"]
        cap = cv2.VideoCapture(d["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        okf, im = cap.read()
        cap.release()
        if not okf:
            continue
        a = panel(im, g, k, hojas, clip, "lineal",
                  "CHOSEN SHAPE | 1 - angle/half-angle | %s | frame %d" % (clip, k),
                  ["what the project uses today, and its own docstring already said it is a CHOSEN",
                   "SHAPE and not a measured probability: a linear ramp away from the axis"])
        b = panel(im, g, k, hojas, clip, None,
                  "MEASURED SHAPE | the user's idea | %s | frame %d" % (clip, k),
                  ["from the SEEN blades: counts 9/4/4/6/4/2 per 6 deg bin, normalised to the max",
                   "a peak hugging the axis and then almost flat | error vs the linear ramp 0.21 "
                   "on a 0-to-1 scale",
                   "NOT OPERATIVE: with n=31 only the first bin rises above Poisson noise"])
        fig = np.hstack([a, b])
        bl = np.zeros((56, fig.shape[1], 3), np.uint8)
        bl[:] = (14, 14, 20)
        cv2.putText(bl, "CONE PROBABILITY FIELD | same axis, same L = 1.10 m wrist-to-tip reach, "
                        "same envelope +-33.0 deg (66.0 deg full cone): the ONLY thing that changes "
                        "is the fill shape",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(bl, "where the blade was SEEN the field is centred on it (origen 0); where it "
                        "was not, on the forearm (origen 1, ASSUMED in 97 % of all cones). A "
                        "measured 1 and an assumed 1 are not worth the same.",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 220, 255), 1, cv2.LINE_AA)
        p = SAL / ("campo_%s_c%04d.png" % (clip, k))
        cv2.imwrite(str(p), np.vstack([fig, bl]))
        figs.append(p.name)
        print("  figura %s" % p.name)

    (PRUEBAS / "cono_e24_campo_dato.json").write_text(json.dumps(
        {"sigma_vista_gr": SIGMA_VISTA, "sigma_supuesta_gr": SIGMA_SUPUESTA,
         "temblor_hoja_gr": TEMBLOR_VISTA,
         "razon_sigmas": SIGMA_SUPUESTA / SIGMA_VISTA, "leyenda": LEYENDA,
         "npz": datos, "total_vista": tot_v, "total_conos": tot_n, "recortes_intentados": n_int, "rectas_detectadas": n_rec,
         "frac_vista": tot_v / max(tot_n, 1), "figuras": figs},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("guardado %s" % (PRUEBAS / "cono_e24_campo_dato.json"))


if __name__ == "__main__":
    main()
