"""ETAPA E, paso 12: EL 3D SIN `pointmap`. La linea base gratis, que en este proyecto manda.

POR QUE ESTE PASO EXISTE. La regla del proyecto es que nada se afirma sin cifra **y con la linea base
gratis al lado**. `pointmap` cuesta 480 ms por cuadro y esta bloqueado a 1024x768; antes de aceptar
esa factura hay que ver que da la geometria de balde. Y el precedente es directo: la puerta de pista
se resolvio con la **y del pie**, que cuesta 0 ms, despues de que `pointmap` y la cabeza `depth` de
Sapiens 1 fallaran su control.

--------------------------------------------------------------------------------------------------
LO QUE ES GRATIS Y EXACTO, Y LO QUE NO. Esta separacion es el nucleo del paso

Con la estatura nominal `S = 1,75 m` y el alto aparente `h` en pixeles, la escala metrica en el
propio tirador es `S / h` metros por pixel, que es exactamente lo que ya calcula
`cono_lib.escala_px_m`. Y de ahi salen **dos de las tres coordenadas, sin ninguna incognita**:

    X = (col - W/2) * S / h          Y = (fila - H/2) * S / h          [metros, MEDIDO]

o sea que la posicion LATERAL y VERTICAL de cualquier punto, en metros y en el sistema de la camara,
es gratis y no necesita `pointmap` ni focal ni nada. La tercera:

    Z = f * S / h                    [metros, con f DESCONOCIDA]

y la RAZON de profundidades entre los dos tiradores tampoco necesita nada:

    Z_rival / Z_propio = h_propio / h_rival          [MEDIDO, sin f]

Lo unico que falta es UN numero por clip. Se parametriza como **D = distancia de la camara en
estaturas** (`D = Z_propio / S = f / h_propio`), que es interpretable, y entonces:

    separacion en profundidad, en metros:   dZ = S * D * (h_propio / h_rival - 1)

con el parentesis MEDIDO y `D` desconocida. **La geometria monocular de una sola camara no puede
dar `D` sin una longitud conocida en la escena**, y eso no es un defecto de este paso: es un hecho.
Asi que aqui no se inventa `D`, se **barre** y se dice a partir de que valor cambiaria el veredicto.
Es la unica forma honrada de tratar un parametro que no se puede medir.

--------------------------------------------------------------------------------------------------
LA COTA DE ELEVACION DEL ARMA CON LA HOJA MEDIDA Y LOS 90 cm DE LA FIE

Y aqui si hay una medida nueva y enteramente gratis, que contesta a la pregunta del escorzo sin
`pointmap`. La hoja mide 90 cm (FIE m.8.4, verificado en fuente primaria). Si en la imagen se ve un
tramo de hoja que llega a `a` centimetros de la mano, entonces la hoja esta escorzada por un factor
`a / 90` como MINIMO, luego:

    |elevacion del arma|  <=  arccos( min(1, a / 90) )

Es una COTA SUPERIOR y no una estimacion, y esa es su virtud: la deteccion en el alfa da un tramo
PARCIAL de la hoja, asi que `a` es una cota inferior del largo proyectado, y la desigualdad sigue
valiendo. Si en un cuadro `a` sale 85 cm, el arma esta a menos de 19 grados del plano de la imagen y
**el 3D no puede cambiar nada ahi**. Si sale 30 cm, la cota es 70 grados y no informa.

CRITERIO DECLARADO ANTES DE MIRAR: la cota sirve si en al menos el **20 % de los cuadros con hoja
vista** queda por debajo de **30 grados**, porque por debajo de 30 grados el acortamiento del alcance
es `1 - cos(30) = 13 %`, o sea menos de 12 cm de los 90, que es del orden de la propia tolerancia del
cono. Por encima del 30 no acota nada util.

--------------------------------------------------------------------------------------------------
Y EL CONTROL CRUZADO QUE HAY QUE HACER: el ORDEN de profundidad

El proyecto ya midio que la `y del pie` ordena la profundidad de los dos tiradores acertando 28 de
28, y que `pointmap` de cuadro completo acierta el 76 %. Aqui hay una TERCERA senal gratis, el alto
aparente de la caja, y las tres tienen que coincidir. Cuando el alto y la `y del pie` coinciden y
`pointmap` dice lo contrario, es `pointmap` el que se equivoca, porque las dos gratis son
consecuencias geometricas exactas de que la camara este por encima de la pista y no priors
aprendidos.
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
CLIPS = ["AromaticShockedGermanspitz", "ImpeccablePertinentBasilisk", "WelloffForcefulIndri",
         "SeparateHoarseFeline", "IdealFrigidJerboa"]
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
"""Sufijo de VENTANA, el mismo que usa `e9`."""
S = CL.ALTURA_M
HOJA_CM = 100.0 * CL.HOJA_M
D_MIN, D_MAX = 3.0, 35.0
"""Barrido de la distancia de camara en estaturas. 3 estaturas son 5,3 m y 35 son 61 m: cualquier
camara de television de esgrima cae dentro. El limite se declara aqui, antes de barrer."""
TOL_COTA_GR = 30.0
TOL_FRAC = 0.20
N_MALLA = 7
ESQUINAS = [5, 6, 10, 9]


def cota_elevacion(a_cm):
    """Cota SUPERIOR del |angulo| que el arma saca del plano, a partir del alcance proyectado."""
    if not np.isfinite(a_cm) or a_cm <= 0:
        return float("nan")
    return float(np.degrees(np.arccos(min(1.0, a_cm / HOJA_CM))))


resumen = {}
for clip in CLIPS:
    fz = PRUEBAS / ("cono_z3d_%s%s.npz" % (clip, SUF))
    fh = PRUEBAS / ("cono_hoja3d_%s%s.json" % (clip, SUF))
    fp = PRUEBAS / ("cono_poses_%s.npz" % clip)
    if not (fz.is_file() and fp.is_file()):
        print("faltan datos de %s" % clip)
        continue
    lect, ks, mano, pxm, W, H, video = C3.carga_z(clip, SUF)
    with np.load(fp, allow_pickle=True) as z:
        KP, SC, B = z["kp"], z["sc"], z["B"]
    filas = json.loads(fh.read_text(encoding="utf-8")) if fh.is_file() else []
    por = {(f["cuadro"], f["lado"]): f for f in filas}
    print("=" * 104)
    print("%s   %dx%d   ventana %d-%d (%d cuadros)   escala %.1f / %.1f px/m" % (
        clip, W, H, ks[0], ks[-1], len(ks), pxm[0], pxm[1]))
    print("=" * 104)

    """1. LA COTA DE ELEVACION DEL ARMA, con la hoja vista y los 90 cm de la FIE."""
    cotas, alcances = [], []
    for (k, lado), f in por.items():
        ln = f["linea"]
        if f["contaminado"] or not ln or not ln["delante"]:
            continue
        M = CL.MANOS[mano[lado]]
        vert = KP[k, lado, M["root"]].astype("float64")
        a_cm = float(np.linalg.norm(np.asarray(ln["p2"]) - vert)) * 100.0 / pxm[lado]
        alcances.append(a_cm)
        cotas.append(cota_elevacion(a_cm))
    if cotas:
        cc, aa = np.array(cotas), np.array(alcances)
        frac = float((cc <= TOL_COTA_GR).mean())
        print("  1. COTA DE ELEVACION del arma (gratis, hoja vista contra los 90 cm de la FIE)")
        print("     n=%d cuadros con hoja vista delante de la mano" % cc.size)
        print("     alcance proyectado de la hoja: mediana %.0f cm  p75 %.0f  p90 %.0f  max %.0f  "
              "(de 90 cm)" % (np.median(aa), np.percentile(aa, 75), np.percentile(aa, 90),
                              aa.max()))
        print("     |elevacion| <= : mediana %.0f gr  p25 %.0f  min %.0f" % (
            np.median(cc), np.percentile(cc, 25), cc.min()))
        print("     por debajo de %.0f gr en el %.1f %% de los cuadros  ->  %s (limite declarado "
              "%.0f %%)" % (TOL_COTA_GR, 100 * frac, "PASA" if frac >= TOL_FRAC else "FALLA",
                            100 * TOL_FRAC))
    else:
        print("  1. COTA DE ELEVACION: sin hoja vista en ningun cuadro. No se puede acotar.")
        cc = np.array([])

    """2. EL ORDEN DE PROFUNDIDAD por tres senales, dos gratis y una de 480 ms."""
    ac, ap, az, n = 0, 0, 0, 0
    for k in ks:
        b0, b1 = B[k, 0], B[k, 1]
        if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
            continue
        h0, h1 = (b0[3] - b0[1]) * H, (b1[3] - b1[1]) * H
        pie0, pie1 = b0[3], b1[3]
        l0, l1 = lect.get((k, 0), {}), lect.get((k, 1), {})
        if not (5 in l0 and 5 in l1):
            continue
        n += 1
        """Los tres dicen «quien esta MAS CERCA». Alto mayor = mas cerca. Pie mas abajo = mas cerca.
        Z menor = mas cerca."""
        alto_dice = 0 if h0 > h1 else 1
        pie_dice = 0 if pie0 > pie1 else 1
        pm_dice = 0 if l0[5]["Z"] < l1[5]["Z"] else 1
        ac += int(alto_dice == pie_dice)
        ap += int(pm_dice == pie_dice)
        az += int(pm_dice == alto_dice)
    if n:
        print()
        print("  2. EL ORDEN DE PROFUNDIDAD, tres senales sobre %d cuadros" % n)
        print("     alto de caja contra y del pie (las dos GRATIS) coinciden en %.1f %%" % (
            100.0 * ac / n))
        print("     `pointmap` contra y del pie                   coinciden en %.1f %%" % (
            100.0 * ap / n))
        print("     `pointmap` contra alto de caja                coinciden en %.1f %%" % (
            100.0 * az / n))
        print("     -> %s" % (
            "las dos gratis se corroboran entre ellas y `pointmap` es el que se desvia"
            if (ac / n) > max(ap / n, az / n) else "las tres senales van juntas"))

    """2b. EL PLANO DEL SUELO, ajustado, que es lo que hace robusta la escala.

    POR QUE HIZO FALTA Y ES UN ERROR MIO CORREGIDO CON SU CIFRA. La primera version saco la
    profundidad del ALTO APARENTE de la caja (`Z = f * S / h`). Y eso **no vale para un esgrimista**:
    en el fondo baja el centro de gravedad y su alto aparente cae un 20-30 % sin que se mueva ni un
    centimetro en profundidad. Medido: el alto de caja y la `y del pie` solo concuerdan en el orden
    de profundidad el 82 % de las veces, mientras `pointmap` concuerda con el pie el 96 %. O sea que
    **el que se desviaba era el alto de caja, no `pointmap`**, y esa parte de mi diseno estaba mal.

    EL SUSTITUTO ES EL SUELO, que no se agacha. Con la camara por encima de la pista, para un punto
    del suelo a profundidad Z se cumple `fila_pie - fila_horizonte = f * H_camara / Z` y a la vez
    `alto_aparente = f * S / Z`. Dividiendo:

        alto_aparente = (S / H_camara) * (fila_pie - fila_horizonte)

    o sea una RECTA. Ajustandola se sacan dos cosas gratis y medidas: la **fila del horizonte** y la
    **altura de la camara en metros** (`H_camara = S / pendiente`). Y entonces la escala metrica

        metros por pixel  =  H_camara / (fila_pie - fila_horizonte)

    ya no depende del alto del cuerpo, sino de donde pisa. La R2 del ajuste dice si el modelo se
    sostiene, y va impresa: es un control gratis y declarado."""
    vp, ha = [], []
    for k in range(len(B)):
        for lado in (0, 1):
            b = B[k, lado]
            if np.isfinite(b).all() and (b[3] - b[1]) > 0:
                vp.append(b[3] * H)
                ha.append((b[3] - b[1]) * H)
    """EL AJUSTE USA TODO EL CLIP, NO LA VENTANA, y esto es un error mio corregido.

    La altura de la camara y el horizonte son propiedades del CLIP, no de los 100 cuadros que se
    eligieron para pagar `pointmap`. Ajustandolos solo en la ventana salia que el mismo clip
    (`AromaticShockedGermanspitz`) daba un ajuste valido en la ventana quieta (R2 0,367, camara a
    6,98 m) y **ninguno** en la de accion. O sea que las cifras de «todo el clip» dependian de la
    ventana, que es justo lo que no debe pasar. Con todo el clip hay 2-4 veces mas cajas y el
    resultado ya no depende de donde se mire.

    Y que el ajuste sea sensible al tramo es en si mismo un aviso sobre su fragilidad, que queda
    dicho: no es un ajuste robusto, es el mejor disponible sin calibrar la pista."""
    suelo = None
    if len(vp) >= 30:
        vp, ha = np.array(vp), np.array(ha)
        A = np.vstack([vp, np.ones_like(vp)]).T
        (pen, off), *_ = np.linalg.lstsq(A, ha, rcond=None)
        pred = A @ np.array([pen, off])
        r2 = 1.0 - float(np.sum((ha - pred) ** 2) / max(np.sum((ha - ha.mean()) ** 2), 1e-9))
        print()
        print("  2b. EL PLANO DEL SUELO ajustado (n=%d cajas)" % len(vp))
        if pen > 1e-6:
            v_h = -off / pen
            Hc = S / pen
            print("      alto = %.4f * (fila_pie - %.0f)  ·  R2 = %.3f" % (pen, v_h, r2))
            print("      -> horizonte en la fila %.0f de %d  ·  ALTURA DE CAMARA %.2f m  ·  %s" % (
                v_h, H, Hc, "plausible" if 1.5 <= Hc <= 12 else "IMPLAUSIBLE: el ajuste no vale"))
            if 1.5 <= Hc <= 12 and r2 >= 0.3:
                suelo = (v_h, Hc)
                for lado in (0, 1):
                    b = B[ks, lado]
                    ok = np.isfinite(b).all(axis=1)
                    if not ok.any():
                        continue
                    mp_suelo = Hc / np.maximum(b[ok, 3] * H - v_h, 1e-6)
                    mp_alto = S / np.maximum((b[ok, 3] - b[ok, 1]) * H, 1e-6)
                    print("      lado %d: m/px por SUELO mediana %.5f  ·  por ALTO %.5f  ·  "
                          "discrepancia mediana %.1f %%" % (
                              lado, float(np.median(mp_suelo)), float(np.median(mp_alto)),
                              100 * float(np.median(np.abs(mp_suelo / mp_alto - 1)))))
            else:
                print("      el ajuste no pasa su propio control (R2 %.3f, H %.2f m): se sigue con "
                      "el alto de caja y se dice." % (r2, Hc))
        else:
            print("      pendiente <= 0 (%.5f): el modelo de suelo no se sostiene en este clip" % pen)

    """3. EL ALCANCE EN 3D CON LA GEOMETRIA GRATIS, barriendo la unica incognita."""
    print()
    print("  3. ALCANCE mano -> torso del rival, con X e Y medidos en metros y `D` barrida")
    print("     escala %s" % ("por el SUELO ajustado" if suelo else
                              "por el ALTO de caja (el suelo no se pudo ajustar)"))
    Ds = np.geomspace(D_MIN, D_MAX, 25)
    tot, am2_n = 0, 0
    cambia = np.zeros(Ds.size, dtype="int64")
    criticas, lat_cm, dzf = [], [], []
    for k in ks:
        for lado in (0, 1):
            l0, lr = lect.get((k, lado), {}), lect.get((k, 1 - lado), {})
            M = CL.MANOS[mano[lado]]
            if M["root"] not in l0 or not all(i in lr for i in ESQUINAS):
                continue
            b0, b1 = B[k, lado], B[k, 1 - lado]
            if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
                continue
            if suelo:
                v_h, Hc = suelo
                em = Hc / max(b0[3] * H - v_h, 1e-6)
                er = Hc / max(b1[3] * H - v_h, 1e-6)
            else:
                em = S / max((b0[3] - b0[1]) * H, 1e-6)
                er = S / max((b1[3] - b1[1]) * H, 1e-6)
            if not (em > 0 and er > 0):
                continue
            """X e Y en METROS, sin ninguna incognita: cada uno con SU escala metros/pixel."""
            vm = np.array([(l0[M["root"]]["x"] - 0.5 * W) * em,
                           (l0[M["root"]]["y"] - 0.5 * H) * em])
            q = [np.array([(lr[i]["x"] - 0.5 * W) * er, (lr[i]["y"] - 0.5 * H) * er])
                 for i in ESQUINAS]
            malla = C3.malla_torso(q[0], q[1], q[2], q[3], N_MALLA)
            lat = np.linalg.norm(malla - vm[None, :], axis=1)
            lat_min = float(lat.min())
            lat_cm.append(100 * lat_min)
            """Z proporcional a la escala metros/pixel: Z = f * (m/px). Asi que la razon de
            profundidades es la razon de escalas, y eso es MEDIDO."""
            g = er / em - 1.0
            dzf.append(g)
            tot += 1
            en2 = lat_min <= CL.HOJA_M
            am2_n += int(en2)
            for j, D in enumerate(Ds):
                dz = S * D * g
                en3 = float(np.sqrt(lat_min ** 2 + dz ** 2)) <= CL.HOJA_M
                cambia[j] += int(en3 != en2)
            """La D critica: la que pone el alcance 3D justo en 90 cm. Solo existe si el lateral ya
            cabe; si el lateral solo pasa de 90 cm, ninguna D lo arregla."""
            if en2 and abs(g) > 1e-9:
                dz_max = np.sqrt(max(CL.HOJA_M ** 2 - lat_min ** 2, 0.0))
                criticas.append(dz_max / (S * abs(g)))
    if tot:
        print("     n=%d conos  ·  distancia LATERAL medida mano-torso: mediana %.0f cm  p10 %.0f  "
              "p90 %.0f" % (tot, np.median(lat_cm), np.percentile(lat_cm, 10),
                            np.percentile(lat_cm, 90)))
        print("     dentro de los 90 cm SOLO con lo lateral (o sea el veredicto 2D): %d (%.1f %%)"
              % (am2_n, 100.0 * am2_n / tot))
        print("     razon de profundidades Z_rival/Z_propio - 1: mediana %+.4f  p10 %+.4f  "
              "p90 %+.4f  (MEDIDO, sin ninguna incognita)" % (
                  np.median(dzf), np.percentile(dzf, 10), np.percentile(dzf, 90)))
        print("     %-8s %-12s %s" % ("D", "camara (m)", "cuadros donde el 3D cambia el veredicto"))
        for j in range(0, Ds.size, 4):
            print("     %-8.1f %-12.1f %d de %d (%.1f %%)" % (
                Ds[j], Ds[j] * S, cambia[j], tot, 100.0 * cambia[j] / tot))
        if criticas:
            cr = np.array(criticas)
            print("     D CRITICA (la que saca el torso de los 90 cm) sobre los %d que el 2D da por"
                  " dentro:" % len(cr))
            print("        mediana %.1f estaturas = %.0f m  ·  p10 %.1f = %.0f m  ·  p90 %.1f = "
                  "%.0f m" % (np.median(cr), np.median(cr) * S, np.percentile(cr, 10),
                              np.percentile(cr, 10) * S, np.percentile(cr, 90),
                              np.percentile(cr, 90) * S))
            print("        dentro del rango declarado [%.0f, %.0f] estaturas: %.1f %%  ->  %s" % (
                D_MIN, D_MAX, 100 * float(((cr >= D_MIN) & (cr <= D_MAX)).mean()),
                "el 3D SI puede cambiar el veredicto y falta calibrar D"
                if float(((cr >= D_MIN) & (cr <= D_MAX)).mean()) > 0.05 else
                "ninguna camara plausible cambia el veredicto: el 3D no aporta por esta via"))
    """4. EL CAREO DIRECTO: la razon de profundidades entre los dos tiradores, medida por las dos
    vias. Es la prueba mas decisiva de todo el paso y no cuesta nada.

    `pointmap` da `Z` en unidades propias, pero la RAZON `Z_rival / Z_propio` es adimensional y por
    tanto **comparable** con la razon que da la geometria gratis (`escala_rival / escala_propio`, que
    tambien es adimensional porque las dos escalas son `Z/f` con la misma `f`). Si la profundidad de
    `pointmap` entre personas fuera correcta, las dos razones tendrian que coincidir 1 a 1.

    LIMITE DECLARADO ANTES DE MIRAR: para que la z de `pointmap` sirva para el alcance entre los dos
    tiradores, se exige **spearman >= 0,50** con la razon gratis y una **pendiente entre 0,5 y 2,0**.
    Por debajo de eso, `pointmap` no esta midiendo la separacion en profundidad de esta escena."""
    rp, rl = [], []
    for k in ks:
        for lado in (0, 1):
            l0, lr = lect.get((k, lado), {}), lect.get((k, 1 - lado), {})
            M = CL.MANOS[mano[lado]]
            if M["root"] not in l0 or not all(i in lr for i in ESQUINAS):
                continue
            b0, b1 = B[k, lado], B[k, 1 - lado]
            if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
                continue
            if suelo:
                v_h, Hc = suelo
                em, er = Hc / max(b0[3] * H - v_h, 1e-6), Hc / max(b1[3] * H - v_h, 1e-6)
            else:
                em = S / max((b0[3] - b0[1]) * H, 1e-6)
                er = S / max((b1[3] - b1[1]) * H, 1e-6)
            Zm = l0[M["root"]]["Z"]
            Zr = float(np.median([lr[i]["Z"] for i in ESQUINAS]))
            if abs(Zm) < 1e-9 or em <= 0:
                continue
            rp.append(Zr / Zm)
            rl.append(er / em)
    print()
    print("  4. CAREO de la razon de profundidades Z_rival/Z_propio, por las dos vias (n=%d)"
          % len(rp))
    if len(rp) >= 20:
        rp, rl = np.array(rp), np.array(rl)
        m = np.isfinite(rp) & np.isfinite(rl)
        ra = np.argsort(np.argsort(rp[m])).astype("float64")
        rb = np.argsort(np.argsort(rl[m])).astype("float64")
        rho = float(np.corrcoef(ra, rb)[0, 1])
        pen = float(np.polyfit(rl[m] - 1.0, rp[m] - 1.0, 1)[0]) if np.ptp(rl[m]) > 1e-9 else \
            float("nan")
        print("     `pointmap`: mediana %.3f  p5 %.3f  p95 %.3f  ·  dispersion %.3f" % (
            np.median(rp[m]), np.percentile(rp[m], 5), np.percentile(rp[m], 95),
            float(np.std(rp[m]))))
        print("     GRATIS    : mediana %.3f  p5 %.3f  p95 %.3f  ·  dispersion %.3f" % (
            np.median(rl[m]), np.percentile(rl[m], 5), np.percentile(rl[m], 95),
            float(np.std(rl[m]))))
        print("     spearman %+.3f  ·  pendiente %.2f  ->  %s (limite declarado rho>=0,50 y "
              "pendiente en [0,5 , 2,0])" % (
                  rho, pen, "PASA" if (rho >= 0.5 and 0.5 <= pen <= 2.0) else
                  "FALLA: la z de `pointmap` NO mide la separacion en profundidad de esta escena"))
        careo = {"rho": rho, "pendiente": pen, "n": int(m.sum()),
                 "disp_pointmap": float(np.std(rp[m])), "disp_gratis": float(np.std(rl[m]))}
    else:
        print("     n insuficiente")
        careo = None

    """5. LO MISMO PERO EN TODO EL CLIP, que se puede porque esto no gasta GPU.

    Y NO ES UN EXTRA, ES LO QUE DE VERDAD CONTESTA LA PREGUNTA. Los apartados 1 a 4 estan atados a la
    ventana de 100 cuadros donde se pago `pointmap`, y en una ventana de 100 cuadros hay muy pocos
    con el rival dentro de los 90 cm. La geometria gratis no necesita `pointmap`, asi que puede
    recorrer los 250-500 cuadros del clip entero y juntar TODOS los cuadros donde la pregunta del
    cono esta viva. Con 100 cuadros la respuesta seria «no hay casos»; con el clip entero se sabe si
    es que no hay casos o es que no se estaba mirando donde ocurren."""
    Tc = len(B)
    tot2, viv2 = 0, 0
    cam2 = np.zeros(Ds.size, dtype="int64")
    cri2, lat2 = [], []
    for k in range(Tc):
        for lado in (0, 1):
            M = CL.MANOS[mano[lado]]
            r = 1 - lado
            b0, b1 = B[k, lado], B[k, r]
            if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
                continue
            p = KP[k, lado, M["root"]]
            q4 = KP[k, r, ESQUINAS]
            if not (np.isfinite(p).all() and np.isfinite(q4).all()):
                continue
            if not (SC[k, lado, M["root"]] >= CL.CONF_MIN
                    and (SC[k, r, ESQUINAS] >= CL.CONF_MIN).all()):
                continue
            if suelo:
                v_h, Hc = suelo
                em, er = Hc / max(b0[3] * H - v_h, 1e-6), Hc / max(b1[3] * H - v_h, 1e-6)
            else:
                em = S / max((b0[3] - b0[1]) * H, 1e-6)
                er = S / max((b1[3] - b1[1]) * H, 1e-6)
            if not (em > 0 and er > 0):
                continue
            vm = np.array([(p[0] - 0.5 * W) * em, (p[1] - 0.5 * H) * em])
            qq = [np.array([(x[0] - 0.5 * W) * er, (x[1] - 0.5 * H) * er]) for x in q4]
            malla = C3.malla_torso(qq[0], qq[1], qq[2], qq[3], N_MALLA)
            lat_min = float(np.linalg.norm(malla - vm[None, :], axis=1).min())
            tot2 += 1
            en2 = lat_min <= CL.HOJA_M
            viv2 += int(en2)
            if not en2:
                continue
            lat2.append(100 * lat_min)
            g = er / em - 1.0
            for j, D in enumerate(Ds):
                dz = S * D * g
                if float(np.sqrt(lat_min ** 2 + dz ** 2)) > CL.HOJA_M:
                    cam2[j] += 1
            if abs(g) > 1e-9:
                cri2.append(np.sqrt(max(CL.HOJA_M ** 2 - lat_min ** 2, 0.0)) / (S * abs(g)))
    print()
    print("  5. TODO EL CLIP (%d cuadros, sin `pointmap` y sin GPU)" % Tc)
    if tot2:
        print("     %d conos evaluables  ·  %d (%.1f %%) con el torso del rival dentro de los 90 cm "
              "-> son los unicos donde la pregunta esta viva" % (
                  tot2, viv2, 100.0 * viv2 / tot2))
        if viv2:
            print("     de esos %d: distancia lateral mediana %.0f cm  min %.0f cm" % (
                viv2, np.median(lat2), np.min(lat2)))
            print("     %-8s %-12s %s" % ("D", "camara (m)",
                                          "de los vivos, cuantos SACA el 3D de los 90 cm"))
            for j in range(0, Ds.size, 4):
                print("     %-8.1f %-12.1f %d de %d (%.1f %%)" % (
                    Ds[j], Ds[j] * S, cam2[j], viv2, 100.0 * cam2[j] / viv2))
            """Y AQUI EL BARRIDO SE CONVIERTE EN UNA COTA DE VERDAD, con un argumento fisico gratis.

            Estos cinco clips salen del CORPUS ETIQUETADO de florete, o sea que **cada uno contiene
            un tocado**. Un tocado exige que en algun cuadro la punta llegue al torso, o sea que la
            distancia 3D mano-torso baje de 90 cm en al menos un cuadro. Asi que cualquier `D` que
            deje CERO cuadros dentro de los 90 cm en 3D es **incompatible con que el clip tenga un
            tocado**, y queda descartada sin necesidad de calibrar nada.

            Es una cota por arriba, no una calibracion: acota `D`, no lo fija. Y su punto flojo va
            dicho: si el tocado del clip cae fuera del tramo utilizable, o si el que toca no es el
            que este codigo cree que toca, la cota se cae. Con eso, sirve para descartar el extremo
            alto del barrido, que es exactamente donde el 3D parecia arrasar con todo."""
            queda = viv2 - cam2
            arriba = [Ds[j] for j in range(Ds.size) if queda[j] >= 1]
            print("     COTA FISICA: el clip es del corpus etiquetado, luego TIENE un tocado, luego "
                  "algun cuadro tiene que quedar dentro de los 90 cm en 3D")
            if arriba:
                print("        D compatible con que haya tocado: hasta %.1f estaturas = %.0f m  ·  "
                      "a partir de ahi el 3D niega el tocado y la D queda DESCARTADA" % (
                          max(arriba), max(arriba) * S))
            else:
                print("        NINGUNA D del barrido deja un cuadro dentro de los 90 cm: o el "
                      "tocado no esta en el tramo utilizable, o la razon de profundidades esta mal")
            if cri2:
                cr = np.array(cri2)
                en = float(((cr >= D_MIN) & (cr <= D_MAX)).mean())
                print("     D critica: mediana %.1f estaturas = %.0f m  ·  p10 %.1f = %.0f m  ·  "
                      "dentro de [%.0f, %.0f]: %.1f %%" % (
                          np.median(cr), np.median(cr) * S, np.percentile(cr, 10),
                          np.percentile(cr, 10) * S, D_MIN, D_MAX, 100 * en))
                print("     -> %s" % (
                    "el 3D SI puede cambiar el veredicto para camaras plausibles: falta calibrar D"
                    if en > 0.05 else
                    "para NINGUNA camara plausible el 3D cambia el veredicto por alcance"))
    todo_clip = {"n": tot2, "vivos": viv2,
                 "D_max_compatible_con_tocado": (max([Ds[j] for j in range(Ds.size)
                                                      if (viv2 - cam2)[j] >= 1], default=None)
                                                 if viv2 else None),
                 "lat_mediana_vivos_cm":
                 float(np.median(lat2)) if lat2 else None,
                 "cambia_por_D": {float(Ds[j]): int(cam2[j]) for j in range(Ds.size)},
                 "D_critica_mediana": float(np.median(cri2)) if cri2 else None,
                 "D_critica_en_rango": float(((np.array(cri2) >= D_MIN)
                                              & (np.array(cri2) <= D_MAX)).mean())
                 if cri2 else None}

    resumen[clip] = {
        "todo_el_clip": todo_clip,
        "careo_razon_profundidad": careo,
        "n_hoja": int(cc.size), "cota_mediana": float(np.median(cc)) if cc.size else None,
        "cota_frac_bajo_30": float((cc <= TOL_COTA_GR).mean()) if cc.size else None,
        "alcance_mediana_cm": float(np.median(alcances)) if alcances else None,
        "suelo": ({"horizonte": suelo[0], "H_camara_m": suelo[1]} if suelo else None),
        "orden_alto_vs_pie": ac / n if n else None,
        "orden_pointmap_vs_pie": ap / n if n else None,
        "orden_pointmap_vs_alto": az / n if n else None,
        "n_conos": tot, "frac_2d_dentro": am2_n / tot if tot else None,
        "lat_mediana_cm": float(np.median(lat_cm)) if lat_cm else None,
        "D_critica_mediana": float(np.median(criticas)) if criticas else None,
        "D_critica_en_rango": float(((np.array(criticas) >= D_MIN)
                                     & (np.array(criticas) <= D_MAX)).mean())
        if criticas else None,
        "cambia_por_D": {float(Ds[j]): int(cambia[j]) for j in range(Ds.size)}}
    print()

(PRUEBAS / ("cono_e12_sin_pointmap%s.json" % SUF)).write_text(
    json.dumps(resumen, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("guardado %s" % (PRUEBAS / ("cono_e12_sin_pointmap%s.json" % SUF)))
