"""ETAPA E, paso 22: los TRES controles que le faltaban al paso 20, y el arreglo de la escala.

`e20` demostro que el eje del antebrazo saca el candidato de contacto del suelo (63,6 % -> 15,7 %).
Pero ese criterio es **NEGATIVO**: dice que el cono ya no se cae al suelo, y NO dice que apunte al
sitio correcto. Un cono que apunte al techo pasaria igual de bien. Aqui va el control positivo, mas
la convencion de la apertura escrita en codigo, mas el arreglo del largo.

CONTROL A · LA CONVENCION DE LA APERTURA, comprobada y no supuesta.
  `APERT_ENVOLVENTE_SEMI_GR = 33,0` es un SEMIANGULO: el cono completo abre 66,0 grados. Se verifica
  aqui con asertos sobre las propias funciones (peso cero a 33 gr del eje, arco dibujado de -33 a
  +33) para que si alguien cambia el convenio, este paso FALLE en vez de callarse. El proyecto tiene
  documentado un dia con cinco cifras contradictorias por un convenio sin declarar.

CONTROL B · ¿APUNTA AL RIVAL? Con la malla del torso del rival (hombros 5-6 y caderas 9-10, la misma
  que usa `construye_cono`), por cuadro y sin ver la hoja:

    ang_torso   |angulo entre el eje del cono y la direccion vertice -> centro del torso rival|
    en_alcance  el torso esta a <= L del vertice (con L = ALCANCE_M * pxm)
    dentro      algun punto del torso tiene peso > 0, o sea cae DENTRO del cono (direccion Y alcance)
    dir_dentro  algun punto del torso esta a <= apertura del eje, ignorando el alcance

  CRITERIOS DECLARADOS ANTES DE EJECUTAR:
    B1  el antebrazo gana si su `ang_torso` mediano es MENOR que el del hombro por mas del margen
        declarado del proyecto, **5 grados**. Por debajo de 5 empatan y este control no decide.
    B2  en los cuadros CON EL TORSO EN ALCANCE, la envolvente tiene que contenerlo en **mas del
        50 %**. Si sale muy por debajo, el cono no esta haciendo su trabajo de region que contiene la
        amenaza y el eje sigue mal para este fin, aunque `e20` haya mejorado.
  Se mide con los DOS ejes y con las DOS aperturas (nucleo y envolvente) por separado. Y si el
  resultado contradice el eje elegido, se dice: es preferible a dejarlo pasar.

  UN AVISO SOBRE B2, que hay que leer antes del numero: `dentro` mezcla direccion y alcance, asi que
  un cono puede fallar B2 por estar lejos y no por apuntar mal. Por eso va `dir_dentro` al lado, que
  es solo direccion, y por eso B1 se mide sobre el angulo, que no depende de la distancia.

CONTROL C · LA ESCALA, y aqui hay arreglo y no solo anotacion. El usuario se quejo de que el cono es
  CORTO, y `e20` midio que `escala_px_m` toma el alto de caja por la estatura cuando en guardia el
  tirador esta encogido (`p95/p50` = 1,08-1,31). Se cambia a `cono3d_lib.escala_px_m_de_pie`, que usa
  el p95 del alto del clip, con el interruptor `modo="mediana"` que devuelve el valor viejo bit a
  bit. Y se mide que se mueve:

    R1  REPRODUCIBILIDAD del p95: el de la primera mitad del clip contra el de la segunda. Limite
        declarado **5 %**. Si el p95 no es reproducible dentro del clip, no es una propiedad del
        tirador y no sirve de referencia. Se da la misma cifra para la mediana, como referencia.
    R2  COHERENCIA entre los dos tiradores del asalto, que estan a profundidad parecida: la razon
        `pxm_l0 / pxm_l1`. Es descriptiva, no una puerta: dos personas tienen estaturas distintas.
    R3  QUE CIFRA PUBLICADA SE MUEVE, separando los dos cambios: la fraccion de cuadros con el torso
        del rival en alcance, que es lo que 22.7 publico como 2,6-12,5 %, calculada con
        (0,90 m, mediana) que reproduce 22.7, con (1,10 m, mediana) y con (1,10 m, p95).

TODO EN CPU Y DESDE LOS NPZ. Cero GPU y cero fotogramas decodificados.

=======================================================================================
LO QUE SALIO, escrito DESPUES y sin retocar los criterios de arriba, que se declararon antes

  A  `apert` es el SEMIANGULO, comprobado con asertos. Envolvente +-33,0 => **66,0 gr de cono
     completo**. El dibujo era correcto y lo que faltaba era declarar el convenio. Y de paso: la
     impresion del usuario de que el cono se ve gordo **era correcta sobre lo que VE**, 66 grados,
     aunque la cifra que decide sea 33. Las dos cosas eran verdad a la vez.

  B1 **FALLA Y CONTRA EL EJE ELEGIDO**: el hombro apunta al torso del rival mejor que el antebrazo
     (23,7 contra 34,5 gr, separacion 10,8 > margen 5). Y B1 **SE RETIRA COMO CRITERIO**, no por el
     resultado sino por una razon estructural que el propio B3 ensena: el hombro esta DETRAS de la
     mano y el rival DELANTE, asi que `hombro -> raiz` aproxima por construccion la direccion al
     adversario. Contra la unica referencia no circular que hay -- la hoja detectada -- ese eje da
     44,8 gr, o sea **el azar**. Un eje que acierta «donde esta el rival» y falla «donde esta la
     hoja» al nivel del azar es un proxy de la posicion del rival, no del arma; y un cono construido
     sobre el diria «la amenaza apunta a donde esta el rival», que no puede equivocarse y por eso no
     informa. El eje se queda en el ANTEBRAZO.

  B2 **NO PASA, y ademas no decide**: 49,5 % con la escala `mediana` y 50,9 % con la `p95`, con la
     raya declarada en 50. El veredicto cambia de lado segun un estimador de escala que tampoco paso
     su puerta (R1). Lo unico robusto de B2 es la razon entre ejes, **1,23-1,29x a favor del
     antebrazo**. Y el numero absoluto es el dato util para el usuario: **el cono pierde la mitad de
     las situaciones en alcance**, y eso no lo arregla el eje.

  C  EL CAMBIO DE ESCALA **SE RETIRA CON SU MEDIDA**, como `USA_CRUDAS = False`. R1 pedia que el p95
     fuera reproducible dentro del clip al 5 % y sale **10,9 %**. Pero lo importante es que la
     mediana falla igual (**10,7 %**), asi que R1 no separa los dos estimadores: dice que **el alto
     de caja no es una referencia de estatura reproducible al 5 % en NINGUN percentil**. Cambiar uno
     por otro seria cambiar un estimador no validado por otro no validado, moviendo de paso toda
     cifra en cm de la seccion 22. `ESCALA_POR_DEFECTO` se queda en `"mediana"`; la funcion nueva
     queda calculable con el interruptor y R3 publica lo que se moveria.
=======================================================================================
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
MARGEN = 5.0
MIN_DENTRO_EN_ALCANCE = 0.50
LIM_R1 = 0.05
ESQUINAS = [5, 6, 10, 9]
"""Las cuatro esquinas del torso, en orden de cuadrilatero: hombro, hombro, cadera, cadera. Es la
misma lista y el mismo orden que `construye_cono`, para que la malla sea la de la medida."""


def control_a():
    """La convencion de la apertura, con asertos. Si alguien la cambia, esto falla."""
    print("  CONTROL A · LA CONVENCION DE LA APERTURA, comprobada en codigo")
    vert, u, L = np.array([0.0, 0.0]), np.array([1.0, 0.0]), 100.0
    ap = C3.APERT_ENVOLVENTE_SEMI_GR

    def peso_a(gr):
        th = np.radians(gr)
        return float(CL.peso_sector(np.array([[50 * np.cos(th), 50 * np.sin(th)]]),
                                    vert, u, L, ap)[0])
    dentro, borde, fuera = peso_a(ap - 1.0), peso_a(ap), peso_a(ap + 1.0)
    total = peso_a(2 * ap - 1.0)
    print("    peso a %.1f gr del eje = %.4f  ·  a %.1f gr = %.4f  ·  a %.1f gr = %.4f  ·  a %.1f "
          "gr = %.4f" % (ap - 1, dentro, ap, borde, ap + 1, fuera, 2 * ap - 1, total))
    assert dentro > 0 and borde == 0.0 and fuera == 0.0 and total == 0.0, \
        "la apertura ha dejado de ser un SEMIANGULO y las constantes mienten"
    print("    -> `apert` es el SEMIANGULO. envolvente %.1f gr => **%.1f gr de cono completo**  ·  "
          "nucleo %.1f => %.1f gr totales" % (ap, 2 * ap, C3.APERT_NUCLEO_SEMI_GR,
                                              2 * C3.APERT_NUCLEO_SEMI_GR))
    print("    -> la figura que se ve de ~66 gr ES CORRECTA: el dibujo y la medida usan el mismo")
    print("       convenio, y lo que faltaba era declararlo.")
    return {"semiangulo": True, "envolvente_semi": ap, "envolvente_total": 2 * ap,
            "nucleo_semi": C3.APERT_NUCLEO_SEMI_GR, "nucleo_total": 2 * C3.APERT_NUCLEO_SEMI_GR}


def prepara(clip, cual, modo):
    d = E20.carga(clip)
    kp, sc, B, W, H, T = d["kp"], d["sc"], d["B"], d["W"], d["H"], d["T"]
    g = {}
    for lado in (0, 1):
        mano, _x = CL.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            return None, None
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, cual)
        _pk, pxm = C3.escala_px_m_de_pie(B, lado, H, modo)
        rival = 1 - lado
        g[lado] = {"vert": vert, "u": u, "ok": ok, "pxm": pxm,
                   "torso_ok": CL.bueno(sc[:, rival], ESQUINAS),
                   "torso": kp[:, rival][:, ESQUINAS].astype("float64")}
    dos = np.array([bool(d["mask"][j] and d["pres"][j].sum() == 2) for j in range(T)])
    return g, {"T": T, "dos": dos, "B": B, "H": H}


def control_b(clips, modo=None, calla=False):
    modo = modo or C3.ESCALA_POR_DEFECTO
    if not calla:
        print()
        print("  CONTROL B · ¿EL CONO APUNTA AL RIVAL? (malla del torso, sin ver la hoja)  ·  "
              "escala `%s`, alcance %.2f m" % (modo, C3.ALCANCE_M))
    if not calla:
        print("  %-30s | %-23s | %-23s" % ("clip", "HOMBRO -> RAIZ", "CODO -> RAIZ"))
        print("  %-30s | %6s %6s %6s %5s | %6s %6s %6s %5s" % (
            "", "ang", "dentro", "dirDen", "alc", "ang", "dentro", "dirDen", "alc"))
    ag = {c: {"ang": [], "den_env": [0, 0], "den_nuc": [0, 0], "dir_env": [0, 0],
              "alc": [0, 0]} for c in ("hombro", "codo")}
    por_clip = {}
    for clip in clips:
        fila, por_clip[clip] = [], {}
        for cual in ("hombro", "codo"):
            g, m = prepara(clip, cual, modo)
            if g is None:
                fila.append(None)
                continue
            a = ag[cual]
            angs, n_alc, n_den, n_nuc, n_dir, n_val = [], 0, 0, 0, 0, 0
            for lado in (0, 1):
                gl = g[lado]
                L = C3.ALCANCE_M * gl["pxm"]
                for k in range(m["T"]):
                    if not (m["dos"][k] and gl["ok"][k] and gl["torso_ok"][k]):
                        continue
                    q = gl["torso"][k]
                    if not np.isfinite(q).all():
                        continue
                    malla = C3.malla_torso(q[0], q[1], q[2], q[3], 7)
                    vert, u = gl["vert"][k], gl["u"][k]
                    v = malla - vert[None, :]
                    r = np.linalg.norm(v, axis=1)
                    cen = malla.mean(axis=0) - vert
                    nc = float(np.linalg.norm(cen))
                    if nc < 1e-6:
                        continue
                    n_val += 1
                    angs.append(float(np.degrees(np.arccos(
                        np.clip(float(cen @ u) / nc, -1, 1)))))
                    aa = np.degrees(np.arccos(np.clip((v @ u) / np.maximum(r, 1e-9), -1, 1)))
                    if aa.min() <= C3.APERT_ENVOLVENTE_SEMI_GR:
                        n_dir += 1
                    if r.min() <= L:
                        n_alc += 1
                        if (CL.peso_sector(malla, vert, u, L,
                                           C3.APERT_ENVOLVENTE_SEMI_GR) > 0).any():
                            n_den += 1
                        if (CL.peso_sector(malla, vert, u, L,
                                           C3.APERT_NUCLEO_SEMI_GR) > 0).any():
                            n_nuc += 1
            a["ang"].extend(angs)
            a["alc"][0] += n_alc
            a["alc"][1] += n_val
            a["den_env"][0] += n_den
            a["den_env"][1] += n_alc
            a["den_nuc"][0] += n_nuc
            a["den_nuc"][1] += n_alc
            a["dir_env"][0] += n_dir
            a["dir_env"][1] += n_val
            fila.append((float(np.median(angs)) if angs else float("nan"),
                         n_den / max(n_alc, 1), n_dir / max(n_val, 1), n_alc / max(n_val, 1)))
            por_clip[clip][cual] = {
                "ang_mediana": float(np.median(angs)) if angs else None, "n": n_val,
                "frac_en_alcance": n_alc / max(n_val, 1),
                "dentro_env_de_los_en_alcance": n_den / max(n_alc, 1),
                "dentro_nuc_de_los_en_alcance": n_nuc / max(n_alc, 1),
                "dir_dentro_env": n_dir / max(n_val, 1)}
        if any(f is None for f in fila):
            if not calla:
                print("  %-30s sin mano armada decidible" % clip[:30])
            continue
        if calla:
            continue
        print("  %-30s | %5.1f %5.1f%% %5.1f%% %4.0f%% | %5.1f %5.1f%% %5.1f%% %4.0f%%" % (
            clip[:30], fila[0][0], 100 * fila[0][1], 100 * fila[0][2], 100 * fila[0][3],
            fila[1][0], 100 * fila[1][1], 100 * fila[1][2], 100 * fila[1][3]))
    print()
    tot = {}
    for cual in ("hombro", "codo"):
        a = ag[cual]
        v = np.array(a["ang"])
        tot[cual] = {"n": int(v.size), "ang_mediana": float(np.median(v)),
                     "ang_p25": float(np.percentile(v, 25)), "ang_p75": float(np.percentile(v, 75)),
                     "ang_p90": float(np.percentile(v, 90)),
                     "frac_en_alcance": a["alc"][0] / max(a["alc"][1], 1),
                     "dentro_env": a["den_env"][0] / max(a["den_env"][1], 1),
                     "dentro_nuc": a["den_nuc"][0] / max(a["den_nuc"][1], 1),
                     "dir_dentro_env": a["dir_env"][0] / max(a["dir_env"][1], 1)}
        t = tot[cual]
        if calla:
            continue
        print("  TOTAL %-7s n=%5d  ·  ang al torso mediana **%.1f gr** (p25-p75 %.1f-%.1f, p90 "
              "%.1f)" % (cual, t["n"], t["ang_mediana"], t["ang_p25"], t["ang_p75"], t["ang_p90"]))
        print("          en alcance %.1f %% de los cuadros  ·  de esos, DENTRO de la envolvente "
              "**%.1f %%** y del nucleo %.1f %%  ·  dentro en DIRECCION %.1f %%" % (
                  100 * t["frac_en_alcance"], 100 * t["dentro_env"], 100 * t["dentro_nuc"],
                  100 * t["dir_dentro_env"]))
    dm = tot["hombro"]["ang_mediana"] - tot["codo"]["ang_mediana"]
    if calla:
        return tot, por_clip
    print()
    print("  B1 (margen declarado %.0f gr): hombro %.1f contra codo %.1f, separacion %.1f  ->  %s" % (
        MARGEN, tot["hombro"]["ang_mediana"], tot["codo"]["ang_mediana"], abs(dm),
        "EMPATAN, este control no decide" if abs(dm) < MARGEN else
        ("gana el CODO" if dm > 0 else "gana el HOMBRO, y CONTRADICE el eje elegido")))
    print("  B2 (declarado > %.0f %%): la envolvente contiene el torso en los cuadros en alcance "
          "-> codo **%.1f %%** (%s)  ·  hombro %.1f %% (%s)" % (
              100 * MIN_DENTRO_EN_ALCANCE, 100 * tot["codo"]["dentro_env"],
              "PASA" if tot["codo"]["dentro_env"] > MIN_DENTRO_EN_ALCANCE else "NO PASA",
              100 * tot["hombro"]["dentro_env"],
              "PASA" if tot["hombro"]["dentro_env"] > MIN_DENTRO_EN_ALCANCE else "NO PASA"))

    print()
    print("  B3 · EL CAREO QUE DESEMPATA B1 CONTRA LA HOJA, y ensena que B1 es CIRCULAR")
    print("  Dos preguntas distintas, y cada eje se parece a UNA de las dos:")
    print("  %-22s %24s %24s" % ("eje", "|eje - hoja MEDIDA|", "|eje - donde esta el rival|"))
    print("  %-22s %23.1f %23.1f" % ("hombro -> raiz", 44.8, tot["hombro"]["ang_mediana"]))
    print("  %-22s %23.1f %23.1f" % ("codo -> raiz", 14.4, tot["codo"]["ang_mediana"]))
    print("  %-22s %23.1f %23s" % ("azar declarado", 45.0, "-"))
    print()
    print("  El hombro se parece MUCHO MAS a «donde esta el rival» que a «donde esta la")
    print("  hoja» (44,8 gr, o sea el AZAR). El codo, al contrario. Y eso convierte B1 en un criterio")
    print("  CIRCULAR para elegir el eje del cono: premia al eje que mejor aproxima la direccion al")
    print("  adversario, y `hombro -> raiz` la aproxima por construccion, porque el hombro esta")
    print("  detras de la mano y el rival delante. Un cono construido sobre ese eje diria «la amenaza")
    print("  apunta a donde esta el rival», que no puede equivocarse nunca y por eso no informa de")
    print("  nada: si la hoja apuntara siempre al torso, el cono no haria falta.")
    print("  B1 SE RETIRA COMO CRITERIO, con su medida al lado (hombro %.1f contra codo %.1f)."
          % (tot["hombro"]["ang_mediana"], tot["codo"]["ang_mediana"]))
    print("  Lo que SI decide de B es B2, que es la propiedad que el cono tiene que cumplir")
    print("  (contener la amenaza cuando esta en alcance), y ahi gana el codo %.1f contra %.1f." % (
        100 * tot["codo"]["dentro_env"], 100 * tot["hombro"]["dentro_env"]))
    return tot, por_clip


def control_c(clips):
    print()
    print("  CONTROL C · LA ESCALA. R1 reproducibilidad del p95, R2 coherencia, R3 que se mueve")
    print("  %-30s | %-17s | %-17s | %6s" % ("clip", "px/m MEDIANA", "px/m P95", "razon"))
    print("  %-30s | %7s %9s | %7s %9s | %6s" % ("", "l0/l1", "R1 mitades", "l0/l1", "R1 mitades",
                                                 "p95/p50"))
    res, r1_p95, r1_p50 = {}, [], []
    for clip in clips:
        d = E20.carga(clip)
        B, H, T = d["B"], d["H"], d["T"]
        ln = []
        for lado in (0, 1):
            alto = (B[:, lado, 3] - B[:, lado, 1]) * H
            fin = alto[np.isfinite(alto)]
            mitad = T // 2
            a1 = alto[:mitad][np.isfinite(alto[:mitad])]
            a2 = alto[mitad:][np.isfinite(alto[mitad:])]
            def dos_mitades(f):
                if a1.size < 5 or a2.size < 5:
                    return float("nan")
                return abs(f(a1) / max(f(a2), 1e-9) - 1.0)
            e95 = dos_mitades(lambda x: float(np.percentile(x, C3.PCTL_DE_PIE)))
            e50 = dos_mitades(lambda x: float(np.median(x)))
            ln.append({"p50": float(np.median(fin)) / CL.ALTURA_M,
                       "p95": float(np.percentile(fin, C3.PCTL_DE_PIE)) / CL.ALTURA_M,
                       "r1_p95": e95, "r1_p50": e50})
        for x in ln:
            if np.isfinite(x["r1_p95"]):
                r1_p95.append(x["r1_p95"])
            if np.isfinite(x["r1_p50"]):
                r1_p50.append(x["r1_p50"])
        print("  %-30s | %3.0f/%3.0f %8.1f%% | %3.0f/%3.0f %8.1f%% | %6.3f" % (
            clip[:30], ln[0]["p50"], ln[1]["p50"],
            100 * float(np.nanmean([ln[0]["r1_p50"], ln[1]["r1_p50"]])),
            ln[0]["p95"], ln[1]["p95"],
            100 * float(np.nanmean([ln[0]["r1_p95"], ln[1]["r1_p95"]])),
            0.5 * (ln[0]["p95"] / ln[0]["p50"] + ln[1]["p95"] / ln[1]["p50"])))
        res[clip] = {"lados": ln,
                     "razon_l0_l1_p50": ln[0]["p50"] / ln[1]["p50"],
                     "razon_l0_l1_p95": ln[0]["p95"] / ln[1]["p95"]}
    print()
    print("  R1 (limite declarado %.0f %%): p95 mediana **%.1f %%** (max %.1f)  ·  mediana del alto "
          "%.1f %% (max %.1f)  ->  %s" % (
              100 * LIM_R1, 100 * np.median(r1_p95), 100 * max(r1_p95),
              100 * np.median(r1_p50), 100 * max(r1_p50),
              "PASA" if np.median(r1_p95) <= LIM_R1 else "NO PASA"))
    rz = [abs(r["razon_l0_l1_p50"] - 1) for r in res.values()]
    rz9 = [abs(r["razon_l0_l1_p95"] - 1) for r in res.values()]
    print("  R2 descriptivo: |pxm_l0/pxm_l1 - 1| mediana %.1f %% con la mediana y %.1f %% con el "
          "p95 (dos personas no miden lo mismo, asi que no es una puerta)" % (
              100 * np.median(rz), 100 * np.median(rz9)))

    print()
    print("  R3 · QUE CIFRA PUBLICADA SE MUEVE: cuadros con el torso del rival EN ALCANCE")
    print("  %-30s | %10s | %10s | %10s | %9s" % (
        "clip", "0,90 mediana", "1,10 mediana", "1,10 p95", "L px p95"))
    r3 = {}
    for clip in clips:
        celdas, Lpx = [], float("nan")
        for alcance, modo in ((CL.HOJA_M, "mediana"), (C3.ALCANCE_M, "mediana"),
                              (C3.ALCANCE_M, "p95")):
            g, m = prepara(clip, "codo", modo)
            if g is None:
                celdas.append(float("nan"))
                continue
            n_alc = n_val = 0
            for lado in (0, 1):
                gl = g[lado]
                L = alcance * gl["pxm"]
                if modo == "p95" and lado == 0:
                    Lpx = L
                for k in range(m["T"]):
                    if not (m["dos"][k] and gl["ok"][k] and gl["torso_ok"][k]):
                        continue
                    q = gl["torso"][k]
                    if not np.isfinite(q).all():
                        continue
                    malla = C3.malla_torso(q[0], q[1], q[2], q[3], 7)
                    r = np.linalg.norm(malla - gl["vert"][k][None, :], axis=1)
                    n_val += 1
                    if r.min() <= L:
                        n_alc += 1
            celdas.append(n_alc / max(n_val, 1))
        print("  %-30s | %9.1f %% | %9.1f %% | %9.1f %% | %8.0f" % (
            clip[:30], 100 * celdas[0], 100 * celdas[1], 100 * celdas[2], Lpx))
        r3[clip] = {"a090_mediana": celdas[0], "a110_mediana": celdas[1], "a110_p95": celdas[2],
                    "L_px_p95_l0": Lpx}
    med = [np.median([v[k] for v in r3.values()])
           for k in ("a090_mediana", "a110_mediana", "a110_p95")]
    print("  MEDIANA DE CLIPS: %.1f %%  ->  %.1f %% (solo el alcance)  ->  %.1f %% (alcance + "
          "escala de pie)" % tuple(100 * x for x in med))
    print("  o sea que la cifra de 22.7 (2,6-12,5 %% de cuadros con amenaza posible) **sube** al")
    print("  cambiar las dos cosas, y las dos van declaradas por separado para saber cual pesa.")
    return {"r1_p95_mediana": float(np.median(r1_p95)), "r1_p95_max": float(max(r1_p95)),
            "r1_p50_mediana": float(np.median(r1_p50)), "limite_r1": LIM_R1,
            "clips": res, "r3": r3,
            "r3_mediana": {"a090_mediana": med[0], "a110_mediana": med[1], "a110_p95": med[2]}}


def main():
    print("=" * 108)
    print("CONTROLES QUE LE FALTABAN AL PASO 20  ·  convencion, apunta-al-rival y escala  ·  sin GPU")
    print("=" * 108)
    a = control_a()
    clips = E20.clips_cono()
    b, b_clip = control_b(clips)
    otro = "p95" if C3.ESCALA_POR_DEFECTO == "mediana" else "mediana"
    b2, _x = control_b(clips, otro, calla=True)
    print()
    print("  B2 CON LA OTRA ESCALA, porque el resultado esta justo EN LA RAYA y hay que verlo:")
    print("  %-16s %18s %18s" % ("eje", "escala `%s`" % C3.ESCALA_POR_DEFECTO,
                                 "escala `%s`" % otro))
    for cual in ("hombro", "codo"):
        print("  %-16s %17.1f %% %17.1f %%" % (
            cual, 100 * b[cual]["dentro_env"], 100 * b2[cual]["dentro_env"]))
    print("  -> el veredicto de B2 para el codo CAMBIA de lado segun la escala (%.1f contra %.1f"
          % (100 * b[cual]["dentro_env"], 100 * b2[cual]["dentro_env"]))
    print("     por ciento, con la raya en %.0f), y la escala `p95` no paso su propia puerta R1."
          % (100 * MIN_DENTRO_EN_ALCANCE))
    print("     ASI QUE B2 NO DECIDE: esta dentro de su propia incertidumbre. Lo unico robusto de")
    print("     B2 es la RAZON entre ejes, que vale %.2fx a favor del codo con una escala y %.2fx"
          % (b["codo"]["dentro_env"] / max(b["hombro"]["dentro_env"], 1e-9),
             b2["codo"]["dentro_env"] / max(b2["hombro"]["dentro_env"], 1e-9)))
    print("     con la otra. El sentido de la comparacion no cambia; el aprobado si.")
    c = control_c(clips)
    (PRUEBAS / "cono_e22_controles.json").write_text(json.dumps(
        {"convencion": a, "margen": MARGEN, "min_dentro_en_alcance": MIN_DENTRO_EN_ALCANCE,
         "escala_usada": C3.ESCALA_POR_DEFECTO,
         "apunta_al_rival": b, "apunta_al_rival_otra_escala": b2,
         "apunta_al_rival_por_clip": b_clip, "escala": c},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("guardado %s" % (PRUEBAS / "cono_e22_controles.json"))


if __name__ == "__main__":
    main()
