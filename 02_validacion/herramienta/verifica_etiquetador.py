"""El careo del etiquetador: la antisimetria, las lineas base y el acierto, con su tabla.

QUE SE COMPRUEBA Y EN QUE ORDEN. Primero lo que decide si el resto de numeros se pueden leer, y
solo despues los numeros:

  1  LA ANTISIMETRIA, con un espejo DE VERDAD. No se comprueba intercambiando las dos ranuras del
     tensor de rasgos -- eso seria comprobar que `A[[1,0]][[1,0]] == A`, que es cierto siempre y no
     dice nada. Se construye un `poses_<clip>.npz` **espejado de verdad** (`x -> W-x` en cada
     keypoint y en cada caja, y las dos personas cambiadas), se le pasa el extractor de rasgos
     entero y se exige que salgan los mismos rasgos con las ranuras cambiadas. Eso es lo que
     prueba que ningun rasgo del bloque propio depende de que lado de la imagen ocupa el tirador.

  2  QUE EL MODELO INVIERTE LA RESPUESTA en el 100 % de los clips. Si no, la antisimetria esta
     deseada y no impuesta, y el acierto de la cabeza 1 no vale: cabe el atajo de "el de la
     izquierda inicia mas".

  3  LAS LINEAS BASE, declaradas antes de entrenar, con **acierto y cobertura por separado**. Una
     linea base que responde al 30 % de los clips con 0,90 y otra que responde al 100 % con 0,62
     no se pueden comparar por el primer numero.

  4  EL ACIERTO en el reservado POR TORNEO, con su matriz de confusion y con el techo humano
     puesto al lado. El techo es 0,688 / 0,619 / 0,481: batir el prior no es aprender esgrima, y
     pasarse del techo no es acertar mas, es sospechoso.

Y UNA PUERTA DE DENOMINADOR ANTES DE DAR NINGUNA CIFRA. Con menos de `--minimo` clips (60) no se
entrena y no se da acierto; entre 60 y 300 se entrena pero se avisa de que el reservado por torneo
se queda en unas decenas y la diferencia con la linea base **no es distinguible del azar**. Bajar
`--minimo` sirve para comprobar que la maquinaria corre, no para obtener un numero.

    python verifica_etiquetador.py [--cabeza initiated] [--epocas 60] [--minimo 60]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import etiquetador as ET  # noqa: E402
import rutas as RU  # noqa: E402


PARES = ([(5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
          (15, 18), (16, 19), (17, 20)]
         + [(21 + i, 42 + i) for i in range(21)])
"""LOS PARES SIMETRICOS del modelo de pose, que un espejo DE VERDAD tambien intercambia.

Espejar un video no solo manda `x -> W-x`: el estimador de pose vuelve a etiquetar las partes, y
lo que era el hombro izquierdo pasa a ser el derecho. Su `flip_pairs` es el emparejamiento por
paridad (5,6) (7,8) (9,10) (11,12) (13,14), los pies 15-20 en tres pares, y las dos manos enteras,
21-41 contra 42-62. Sin aplicar esto, el "espejo" de la prueba seria un espejo que no existe y la
prueba seria mas floja de lo que dice ser. La cara (63 en adelante) no se toca porque ningun rasgo
la usa."""


def espeja_npz(clip, destino):
    """QUE HACE: escribe un `poses_<clip>_ESPEJO.npz` que es el clip visto en un espejo.

    `x -> W - x` en cada keypoint, `xyxy -> W-x2, y1, W-x1, y2` en cada caja, **los pares
    simetricos intercambiados** (ver `PARES`) y **las dos personas cambiadas de sitio**, porque en
    el espejo el de la izquierda es el de la derecha. Es la unica forma de comprobar el extractor
    de punta a punta: cualquier rasgo que mire la x absoluta saltara aqui y en ningun otro sitio.
    """
    p = RU.TRABAJO / ("poses_%s.npz" % clip)
    with np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    W = float(d["W"])
    KP = d["KP"].copy()
    SC = d["SC"].copy()
    KP[:, :, :, 0] = W - KP[:, :, :, 0]
    perm = np.arange(KP.shape[2])
    for a, b in PARES:
        if a < len(perm) and b < len(perm):
            perm[a], perm[b] = b, a
    KP = KP[:, :, perm]
    SC = SC[:, :, perm]
    d["KP"] = KP[:, [1, 0]]
    d["SC"] = SC[:, [1, 0]]
    for k in ("B", "Be"):
        b = d[k].copy()
        x1 = 1.0 - b[:, :, 2]
        x2 = 1.0 - b[:, :, 0]
        b[:, :, 0], b[:, :, 2] = x1, x2
        d[k] = b[:, [1, 0]]
    for k in ("pres", "origen", "hueco"):
        if k in d:
            d[k] = d[k][:, [1, 0]]
    np.savez_compressed(destino / ("poses_%s_ESPEJO.npz" % clip), **d)
    return "%s_ESPEJO" % clip


def prueba_espejo_rasgos(clips, tol=1e-4):
    """QUE DEVUELVE: (clips comprobados, clips que fallan, peor discrepancia y en que rasgo).

    Los canales de EVENTOS se excluyen de la comparacion y se dice por que: los eventos los escribe
    la etapa 7 leyendo el clip real, y el clip espejo sintetico no tiene serie propia, asi que
    compararlos midiria la ausencia del fichero y no el extractor.
    """
    nombres = (ET.RASGOS_PROPIOS[:12] + ET.RASGOS_COMPARTIDOS)
    n_cmp = len(nombres)
    tmp = RU.TRABAJO
    peor, peor_n, mal, n = 0.0, "", [], 0
    for c in clips:
        r = ET.rasgos_clip(c)
        if r is None:
            continue
        e = espeja_npz(c, tmp)
        re = ET.rasgos_clip(e, fps=r["fps"])
        """La cadencia se le pasa a mano porque el clip espejo es sintetico y no tiene ni video ni
        CSV de la etapa 7. Eso mismo fue lo primero que esta prueba encontro: sin pasarla, el
        espejo cogia el 25 por defecto y TODAS sus velocidades salian 1,2 veces las del original,
        que es exactamente 30/25. El arreglo de verdad fue que la cadencia viaje dentro del NPZ."""
        (tmp / ("poses_%s.npz" % e)).unlink(missing_ok=True)
        if re is None:
            continue
        n += 1
        a = r["prop"][:, :, :n_cmp]
        b = re["prop"][[1, 0], :, :n_cmp]
        d = np.abs(a - b)
        esc = np.maximum(np.abs(a).max(axis=(0, 1)), 1e-6)
        rel = d.max(axis=(0, 1)) / esc
        if rel.max() > peor:
            peor, peor_n = float(rel.max()), nombres[int(rel.argmax())]
        if rel.max() > tol:
            mal.append((c, nombres[int(rel.argmax())], float(rel.max())))
    return n, mal, peor, peor_n


def tabla(filas, mod, ev, cabeza):
    import torch
    Pre, Are, Sre, yre = ev["tensores"]
    ire = ev["ire"]
    y = yre.numpy()
    p = ET.predice(mod, Pre, Are, Sre, cabeza)
    clases = ET.CLASES[cabeza]
    acc = float((p == y).mean())
    ytr = np.array([filas[i]["y_" + cabeza] for i in ev["itr"]])
    may, icl = ET.base_mayoritaria(ytr, y)
    techo = CA.techo_humano()[cabeza]

    print()
    print("CABEZA: %s" % cabeza)
    print("reservado: %d clips de %d torneos que NO estan en el entrenamiento"
          % (len(y), ev["diag"]["torneos_re"]))
    print()
    print("%-46s %8s %10s %10s" % ("", "acierto", "cobertura", "acierto*"))
    print("-" * 80)
    print("%-46s %8.3f %10.3f %10.3f"
          % ("linea base: clase mayoritaria (%s)" % clases[icl], may, 1.0, may))
    if cabeza == "initiated":
        a, cob, ac = ET.base_brazo_primero(filas, ire)
        print("%-46s %8.3f %10.3f %10.3f"
              % ("linea base: quien extiende el brazo primero", a, cob, ac))
    if cabeza == "strip_location":
        a, cob, ac = ET.base_x_pista(filas, ire)
        print("%-46s %8.3f %10.3f %10.3f"
              % ("linea base: NUESTRA x medida de la pista", a, cob, ac))
    print("%-46s %8.3f %10.3f %10.3f" % ("EL ETIQUETADOR", acc, 1.0, acc))
    print("%-46s %8.3f %10s %10s"
          % ("techo: acuerdo entre anotadores (n=%d)" % techo["n"], techo["acuerdo"], "-", "-"))
    print()
    print("* 'acierto' es sobre los clips que la linea base RESPONDE; 'acierto*' cuenta los mudos")
    print("  como fallo. Dar solo el primero de una linea base con cobertura 0,3 seria trampa.")
    print()
    print("matriz de confusion (filas = verdad, columnas = prediccion):")
    M = ET.matriz(y, p, clases)
    print("%-20s %s" % ("", " ".join("%6s" % c[:6] for c in clases)))
    for i, c in enumerate(clases):
        print("%-20s %s   n=%d" % (c, " ".join("%6d" % v for v in M[i]), M[i].sum()))
    if cabeza == "strip_location":
        d = np.abs(p - y)
        print()
        print("error ORDINAL medio: %.3f zonas   (exacto %.3f, a una zona o menos %.3f)"
              % (d.mean(), (d == 0).mean(), (d <= 1).mean()))
        print("Una clasificacion plana no distinguiria estas tres cifras, y son la razon de ser")
        print("de la cabeza ordinal.")
    return acc, may


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabeza", default="initiated", choices=ET.CABEZAS)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--espejo", type=int, default=12)
    ap.add_argument("--minimo", type=int, default=60,
                    help="clips minimos para entrenar. Bajarlo NO produce una cifra "
                         "interpretable: sirve para comprobar que la maquinaria corre.")
    a = ap.parse_args()

    print("CAREO DEL ETIQUETADOR")
    print("=" * 80)
    filas = ET.reune(cabeza=a.cabeza)
    if not filas:
        print("no hay ningun clip con rasgos Y etiqueta: falta extraer.")
        return 2

    print()
    print("1. LA ANTISIMETRIA EN LOS RASGOS, CON UN ESPEJO DE VERDAD")
    n, mal, peor, peor_n = prueba_espejo_rasgos([f["clip"] for f in filas[:a.espejo]])
    print("   clips espejados: %d   con discrepancia: %d" % (n, len(mal)))
    print("   peor discrepancia relativa: %.2e  (rasgo `%s`)" % (peor, peor_n))
    for c, r, v in mal[:5]:
        print("   MAL  %-38s rasgo `%s` %.3e" % (c, r, v))
    if mal:
        print("   -> un rasgo del bloque propio depende del lado de la imagen. La antisimetria")
        print("      NO esta impuesta y el acierto de la cabeza 1 no se puede leer.")

    if len(filas) < a.minimo:
        print()
        print("2-4. NO SE ENTRENA: solo hay %d clips con rasgos y etiqueta (minimo %d)."
              % (len(filas), a.minimo))
        print("     La puerta [C] necesita extraccion. No se da ningun numero de acierto,")
        print("     porque un acierto sobre %d clips no distingue nada de nada." % len(filas))
        print("     Para extraer:  python 01_extraccion/lotes.py --conjunto disco --etapas 1,4,7")
        return 0
    if len(filas) < 300:
        print()
        print("AVISO DE DENOMINADOR: %d clips. El reservado por torneo se queda en unas decenas,"
              % len(filas))
        print("y con ese n la diferencia entre el modelo y su linea base NO es distinguible del")
        print("azar. Lo de abajo prueba que la MAQUINARIA corre; no es la puerta [C].")

    mod, ev = ET.entrena(filas, a.cabeza, epocas=a.epocas)

    print()
    print("2. EL MODELO INVIERTE LA RESPUESTA AL ESPEJAR")
    Pre, _A, Sre, _y = ev["tensores"]
    if a.cabeza == "initiated":
        frac, n_mal = ET.control_espejo(mod, Pre, Sre)
        print("   respuestas invertidas: %.4f  (%d clips no invierten de %d)"
              % (frac, n_mal, len(Pre)))
        if frac < 1.0:
            print("   -> FALLO DE CONSTRUCCION. El numero de abajo no vale.")
    else:
        print("   (la cabeza %s no es antisimetrica por ranuras; su simetria es la impar de la"
              " zona)" % a.cabeza)

    print()
    print("3-4. LINEAS BASE Y ACIERTO")
    tabla(filas, mod, ev, a.cabeza)
    return 0


if __name__ == "__main__":
    sys.exit(main())
