"""El careo de la mano armada contra la FICHA FEDERATIVA, que es verdad de referencia de verdad.

## EL PROBLEMA QUE ESTO ATACA

La mano armada falla a menudo, y el limite no es el criterio: es el **denominador**. Toda la
cascada se eligio y se midio contra **59 tiradores anotados a ojo** por un agente mirando mosaicos.
Con 59 casos, la diferencia entre 0,90 y 0,95 son **tres tiradores**, asi que no se puede decidir
casi nada -- y de hecho ya paso: una variante se descarto porque los 2 tiradores que anadia
estaban **los dos en entrenamiento**.

La FIE declara la mano de cada licencia y el catalogo trae los dos nombres de cada clip. Medido:
**970 de 1.100** tiradores del conjunto anotado resuelven sin ambiguedad, y `hand` viene poblado en
el **100 %** de la muestra. Eso es una verdad de referencia **mas de diez veces mayor**, gratis.

## LO QUE SE COMPRUEBA, EN ESTE ORDEN, Y POR QUE ESE ORDEN

  0  **La linea base de azar informado.** Si el 75 % de los tiradores son diestros, contestar
     siempre "la diestra" acierta 0,75 sin mirar la imagen. Esta linea base **no existia**, porque
     no habia datos de lateralidad, y cambia como se leen todas las cifras publicadas.

  1  **El sentido del mapeo `A`/`B` contra `R`/`L`, MEDIDO y no supuesto.** La FIE habla de manos
     anatomicas; el proyecto habla de indices del modelo de pose (`A` = raiz 21, del hombro 6;
     `B` = raiz 42, del hombro 5). Por el convenio de paridad *deberia* ser `A` = diestra, pero eso
     es una hipotesis. Se prueban los dos sentidos contra los 59 anotados a ojo y gana el que mas
     coincide. **Si los dos dan parecido, el mapeo no esta determinado y esta verdad no se puede
     usar** -- y eso tambien seria un resultado.

  2  **Cuanto coincide la ficha con el ojo**, en el solape. Son dos verdades independientes: una
     mira la imagen y la otra mira la licencia. Donde discrepen hay que mirar el clip, porque o la
     anotacion a ojo se equivoco o la tuberia intercambio los dos tiradores.

  3  **La cascada y cada peldano contra la ficha**, con acierto y **cobertura por separado**.

## LO QUE ESTA VERDAD NO ES

Es la mano de la **licencia**, no la del clip; y **no valida quien es quien**. Comprueba "de las
dos manos de esta persona, cual es la armada" **dado** que la persona de la izquierda es la que el
catalogo llama `fotl_name`. Si la tuberia intercambiase los dos tiradores, la ficha confirmaria la
asignacion equivocada. Por eso el paso 2 existe y va antes que el 3.

    python verifica_mano_fie.py
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import fie as FIE  # noqa: E402
import rutas as RU  # noqa: E402

VERDAD_OJO = RU.MUESTRAS / "mano_dominante" / "verdad_mano.csv"

MARGEN_MAPEO = 0.15
"""Cuanto tiene que ganar un sentido del mapeo al otro para darlo por determinado. Con 0,50 contra
0,50 los dos sentidos son igual de buenos, o sea que la ficha no aporta nada; se exige que el
ganador saque al perdedor al menos esto."""


def lee_ojo():
    """QUE DEVUELVE: {(clip, lado): 'A'|'B'}. Los `?` se dejan fuera: el anotador escribio que no
    se decidia, y convertir eso en una etiqueta seria inventarla."""
    out = {}
    if not VERDAD_OJO.is_file():
        return out
    with open(VERDAD_OJO, encoding="utf-8") as fh:
        for f in csv.DictReader(l for l in fh if not l.startswith("#")):
            if f.get("mano") in ("A", "B"):
                out[(f["clip"], int(f["lado"]))] = f["mano"]
    return out


def mano_fie_de(clip, lado, df, mf):
    """QUE DEVUELVE: 'R' | 'L' | None para el tirador de ese lado de ese clip."""
    f = df.loc[df["clip_id"] == clip]
    if f.empty:
        return None
    nom = f.iloc[0]["fotl_name" if lado == 0 else "fotr_name"]
    return mf.get(FIE.norm(nom))


def main():
    df = CA.manifiesto()
    mf = FIE.manos()
    print("CAREO DE LA MANO ARMADA CONTRA LA FICHA FEDERATIVA")
    print("=" * 92)
    if not mf:
        print("no hay %s. Se construye con:" % FIE.MANOS)
        print("   python 01_extraccion/codigo/fie.py --censo")
        print("   python 01_extraccion/codigo/fie.py --manos")
        return 2

    print()
    print("0. LA LINEA BASE DE AZAR INFORMADO, QUE NO EXISTIA")
    fr, n = FIE.base_diestros()
    print("   tiradores con mano declarada        : %d" % n)
    print("   diestros %.3f   zurdos %.3f" % (fr, 1 - fr))
    print("   -> contestar SIEMPRE 'la diestra' acierta **%.3f** sin mirar la imagen" % fr)
    ok_pob = 0.20 <= (1 - fr) <= 0.40
    print("   zurdos en 20-40 %% (esgrima de elite): %s" % ("si" % () if ok_pob else "NO -- "
          "sospechar del campo antes que del deporte"))
    print()
    print("   Con esa linea base al lado, las cifras publicadas se leen asi:")
    print("      contestar siempre la diestra          %.3f   <- NUEVO" % fr)
    print("      peldano geometrico solo               0.741 (20 de 27), 0.831 sobre 59")
    print("      la cascada entera                     0.949 sobre 59, 1.000 sobre 19 reservados")
    if fr > 0.741:
        print("   -> el peldano geometrico suelto **NO bate al azar informado**.")

    print()
    print("1. EL SENTIDO DEL MAPEO A/B <-> R/L, MEDIDO CONTRA LOS 59 ANOTADOS A OJO")
    ojo = lee_ojo()
    pares = []
    for (clip, lado), ab in ojo.items():
        h = mano_fie_de(clip, lado, df, mf)
        if h:
            pares.append((ab, h))
    if not pares:
        print("   0 tiradores con anotacion a ojo Y ficha: no se puede calibrar.")
        print("   (la anotacion a ojo cubre 17 clips; hace falta que sus nombres resuelvan)")
        return 0
    a_es_r = sum(1 for ab, h in pares if (ab == "A") == (h == "R")) / len(pares)
    print("   tiradores con las DOS verdades      : %d" % len(pares))
    print("   hipotesis  A = diestra              : %.3f de acuerdo" % a_es_r)
    print("   hipotesis  A = zurda                : %.3f de acuerdo" % (1 - a_es_r))
    gana = "A=diestra" if a_es_r >= 0.5 else "A=zurda"
    marg = abs(2 * a_es_r - 1)
    print("   gana %s por %.3f" % (gana, marg))
    if marg < MARGEN_MAPEO:
        print("   -> EL MAPEO NO ESTA DETERMINADO (margen < %.2f). La ficha NO se puede usar como"
              % MARGEN_MAPEO)
        print("      verdad de A/B hasta saber por que, y eso es un resultado, no un contratiempo.")
        return 0
    print("   -> por el convenio de paridad del modelo (`A` cuelga del hombro 6) lo esperado era")
    print("      A = diestra; la medida %s." % ("lo confirma" if a_es_r >= 0.5 else "lo CONTRADICE"))

    print()
    print("2. CUANTO COINCIDEN LAS DOS VERDADES, QUE SON INDEPENDIENTES")
    disc = [(c, l) for (c, l), ab in ojo.items()
            if (h := mano_fie_de(c, l, df, mf)) and ((ab == "A") != (h == "R")) == (a_es_r >= 0.5)]
    print("   coincidencia ojo vs ficha           : %.3f (%d de %d)"
          % (max(a_es_r, 1 - a_es_r), int(round(max(a_es_r, 1 - a_es_r) * len(pares))), len(pares)))
    print("   discrepan en %d tiradores, y hay que MIRARLOS: o el ojo se equivoco, o la tuberia"
          % len(disc))
    print("   intercambio los dos tiradores de ese clip, que es un fallo mas grave.")
    por_clip = collections.Counter(c for c, _l in disc)
    for c, l in disc[:8]:
        print("      %-40s lado %d%s"
              % (c, l, "   <- discrepa en LOS DOS lados" if por_clip[c] == 2 else ""))
    dobles = [c for c, n in por_clip.items() if n == 2]
    if dobles:
        print()
        print("   %d clip(s) discrepan en LOS DOS lados. Esa seria la FIRMA DE UN INTERCAMBIO"
              % len(dobles))
        print("   de tiradores, y hay que descartarlo mirando si los nombres cambiados arreglan")
        print("   los dos lados. Si no los arreglan, son dos errores y no un intercambio.")
        for c in dobles:
            f = df.loc[df["clip_id"] == c]
            if f.empty:
                continue
            r = f.iloc[0]
            hl = mf.get(FIE.norm(r["fotl_name"]))
            hr = mf.get(FIE.norm(r["fotr_name"]))
            e0, e1 = ojo.get((c, 0)), ojo.get((c, 1))
            if not (hl and hr and e0 and e1):
                continue
            def ok(ab, h):
                return (ab == "A") == (h == "R") if a_es_r >= 0.5 else (ab == "A") != (h == "R")
            tal = sum((ok(e0, hl), ok(e1, hr)))
            cam = sum((ok(e0, hr), ok(e1, hl)))
            print("      %-34s tal cual %d/2, con los nombres CAMBIADOS %d/2  -> %s"
                  % (c[:34], tal, cam,
                     "INTERCAMBIO" if cam > tal else "no es intercambio, son dos errores"))
        print("   Un intercambio invierte la etiqueta de TODO lo que cuelgue de ese clip -- quien")
        print("   inicia, la zona, el cono -- asi que vale mas que un fallo de mano.")

    print()
    print("3. LA CASCADA CONTRA LA FICHA, POR PELDANO")
    import cono_lib as CL
    import glob
    hay = sorted(pathlib.Path(p).name[6:-4]
                 for p in glob.glob(str(RU.TRABAJO / "poses_*.npz")))
    evaluables = []
    for clip in hay:
        for lado in (0, 1):
            h = mano_fie_de(clip, lado, df, mf)
            if h:
                evaluables.append((clip, lado, h))
    print("   clips con poses extraidas           : %d" % len(hay))
    print("   tiradores con poses Y ficha         : %d" % len(evaluables))
    if len(evaluables) < 30:
        print("   -> con menos de 30 no se da ninguna cifra: el careo necesita extraccion.")
        print("      python 01_extraccion/lotes.py --conjunto disco --etapas 1,4")
        return 0

    por_peldano = collections.defaultdict(lambda: [0, 0])
    glob_ok = glob_n = 0
    for clip, lado, h in evaluables:
        try:
            with np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True) as z:
                KP, SC, B = z["KP"], z["SC"], z["B"]
                W, H = int(z["W"]), int(z["H"])
            mano, d = CL.mano_armada(KP, SC, B, W, H, lado)
        except Exception:
            continue
        if mano not in ("A", "B"):
            continue
        esperado = "A" if ((h == "R") == (a_es_r >= 0.5)) else "B"
        glob_n += 1
        acierta = (mano == esperado)
        glob_ok += acierta
        p = por_peldano[d.get("por", "?")]
        p[0] += acierta
        p[1] += 1
    print()
    print("   %-26s %8s %8s %10s" % ("peldano que decidio", "acierto", "n", "cobertura"))
    print("   " + "-" * 58)
    for k in sorted(por_peldano, key=lambda k: -por_peldano[k][1]):
        ok, tot = por_peldano[k]
        print("   %-26s %8.3f %8d %10.3f" % (k, ok / max(tot, 1), tot, tot / max(glob_n, 1)))
    print("   " + "-" * 58)
    print("   %-26s %8.3f %8d %10.3f" % ("LA CASCADA ENTERA", glob_ok / max(glob_n, 1),
                                         glob_n, 1.0))
    print("   %-26s %8.3f %8s %10s" % ("azar informado (siempre R)", fr, "-", "-"))
    print()
    print("   Acierto y cobertura van en columnas distintas a proposito: un peldano que decide 12")
    print("   de 19 sin fallar y otro que decide 19 de 19 con 0,895 no se comparan por el primero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
