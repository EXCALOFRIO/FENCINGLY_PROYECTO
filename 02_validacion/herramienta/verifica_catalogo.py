"""El careo del manifiesto: cada cifra contra lo que tiene que reproducir, y su denominador.

QUE COMPRUEBA Y POR QUE ESO. El manifiesto es un derivado de dos ficheros de texto, y un derivado
mal construido no falla: sale, y sale con numeros plausibles. Asi que aqui no se comprueba que el
fichero exista, se comprueba que **reproduce recuentos que se sabian de antes** y que las tres
intersecciones que deciden el trabajo de esta tanda dan lo que tienen que dar.

Cada linea de la tabla lleva su n y su denominador. La razon esta escrita en el proyecto cinco
veces: el mismo nombre de medida da numeros distintos segun sobre que se divida, y el caso de esta
tanda es "1.370 clips anotados ya estan en disco", que es cierto con un denominador (el corpus ya
procesado) y falso con otro (el disco: son 3.654).

    python verifica_catalogo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as C  # noqa: E402

CORPUS = (r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop"
          r"\187cfd83-d8d9-4c4d-bed4-f255724f6b21\scratchpad\corpus_etiquetado.txt")
"""El fichero que el manifiesto sustituye. Vive FUERA del repo, en un scratchpad temporal de otra
sesion, que es exactamente el defecto que se esta cerrando. Si ya no esta, el careo lo dice y sigue
con el resto en vez de caerse: el dia que ese directorio se borre, esta comprobacion se pierde y
hay que saberlo, no descubrirlo."""


def _fila(ok, nombre, obtenido, esperado, nota=""):
    marca = "ok  " if ok else "MAL "
    print("%s %-52s %12s   esperado %-12s %s"
          % (marca, nombre, obtenido, esperado, nota))
    return 1 if ok else 0


def main():
    df = C.manifiesto()
    mal = 0
    print("CAREO DEL MANIFIESTO  (%s)" % C.MANIFIESTO.name)
    print("=" * 108)
    print()
    print("1. EL CATALOGO SE REPRODUCE ENTERO")
    mal += 1 - _fila(len(df) == 362950, "clips en el manifiesto", len(df), 362950)
    mal += 1 - _fila(df["torneo"].nunique() == 172, "torneos", df["torneo"].nunique(), 172)
    mal += 1 - _fila(df["clip_id"].is_unique, "clip_id unico", df["clip_id"].is_unique, True)
    arma = df["weapon"].value_counts().to_dict()
    for a, esp in (("foil", 106432), ("epee", 140639), ("sabre", 102618)):
        mal += 1 - _fila(arma.get(a) == esp, "clips de %s" % a, arma.get(a, 0), esp)
    mal += 1 - _fila(arma.get("champs") == 13260,
                     "clips con weapon = champs (NO es un arma)", arma.get("champs", 0), 13260,
                     "los 106.432 de florete son COTA INFERIOR")

    print()
    print("2. `touch` VIENE GRATIS PARA LOS 362.950, SIN MODELO")
    t = df["touch"].value_counts().to_dict()
    for a, esp in (("right", 173640), ("left", 165194), ("double", 24106)):
        mal += 1 - _fila(t.get(a) == esp, "touch = %s" % a, t.get(a, 0), esp)

    print()
    print("3. LA ANOTACION HUMANA, CON EL CONVENIO DECLARADO (mayoria, empates fuera)")
    for col, esp in (("initiated", 5596), ("strip_location", 5544), ("body_location", 1699)):
        n = int((df[col] != "").sum())
        mal += 1 - _fila(n == esp, "%s con etiqueta" % col, n, esp, "de 362.950 clips")
    an = C.anotados()
    mal += 1 - _fila(len(an) == 5856, "clips con ALGUNA de las tres", len(an), 5856,
                     "de 6.748 con fila; 976 la tienen vacia")

    print()
    print("4. EL TECHO HUMANO: LO QUE EL ETIQUETADOR NO PUEDE BATIR SIN SOSPECHA")
    techo = C.techo_humano()
    for col, esp_n, esp_a in (("initiated", 911, 0.688), ("strip_location", 908, 0.619),
                              ("body_location", 108, 0.481)):
        h = techo[col]
        ok = h["n"] == esp_n and abs(h["acuerdo"] - esp_a) < 0.001
        mal += 1 - _fila(ok, "acuerdo entre anotadores, %s" % col,
                         "%.3f (n=%d)" % (h["acuerdo"], h["n"]),
                         "%.3f (n=%d)" % (esp_a, esp_n),
                         "clase mayoritaria %.2f por abajo" % (
                             df.loc[df[col] != "", col].value_counts(normalize=True).iloc[0]))

    print()
    print("5. EL CORPUS DE 8.555 ESTA DENTRO, Y ES TODO FLORETE")
    if os.path.exists(CORPUS):
        ids = set(l.strip() for l in open(CORPUS, encoding="utf-8") if l.strip())
        m = df.set_index("clip_id")
        hay = ids & set(m.index)
        mal += 1 - _fila(len(hay) == 8555, "ids del corpus que estan en el catalogo",
                         len(hay), 8555, "de %d del fichero" % len(ids))
        armas = set(m.loc[sorted(hay), "weapon"])
        mal += 1 - _fila(armas == {"foil"}, "armas del corpus", sorted(armas), "['foil']")
    else:
        print("--   corpus_etiquetado.txt ya no esta en %s" % CORPUS)
        print("     (era un scratchpad temporal: por eso el manifiesto lo sustituye)")

    print()
    print("6. LOS TRES DENOMINADORES DE 'YA ESTA EN DISCO', QUE DAN TRES NUMEROS")
    disco = C.en_disco_ahora()
    print("     mp4 distintos bajo %s: %d" % (C.VIDEOS, len(disco)))
    if os.path.exists(CORPUS):
        ids = set(l.strip() for l in open(CORPUS, encoding="utf-8") if l.strip())
        ano_fila = set(df.loc[df["n_anot"] > 0, "clip_id"])
        print("     anotados (fila que casa) x corpus ya procesado : %5d   <- el 1.370 del plan"
              % len(ano_fila & ids))
        print("     con etiqueta             x corpus ya procesado : %5d" % len(set(an["clip_id"]) & ids))
    print("     anotados (fila que casa) x disco                : %5d"
          % len(set(df.loc[df["n_anot"] > 0, "clip_id"]) & disco))
    print("     con etiqueta             x disco                : %5d   <- lo que [C] puede usar"
          % len(set(an["clip_id"]) & disco))
    print("     con etiqueta SIN mp4 (lo que habria que bajar)  : %5d   ~%.1f GB a 3,5 MB/clip"
          % (len(set(an["clip_id"]) - disco), 3.5 * len(set(an["clip_id"]) - disco) / 1024))

    print()
    print("7. Y LA TRAMPA: 'EL MP4 ESTA' NO ES 'EL CLIP ESTA EXTRAIDO'")
    import glob
    pru = r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas"
    ext = set(os.path.basename(f)[6:-4] for f in glob.glob(os.path.join(pru, "poses_*.npz")))
    print("     clips con poses_<clip>.npz en 01_extraccion/pruebas : %d" % len(ext))
    print("     de ellos anotados                                   : %d" % len(ext & set(an["clip_id"])))
    print("     -> el coste de [C] NO es bajar video, es EXTRAER")

    print()
    print("=" * 108)
    print("lineas comprobadas mal: %d" % mal)
    return 1 if mal else 0


if __name__ == "__main__":
    sys.exit(main())
