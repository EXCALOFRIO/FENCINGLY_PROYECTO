"""La FICHA FEDERATIVA de cada tirador: la mano que declara la FIE, para 66.544 esgrimistas.

## POR QUE ESTO EXISTE

La mano armada se decide hoy con una cascada de criterios geometricos, y **falla a menudo**. El
limite no es el criterio: es la **verdad de referencia**. Toda la cascada se eligio y se midio
contra **59 tiradores anotados a ojo por un agente mirando mosaicos**, en 33 clips. Con 59 casos,
la diferencia entre 0,90 y 0,95 son tres tiradores y no se puede decidir nada.

La FIE publica la mano de cada licencia. Los nombres de los dos tiradores estan en el catalogo de
los 362.950 clips. Cruzarlos da una verdad de referencia **mas de diez veces mayor y sin anotar
nada**.

## LO MEDIDO ANTES DE CONSTRUIR NADA, que es lo que decide si la idea es real

    el censo responde                    HTTP 200, 66.544 tiradores, 34 paginas de 2.000
    `hand` NO esta en el listado         hay que pedir el perfil individual, uno por tirador
    `hand` SI esta en el perfil          poblado en el **100 %** de una muestra de 60 de los
                                         NUESTROS (45 diestros, 15 zurdos)
    `height` casi nunca                  7 de 60. Para la estatura esto NO sirve

Y el casado por nombre, que es lo que de verdad estaba en duda:

    conjunto                    tiradores   casan sin ambiguedad   clips con LOS DOS
    anotados con etiqueta           1.100      970  (88,2 %)       4.272 de 5.856 (73,0 %)
    el corpus de 8.555                285      245  (86,0 %)       6.295 de 8.555 (73,6 %)
    florete entero                    826      649  (78,6 %)      80.921 de 106.432 (76,0 %)

O sea: **970 tiradores con la mano declarada por su federacion, contra los 59 anotados a ojo.**

## EL CONTROL QUE HACE QUE ESTO NO SEA UN 25 % DE REGALO

De la muestra de 60, **el 25 % son zurdos**. En la poblacion general los zurdos son ~10 %; en
esgrima de elite lo esperable es **25-35 %**, porque el zurdo tiene ventaja tactica. Que la cifra
caiga justo en esa banda es una comprobacion barata de que el campo trae dato y no un valor por
defecto: si `hand` viniera relleno con "R" por descuido, saldria 0 % de zurdos, y si viniera al
azar saldria 50 %.

**Y ESE MISMO 25 % ES UNA LINEA BASE QUE NADIE TENIA.** Si tres de cada cuatro tiradores son
diestros, *contestar siempre "la mano derecha"* acierta el **0,75** sin mirar la imagen. Con eso al
lado, las cifras publicadas de la mano armada se leen de otra manera:

    contestar siempre la mano diestra      ~0,75   <- el azar INFORMADO, nuevo
    el peldano geometrico solo              0,741 (20 de 27), 0,831 sobre 59
    la cascada entera                       0,949 sobre 59, 1,000 sobre los 19 reservados

El peldano geometrico suelto **no bate al azar informado**. La cascada si, pero por menos de lo
que parecia. Esta linea base no existia porque no habia datos de lateralidad; ahora los hay.

## LAS DOS COSAS QUE ESTA VERDAD **NO** ES, y hay que decirlas

**1. Es la mano de la LICENCIA, no la del clip.** Un tirador no cambia de mano en mitad de un
asalto, asi que como verdad por tirador es exactamente la forma correcta. Pero no observa el clip:
si la ficha estuviera mal, esto lo repetiria.

**2. No valida QUIEN es quien.** Comprueba "de las dos manos de esta persona, cual es la armada"
**dado** que la persona de la izquierda es el tirador que el catalogo llama `fotl_name`. Si la
tuberia intercambiase los dos tiradores, la mano de la FIE confirmaria alegremente la asignacion
equivocada. Es una verdad CONDICIONADA, y por eso el careo la contrasta primero contra los 59
anotados a ojo -- que si miran la imagen -- antes de usarla para nada.

## Y HAY QUE CALIBRAR EL SENTIDO ANTES DE USARLA

La FIE dice `R` o `L` en terminos anatomicos. El proyecto dice `A` (raiz de mano 21, del hombro 6)
o `B` (raiz 42, del hombro 5), que son indices del modelo de pose. Cual de las dos letras
corresponde a la derecha anatomica **se mide**, no se supone: se compara contra los 59 anotados a
ojo y gana el sentido que mas coincide. Si los dos sentidos dieran parecido, el mapeo no esta
determinado y esta verdad no se puede usar -- y eso tambien seria un resultado.

    python fie.py --censo        baja las 34 paginas (66.544) y las cachea
    python fie.py --manos        resuelve NUESTROS nombres y baja su `hand`
    python fie.py                imprime el estado de lo cacheado
"""
from __future__ import annotations

import collections
import csv
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rutas as RU  # noqa: E402

CENSO = RU.CATALOGO / "fie_censo.json.gz"
MANOS = RU.CATALOGO / "mano_fie.csv"
UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://fie.org/api/fie"

COLUMNAS = ["nombre_catalogo", "fie_id", "fie_nombre", "pais", "hand", "via_casado",
            "n_candidatos", "genero", "altura"]


def _pide(url, intentos=4, espera=2.0):
    """QUE HACE: una peticion con reintentos. QUE DEVUELVE: el json o None.

    None y no excepcion: un 404 en el tirador 300 de 1.000 no puede tumbar la corrida, y el
    recuento de los que fallaron es un dato que hay que dar, no un accidente que tapar."""
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if k == intentos - 1:
                return None
            time.sleep(espera * (k + 1))
    return None


def norm(s):
    """QUE HACE: normaliza un nombre para comparar. Quita acentos, pasa a mayusculas y convierte
    guiones, apostrofes y puntos en separadores.

    El catalogo trae `SHAMEL Youssef` y `LIN Yik Hei Coco`; la FIE trae `lastName` y `firstName`
    por separado. Los dos ordenes se indexan, porque el catalogo no siempre pone el apellido
    primero."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("-", " ").replace("'", " ").replace(".", " ")
    return " ".join(re.split(r"[\s,]+", s)).strip()


def baja_censo(destino=None, verboso=True):
    """QUE HACE: baja las 34 paginas de 2.000 del censo entero y lo guarda comprimido.

    El listado **no trae `hand`**: trae id, nombre, pais, genero y poco mas. Sirve para RESOLVER
    el nombre; la mano hay que pedirla perfil a perfil."""
    import gzip
    destino = pathlib.Path(destino or CENSO)
    items, pg = [], 1
    while True:
        d = _pide("%s/fencers-all?page=%d&pageSize=2000" % (BASE, pg))
        if not d:
            raise SystemExit("no se pudo bajar la pagina %d del censo" % pg)
        items.extend(d["items"])
        if verboso:
            print("pagina %2d  %6d de %d" % (pg, len(items), d["totalFound"]), flush=True)
        if len(items) >= d["totalFound"] or not d["items"]:
            break
        pg += 1
    destino.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destino, "wt", encoding="utf-8") as fh:
        json.dump(items, fh)
    if verboso:
        print("censo -> %s  (%d tiradores, %d bytes)"
              % (destino, len(items), destino.stat().st_size))
    return items


def censo(ruta=None):
    import gzip
    ruta = pathlib.Path(ruta or CENSO)
    if not ruta.is_file():
        raise FileNotFoundError("no esta %s. Se baja con: python fie.py --censo" % ruta)
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def indices(items):
    """QUE DEVUELVE: (indice exacto, indice laxo).

    El exacto es `apellido nombre` y tambien `nombre apellido`. El laxo es
    `apellido|primer nombre de pila`, y existe porque buena parte de los fallos son nombres
    compuestos escritos distinto o truncados: `CHAN Nok` contra `CHAN Nok Hin`. Medido: la regla
    laxa recupera **16 tiradores mas** de 1.100, o sea del 86,7 % al 88,2 %. Poco, y por eso va
    marcada en la columna `via_casado` en vez de mezclarse con el casado exacto."""
    ex = collections.defaultdict(list)
    lx = collections.defaultdict(list)
    for f in items:
        ape, nom = norm(f.get("lastName")), norm(f.get("firstName"))
        if not (ape and nom):
            continue
        ex[ape + " " + nom].append(f)
        ex[nom + " " + ape].append(f)
        lx[ape + "|" + nom.split()[0]].append(f)
    return ex, lx


def resuelve(nombre, ex, lx):
    """QUE DEVUELVE: (ficha, via, n_candidatos). `via` es `exacto`, `laxo` o `''`.

    **Un nombre con dos candidatos NO se resuelve.** Se devuelve la cuenta y se deja fuera: elegir
    uno de dos homonimos al azar meteria ruido en la verdad de referencia, que es justo lo que esto
    viene a arreglar."""
    n = norm(nombre)
    if n in ex:
        v = ex[n]
        return (v[0] if len(v) == 1 else None), "exacto", len(v)
    t = n.split()
    for k in range(len(t) - 1, 0, -1):
        c = " ".join(t[:k]) + "|" + t[k]
        if c in lx:
            v = lx[c]
            return (v[0] if len(v) == 1 else None), "laxo", len(v)
    return None, "", 0


def baja_manos(nombres, salida=None, verboso=True, pausa=0.15):
    """QUE HACE: resuelve cada nombre y pide su perfil para sacar `hand`. Escribe el CSV.

    Una peticion por tirador, no por clip: la mano es del tirador y no cambia. Para los 970 del
    conjunto anotado son 970 peticiones, unos cinco minutos."""
    it = censo()
    ex, lx = indices(it)
    salida = pathlib.Path(salida or MANOS)
    filas, sin_casar, ambiguos, sin_hand = [], 0, 0, 0
    res = []
    for nom in sorted(set(nombres)):
        f, via, n = resuelve(nom, ex, lx)
        if f is None:
            if n > 1:
                ambiguos += 1
            else:
                sin_casar += 1
            continue
        res.append((nom, f, via, n))
    if verboso:
        print("nombres %d: casan %d, ambiguos %d, sin casar %d"
              % (len(set(nombres)), len(res), ambiguos, sin_casar))
    for i, (nom, f, via, n) in enumerate(res):
        d = _pide("%s/fencer/%d" % (BASE, f["id"]))
        mano = (d or {}).get("hand") or ""
        if not mano:
            sin_hand += 1
        filas.append({"nombre_catalogo": nom, "fie_id": f["id"], "fie_nombre": f.get("name"),
                      "pais": f.get("countryCode"), "hand": mano, "via_casado": via,
                      "n_candidatos": n, "genero": f.get("gender"),
                      "altura": (d or {}).get("height") or ""})
        if verboso and (i + 1) % 100 == 0:
            print("   %4d de %d perfiles" % (i + 1, len(res)), flush=True)
        time.sleep(pausa)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    zur = sum(1 for f in filas if f["hand"] == "L")
    con = sum(1 for f in filas if f["hand"] in ("L", "R"))
    if verboso:
        print("mano_fie.csv: %d tiradores, %d con `hand` (%d sin), zurdos %d (%.1f %%)"
              % (len(filas), con, sin_hand, zur, 100.0 * zur / max(con, 1)))
        print("   la fraccion de zurdos tiene que caer en 25-35 %% (esgrima de elite). Fuera de")
        print("   ahi, sospechar del campo antes que del deporte.")
    return filas


def manos(ruta=None):
    """QUE DEVUELVE: dict nombre normalizado -> 'L' | 'R'."""
    ruta = pathlib.Path(ruta or MANOS)
    if not ruta.is_file():
        return {}
    out = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["hand"] in ("L", "R"):
                out[f["nombre_catalogo"]] = f["hand"]
    return out


def nombres_de(df=None, filtro=None):
    """QUE DEVUELVE: los nombres normalizados de los dos tiradores de los clips que pasen el
    filtro."""
    import catalogo as CA
    df = CA.manifiesto() if df is None else df
    if filtro:
        df = df.query(filtro)
    out = set()
    for c in ("fotl_name", "fotr_name"):
        out |= {norm(v) for v in df[c] if v}
    return out


def base_diestros(ruta=None):
    """QUE DEVUELVE: (fraccion de diestros, n). **Es la linea base de azar informado** para la mano
    armada: contestar siempre "la diestra" acierta eso sin mirar la imagen."""
    m = manos(ruta)
    if not m:
        return float("nan"), 0
    d = sum(1 for v in m.values() if v == "R")
    return d / len(m), len(m)


def estado():
    print("censo   : %s" % (CENSO if CENSO.is_file() else "NO bajado (python fie.py --censo)"))
    if CENSO.is_file():
        print("          %d tiradores, %d bytes" % (len(censo()), CENSO.stat().st_size))
    print("manos   : %s" % (MANOS if MANOS.is_file() else "NO bajado (python fie.py --manos)"))
    if MANOS.is_file():
        m = manos()
        fr, n = base_diestros()
        print("          %d tiradores con mano declarada" % n)
        print("          diestros %.3f  zurdos %.3f" % (fr, 1 - fr))
        print("          -> LINEA BASE de azar informado para la mano armada: %.3f" % fr)


if __name__ == "__main__":
    if "--censo" in sys.argv:
        baja_censo()
    elif "--manos" in sys.argv:
        import catalogo as CA
        df = CA.manifiesto()
        m = (df["initiated"] != "") | (df["strip_location"] != "") | (df["body_location"] != "")
        if "--corpus" in sys.argv:
            m = m | (df["weapon"] == "foil")
        baja_manos(nombres_de(df.loc[m]))
    else:
        estado()
