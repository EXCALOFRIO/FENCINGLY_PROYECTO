"""El CATALOGO de clips y la ANOTACION HUMANA: un solo lector para las dos cosas.

QUE HAY AQUI Y DE DONDE SALE. Dos ficheros publicos de `clips.fencingdatabase.com`, volcados de
psql en texto. Se guardan en `03_datos/catalogo/` con su instantanea apuntada, porque un fichero
servido por un CDN cambia sin avisar y una cifra calculada sobre "el catalogo" sin decir cual es
una cifra sin denominador:

    gfycats.txt         103.444.712 B  ETag ad2e66d0735e81d9e9ed1639afc231bb
                        Last-Modified  Fri, 11 Sep 2026 16:05:48 GMT
                        362.950 clips, 172 torneos. En el repo va COMPRIMIDO
                        (`gfycats.txt.gz`, 9.398.476 B): 103 MB pasa del limite duro
                        de 100 MB por fichero de GitHub y el repo se espeja alli.

    form_responses.txt    1.133.242 B  ETag cab79689db8c2c909e68d763e2f5afb4
                        Last-Modified  Fri, 11 Sep 2026 11:23:16 GMT
                        8.513 filas de anotacion HUMANA. Va tal cual.

LO QUE TRAE CADA UNO. El catalogo trae, por clip: `weapon`, `url`, los dos nombres, el marcador,
`touch` (que lado toco), `event_type` y `status`. La anotacion trae, por clip anotado: `initiated`
(quien inicia), `strip_location` (zona de la pista), `body_location` (blanco tocado) y la fecha.

`touch` ES GRATIS PARA LOS 362.950. No necesita modelo ninguno: ya esta en el catalogo. Medido:
173.640 right, 165.194 left, 24.106 double, 9 unknown, 1 vacio.

## EL DEFECTO DE LA COLUMNA `weapon`, MEDIDO

`weapon` NO siempre trae un arma. Recuento exacto de los 362.950:

    epee    140.639        foil     106.432        sabre   102.618
    champs   13.260        (vacio)        1

`champs` no es un arma: son los seis torneos de campeonato (`worldchamps2026` 6.867,
`asianchamps2026` 3.890, `africanchamps2026` 1.414, `eurochamps2026` 956, `asianchamps2025` 123,
`eurochamps2025` 10) donde la columna trae la familia del torneo en vez del arma. Y hay una fila
con `weapon` y `url` vacios (`heidenheim2017/FoolishSmoothGreatargus.mp4`, linea 123.482).

**La consecuencia, que hay que decir cada vez que se cite el 106.432:** los 106.432 de florete son
una COTA INFERIOR. Hay 13.261 clips mas cuya arma el catalogo no dice, y alguno de ellos sera
florete. El manifiesto los guarda con `weapon` tal como viene, para que nadie los cuente como
florete por descuido, y `ARMAS` solo contiene las tres declaradas.

## LA NORMALIZACION, Y EL NULO QUE NO SE IMPUTA

En `form_responses.txt` conviven `FOTL`/`fotl` (160 y 3.210) y `FOTR`/`fotr` (168 y 3.050). Se
normaliza a minusculas al leer, y aqui queda dicho.

Las filas con la etiqueta VACIA **no son un valor**: son ausencia de anotacion. `initiated` tiene
1.323 vacias y ademas una clase real `neither` (602) -- son cosas distintas y confundirlas
inventaria 1.323 ejemplos de una clase que el anotador no eligio. Asi que el vacio se deja vacio,
`etiquetas()` no lo devuelve, y el entrenamiento lo excluye en vez de imputarlo.

Recuento de las tres etiquetas sobre las 8.513 filas del fichero:

    initiated        fotl 3.370   fotr 3.218   neither 602            sin anotar 1.323
    strip_location   middle 2.875  fotl_half 1.469  fotr_half 1.344
                     fotl_warning_box 995  fotr_warning_box 520       sin anotar 1.310
    body_location    torso 1.063  front_arm 496  hand 293  head 136
                     front_leg 56  foot 46  back_arm 14  back_leg 12  sin anotar 6.397

## 8.513 FILAS NO SON 8.513 CLIPS, Y EL CONVENIO DECIDE LA CIFRA

Medido: las 8.513 filas son **6.829 clips distintos**; 1.184 estan anotados mas de una vez, uno de
ellos **35 veces**. De los que casan con el catalogo, 6.748 clips en 7.925 filas.

Asi que hay que resolver los repetidos, y el convenio elegido cambia el numero. Los tres medidos,
sobre los 6.748:

    convenio                          initiated  strip_location  body_location
    la ultima fila gana                   5.628           5.635          1.698
    cualquiera no vacia                   5.803           5.800          1.734
    MAYORIA, empates fuera                5.596           5.544          1.699

**Se usa MAYORIA con los empates fuera**, y queda escrito aqui. Usa toda la anotacion en vez de
tirar la mitad, y donde los anotadores empatan no inventa una respuesta: deja el clip sin etiqueta
para esa cabeza. Los empates que se dejan fuera son 207 de `initiated`, 256 de `strip_location` y
35 de `body_location`.

Nota para quien vuelva a este numero: **las cifras 5.628 / 5.635 / 1.698 del plan de esta tanda son
el convenio "la ultima fila gana"**, que nadie eligio -- es lo que sale de sobrescribir un dict
mientras se lee. No estan mal, estan sin declarar; las de aqui son las del convenio declarado.

## EL TECHO: LOS ANOTADORES NO SE PONEN DE ACUERDO, Y ESO SE MIDE

Lo mejor que trae el fichero repetido no es mas etiqueta: es saber **cuanto se puede acertar**.
Sobre los clips con dos o mas respuestas independientes, la fraccion en que TODAS coinciden:

    cabeza            clips con >=2 respuestas   en desacuerdo   acuerdo
    initiated                              911             284     0,688
    strip_location                         908             346     0,619
    body_location                          108              56     0,481

**Eso es el techo del etiquetador, no su objetivo.** Un modelo que saque 0,69 en `initiated`
contra estas etiquetas ha igualado a los humanos; uno que saque 0,80 hay que mirarlo con
desconfianza antes que con alegria, porque estaria acertando mas que la propia verdad de
referencia. Y el 0,51 de la clase mayoritaria tiene ahora, por fin, una cota por arriba al lado.

Por eso el manifiesto lleva dos columnas mas de las que pedia el plan: `n_anot` (cuantas filas de
anotacion tiene el clip) y `acuerdo` (si todas las no vacias coincidian). Sin ellas la unica manera
de volver a saber esto seria releer el txt, que es justo lo que este fichero existe para evitar.

## POR QUE ESTE FICHERO ES EL UNICO LECTOR

Antes de esto, `corpus_etiquetado.txt` vivia FUERA del repo, en el scratchpad temporal de otra
sesion, y unos diez ficheros rederivaban esa ruta cada uno por su cuenta. Un scratchpad temporal se
borra. El manifiesto lo sustituye, entra en el repo, y nadie vuelve a leer los `.txt` directamente:
si hace falta un campo nuevo se anade aqui, una vez.

## EL CAMPO `en_disco` NO ES UNA PROPIEDAD DEL CLIP

Es el estado de `D:/FENCINGLY_VIDEOS_RAW` el dia que se construyo el manifiesto. En Colab no
significa nada. Para preguntar por el disco de AHORA esta `en_disco_ahora()`, que recorre el arbol
y devuelve el conjunto. El campo del CSV sirve para reproducir las cuentas de esta tanda, no para
decidir en otra maquina.
"""
from __future__ import annotations

import collections
import csv
import gzip
import io
import os
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
CAT = RAIZ / "03_datos" / "catalogo"
GFY = CAT / "gfycats.txt.gz"
FORM = CAT / "form_responses.txt"
MANIFIESTO = CAT / "manifiesto.csv.gz"

def _videos():
    """QUE DEVUELVE: la raiz de los videos, preguntandosela a `rutas`, que es quien la sabe.

    Aqui habia una ruta absoluta repetida. Repetir una ruta no es un pecado de estilo: es que
    cuando cambia, cambia en un sitio y no en el otro, y el que se queda viejo **no da error** --
    devuelve vacio. `rutas.py` es el unico fichero del proyecto donde una ruta absoluta es
    correcta, porque su trabajo es concentrarlas."""
    try:
        import rutas as RU
        return str(RU.VIDEOS)
    except Exception:
        return os.environ.get("FENCINGLY_VIDEOS", "")


VIDEOS = _videos()

BASE_URL = "https://clips.fencingdatabase.com"

ARMAS = ("foil", "epee", "sabre")
"""Las tres que el catalogo declara. `champs` (13.260) y el vacio (1) NO estan aqui a proposito:
no son armas, son arma DESCONOCIDA."""

COLUMNAS = [
    "clip_id", "weapon", "torneo", "url",
    "fotl_name", "fotr_name", "left_score", "right_score",
    "touch", "event_type", "status",
    "initiated", "strip_location", "body_location", "en_disco",
    "n_anot", "acuerdo",
]
"""Las quince del plan, mas dos que la medida obligo a anadir: `n_anot` (filas de anotacion de ese
clip) y `acuerdo` (1 si todas las no vacias coincidian, 0 si no, vacio si solo hay una). Sin ellas
el techo de 0,688 / 0,619 / 0,481 no se puede recalcular sin releer el txt."""

ZONAS = ["fotl_warning_box", "fotl_half", "middle", "fotr_half", "fotr_warning_box"]
"""ORDINALES, de izquierda a derecha. El orden es el dato: confundir `middle` con `fotr_half` no
cuesta lo mismo que confundirlo con `fotr_warning_box`, y una clasificacion plana trata los dos
errores igual. Quien use esto como cinco clases sueltas esta tirando informacion."""

BLANCOS = ["torso", "front_arm", "back_arm", "hand", "head", "front_leg", "back_leg", "foot"]

INICIA = ["fotl", "fotr", "neither"]

_MAN = None


def _abre(ruta, modo="rt"):
    """QUE HACE: abre en texto, comprimido o no, decidiendo por la extension."""
    ruta = str(ruta)
    if ruta.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(ruta, modo.replace("t", "b")),
                                encoding="utf-8", errors="replace",
                                newline="" if "w" in modo else None)
    return open(ruta, modo, encoding="utf-8", errors="replace",
                newline="" if "w" in modo else None)


def filas_psql(ruta):
    """QUE HACE: lee un volcado de psql en texto. Cabecera, raya de guiones, filas separadas por
    ' | ' con relleno de espacios, y un pie '(N rows)'.

    POR QUE NO EL MODULO `csv`: el separador es una barra rodeada de relleno, no una coma. Se
    comprueba que cada fila trae EXACTAMENTE tantos campos como la cabecera; la que no, se cuenta
    aparte y no se cuela. Las lineas en blanco se saltan -- `form_responses.txt` acaba con una, y
    contarla daba 8.514 filas donde el pie dice 8.513.

    QUE DEVUELVE: (dict de la fila, n) y al final ('__pie__', {texto, filas, raras}).
    """
    with _abre(ruta) as fh:
        cols = [c.strip() for c in next(fh).split("|")]
        next(fh)
        n = 0
        raras = 0
        for ln in fh:
            ln = ln.rstrip("\n").rstrip("\r")
            if not ln.strip():
                continue
            if ln.startswith("(") and ln.rstrip().endswith("rows)"):
                yield "__pie__", {"texto": ln.strip(), "filas": n, "raras": raras}
                return
            p = ln.split("|")
            if len(p) != len(cols):
                raras += 1
                continue
            n += 1
            yield dict(zip(cols, (x.strip() for x in p))), n
        yield "__pie__", {"texto": "(sin pie)", "filas": n, "raras": raras}


def _id_y_torneo(u):
    """QUE HACE: saca `clip_id` y torneo de la url. Comprobado sobre los 362.950: la unica que no
    tiene la forma `<base>/<torneo>/<clip>.mp4` es la fila con la url vacia."""
    if not u.startswith(BASE_URL + "/") or not u.endswith(".mp4"):
        return None, None
    resto = u[len(BASE_URL) + 1:-4]
    if "/" not in resto:
        return None, None
    torneo, cid = resto.split("/", 1)
    return cid, torneo


def url(clip, torneo):
    """QUE DEVUELVE: la url del mp4 en el CDN. Es lo que `lotes.py` descarga en Colab."""
    return "%s/%s/%s.mp4" % (BASE_URL, torneo, clip)


def lee_catalogo(ruta=None):
    """QUE DEVUELVE: dict clip_id -> campos del catalogo. Los ids son unicos: medido, 0 duplicados
    en 362.950. La clave `__sin_url__` trae cuantas filas se descartaron por url no canonica."""
    ruta = ruta or GFY
    out = {}
    sin_url = 0
    for fila, _n in filas_psql(ruta):
        if fila == "__pie__":
            break
        cid, torneo = _id_y_torneo(fila["url"])
        if cid is None:
            sin_url += 1
            continue
        out[cid] = {
            "weapon": fila["weapon"],
            "torneo": torneo,
            "url": fila["url"],
            "fotl_name": fila["left_fencer_name"],
            "fotr_name": fila["right_fencer_name"],
            "left_score": fila["left_score"],
            "right_score": fila["right_score"],
            "touch": fila["touch"],
            "event_type": fila["event_type"],
            "status": fila["status"],
        }
    out["__sin_url__"] = sin_url
    return out


ETIQUETAS = ("initiated", "strip_location", "body_location")


def _resuelve(valores):
    """QUE HACE: resuelve las respuestas de VARIOS anotadores sobre la misma etiqueta.

    EL CRITERIO, DECLARADO: mayoria simple entre las respuestas no vacias; **empate -> sin
    etiqueta**. Una vacia no vota, porque no responder no es responder `neither`.

    QUE DEVUELVE: (valor, n_respuestas_no_vacias, coinciden_todas).
    `coinciden_todas` es None cuando solo hay una respuesta: con una sola no hay acuerdo ni
    desacuerdo que medir, y meterla en el denominador del techo lo inflaria hasta 0,97.
    """
    nov = [v for v in valores if v]
    if not nov:
        return "", 0, None
    if len(nov) == 1:
        return nov[0], 1, None
    cnt = collections.Counter(nov).most_common()
    coinciden = len(cnt) == 1
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        return "", len(nov), coinciden
    return cnt[0][0], len(nov), coinciden


def lee_anotacion(ruta=None):
    """QUE DEVUELVE: dict clip_id -> {initiated, strip_location, body_location, n_anot, acuerdo}.

    Ya en minusculas y ya resuelto por mayoria (ver `_resuelve`). El vacio se queda vacio: NO se
    convierte en `neither` ni en nada. `n_anot` son las FILAS del clip en el fichero, que es lo que
    hace visible que 8.513 filas son 6.829 clips.
    """
    ruta = ruta or FORM
    crudo = collections.defaultdict(list)
    for fila, _n in filas_psql(ruta):
        if fila == "__pie__":
            break
        cid = fila["stats_id"]
        if not cid:
            continue
        crudo[cid].append({k: fila[k].strip().lower() for k in ETIQUETAS})
    out = {}
    for cid, vs in crudo.items():
        d = {"n_anot": len(vs)}
        acuerdos = []
        for col in ETIQUETAS:
            val, _n, coinciden = _resuelve([v[col] for v in vs])
            d[col] = val
            if coinciden is not None:
                acuerdos.append(coinciden)
        d["acuerdo"] = "" if not acuerdos else ("1" if all(acuerdos) else "0")
        out[cid] = d
    return out


def techo_humano(ruta=None, ids=None, solo_catalogo=True):
    """QUE DEVUELVE: por cabeza, el acuerdo entre anotadores -- el TECHO del etiquetador.

    DOS DENOMINADORES QUE HAY QUE FIJAR, o esto da cuatro numeros distintos:

    1. Se mide **solo sobre los clips con dos o mas respuestas no vacias** a ESA cabeza. Meter los
       de una sola respuesta daria un acuerdo aparente casi perfecto sin haber medido nada.
    2. Se mide **solo sobre los clips que estan en el catalogo** (`solo_catalogo=True`), que son
       los unicos entrenables. Sobre las 6.829 filas enteras sale 0,689 / 0,619 / 0,491 con
       n = 916 / 914 / 110; sobre los 6.748 del catalogo sale **0,688 / 0,619 / 0,481** con
       n = 911 / 908 / 108. La diferencia son cinco clips, y se declara porque es justo el tamano
       de discrepancia que luego nadie sabe explicar.
    """
    ruta = ruta or FORM
    if ids is None and solo_catalogo:
        cat = lee_catalogo()
        cat.pop("__sin_url__", None)
        ids = set(cat)
    crudo = collections.defaultdict(list)
    for fila, _n in filas_psql(ruta):
        if fila == "__pie__":
            break
        cid = fila["stats_id"]
        if not cid or (ids is not None and cid not in ids):
            continue
        crudo[cid].append({k: fila[k].strip().lower() for k in ETIQUETAS})
    out = {}
    for col in ETIQUETAS:
        n = d = 0
        for vs in crudo.values():
            nov = [v[col] for v in vs if v[col]]
            if len(nov) < 2:
                continue
            n += 1
            if len(set(nov)) > 1:
                d += 1
        out[col] = {"n": n, "desacuerdo": d, "acuerdo": (1 - d / n) if n else None}
    return out


def en_disco_ahora(raiz=None):
    """QUE DEVUELVE: el conjunto de clip_id con un .mp4 bajo `raiz`, AHORA. Recorre el arbol, asi
    que tarda; se usa para construir el manifiesto y para comprobarlo, no dentro de un bucle."""
    raiz = pathlib.Path(raiz or VIDEOS)
    vistos = set()
    if not raiz.is_dir():
        return vistos
    for d, _sub, fs in os.walk(raiz):
        for f in fs:
            if f.lower().endswith(".mp4"):
                vistos.add(f[:-4])
    return vistos


def construye(salida=None, videos=None, verboso=True):
    """QUE HACE: escribe el manifiesto, una fila por clip del catalogo, con la anotacion humana
    pegada cuando existe y `en_disco` mirado en el momento.

    QUE NO HACE: inventar. Un clip sin anotacion sale con las tres etiquetas vacias; un clip con
    `weapon = champs` sale con `champs`, no con una suposicion.
    """
    salida = pathlib.Path(salida or MANIFIESTO)
    cat = lee_catalogo()
    sin_url = cat.pop("__sin_url__", 0)
    ano = lee_anotacion()
    disco = en_disco_ahora(videos)
    if verboso:
        print("catalogo %d clips (%d sin url), anotacion %d filas, en disco %d mp4"
              % (len(cat), sin_url, len(ano), len(disco)))
    salida.parent.mkdir(parents=True, exist_ok=True)
    casan = 0
    with _abre(salida, "wt") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNAS)
        for cid in sorted(cat):
            d = cat[cid]
            e = ano.get(cid)
            if e:
                casan += 1
            w.writerow([
                cid, d["weapon"], d["torneo"], d["url"],
                d["fotl_name"], d["fotr_name"], d["left_score"], d["right_score"],
                d["touch"], d["event_type"], d["status"],
                (e or {}).get("initiated", ""),
                (e or {}).get("strip_location", ""),
                (e or {}).get("body_location", ""),
                "1" if cid in disco else "0",
                (e or {}).get("n_anot", 0),
                (e or {}).get("acuerdo", ""),
            ])
    if verboso:
        print("manifiesto: %s  (%d filas, %d con anotacion, %d bytes)"
              % (salida, len(cat), casan, salida.stat().st_size))
        print("anotacion que NO casa con el catalogo: %d de %d"
              % (len(ano) - casan, len(ano)))
    return {"clips": len(cat), "anotados": casan, "en_disco": len(disco),
            "sin_url": sin_url, "anotacion": len(ano)}


def manifiesto(recarga=False):
    """QUE DEVUELVE: el manifiesto entero como DataFrame de pandas, cacheado en el proceso.

    `en_disco` viene como booleano; los marcadores como enteros con NA donde el catalogo no trae
    numero. Las tres etiquetas humanas vienen como texto, con cadena vacia donde no hay anotacion:
    vacio significa SIN ANOTAR, no `neither`.
    """
    global _MAN
    if _MAN is not None and not recarga:
        return _MAN
    import pandas as pd
    if not pathlib.Path(MANIFIESTO).exists():
        raise FileNotFoundError(
            "no esta %s. Se construye con: python catalogo.py --construye" % MANIFIESTO)
    df = pd.read_csv(MANIFIESTO, dtype=str, keep_default_na=False, na_values=[])
    for c in ("left_score", "right_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["en_disco"] = df["en_disco"] == "1"
    df["n_anot"] = pd.to_numeric(df["n_anot"], errors="coerce").fillna(0).astype(int)
    _MAN = df
    return df


def etiquetas(clip):
    """QUE DEVUELVE: las etiquetas HUMANAS de un clip, solo las que existen, o `None` si el clip no
    esta anotado. Una etiqueta vacia NO sale en el dict: si no esta la clave, no hay anotacion, y
    eso es distinto de que el anotador dijese `neither`.

    Lo que sale de aqui lleva `origen_etiqueta = 'humana'` y **nunca** se mezcla con lo que prediga
    el etiquetador. Los clips anotados a mano no se reetiquetan.
    """
    df = manifiesto()
    f = df.loc[df["clip_id"] == clip]
    if f.empty:
        return None
    r = f.iloc[0]
    out = {k: r[k] for k in ("initiated", "strip_location", "body_location") if r[k]}
    if not out:
        return None
    out["origen_etiqueta"] = "humana"
    out["clip_id"] = clip
    out["torneo"] = r["torneo"]
    out["weapon"] = r["weapon"]
    return out


def lote(filtro=None, n=None, semilla=20260922):
    """QUE DEVUELVE: un subconjunto reproducible del manifiesto.

    `filtro` es una consulta de pandas (`"weapon == 'foil' and en_disco"`) o un invocable que
    recibe el DataFrame y devuelve otro. `n` sortea sin reemplazo con `semilla`; sin `n` devuelve
    todo lo que pase el filtro. Siempre ORDENADO por clip_id, que es lo que hace que dos maquinas
    produzcan la misma lista y que `--desde/--cuantos` signifiquen lo mismo tras una caida.
    """
    df = manifiesto()
    if callable(filtro):
        df = filtro(df)
    elif filtro:
        df = df.query(filtro)
    df = df.sort_values("clip_id")
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=semilla).sort_values("clip_id")
    return df.reset_index(drop=True)


def anotados(arma=None, etiqueta=None, solo_en_disco=False):
    """QUE DEVUELVE: los clips con anotacion humana.

    `etiqueta` restringe a los que tienen ESA etiqueta rellena, que es el denominador correcto para
    entrenar esa cabeza y no otro: `initiated` tiene 5.628 y `body_location` 1.698, y usar "6.748
    anotados" como denominador de las tres seria contar tres veces cosas distintas.
    """
    df = manifiesto()
    m = (df["initiated"] != "") | (df["strip_location"] != "") | (df["body_location"] != "")
    if etiqueta:
        m = df[etiqueta] != ""
    if arma:
        m = m & (df["weapon"] == arma)
    if solo_en_disco:
        m = m & df["en_disco"]
    return df.loc[m].sort_values("clip_id").reset_index(drop=True)


def resumen():
    """QUE HACE: imprime los recuentos con los que se verifica el manifiesto. Cada cifra con su
    denominador, porque el mismo nombre da numeros distintos segun sobre que se divida."""
    df = manifiesto()
    print("clips en el manifiesto: %d" % len(df))
    print("torneos: %d" % df["torneo"].nunique())
    print()
    print("por arma DECLARADA:")
    for a, k in df["weapon"].value_counts().items():
        marca = "" if a in ARMAS else "   <- NO es un arma"
        print("   %-10s %7d%s" % (a or "(vacio)", k, marca))
    print()
    print("touch, gratis para todos, sin modelo:")
    for a, k in df["touch"].value_counts().items():
        print("   %-10s %7d" % (a or "(vacio)", k))
    print()
    for col in ("initiated", "strip_location", "body_location"):
        con = df[col] != ""
        print("%s: %d anotados de %d clips del catalogo" % (col, int(con.sum()), len(df)))
        for a, k in df.loc[con, col].value_counts().items():
            print("   %-20s %6d" % (a, k))
        print()
    an = anotados()
    print("clips con ALGUNA etiqueta humana: %d" % len(an))
    print("   en disco: %d      por bajar: %d"
          % (int(an["en_disco"].sum()), int((~an["en_disco"]).sum())))
    print("   por arma: %s" % dict(an["weapon"].value_counts()))
    print()
    foil = df.loc[df["weapon"] == "foil"]
    print("florete declarado: %d, en disco %d" % (len(foil), int(foil["en_disco"].sum())))


if __name__ == "__main__":
    if "--construye" in sys.argv:
        construye()
    else:
        resumen()
