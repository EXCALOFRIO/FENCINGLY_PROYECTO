"""Sube a Hugging Face la evidencia visual que se va verificando, para tenerla documentada fuera.

LO QUE PIDIO EL USUARIO, literal: "podrias con el token que tienes de huggingface ir subiendo todo
para poderlo tener ahi documentado, todas estas figuras y videos que estamos verificando, ya las
ultimas de las pruebas estas que te digo que hagas, las mas finas, por ejemplo la ultima que te pedi
que estaban muy bien porfa, y metes lo de la mascara esta que en teoria ibas a arreglar, y luego
subes lo nuevo de los floretes cuando lo tengas, lo de la pista etc, y lo vas dejando ahi por tenerlo
subido, ejemplos de lo que vamos haciendo".

Asi que esto se corre VARIAS VECES, cada vez que hay material nuevo, y tiene que ser incremental.
Hugging Face compara por hash: un fichero que no ha cambiado no se vuelve a subir.

## AL REPO PRIVADO, Y ESTO NO ES UNA PREFERENCIA

En la cuenta hay dos repos de datos:
  `EXCALORIO/fencingly-extraccion`   privado
  `EXCALORIO/fencingly-datos`        PUBLICO

Todo esto va al PRIVADO. Los clips y las figuras muestran a **atletas identificables en competicion**
con la cara visible, y eso no puede acabar en un repo publico ni indexado. La comprobacion no se
delega a la memoria de nadie: antes de subir, este script pregunta al servidor si el repo es privado
y **se niega a subir si no lo es**.

## EL TOKEN NO SE IMPRIME NI SE ESCRIBE EN NINGUN SITIO

Se lee de `HF_TOKEN` si esta en el entorno y, si no, de la clave `HUGGINFACE` del `.env` del
Escritorio. Se pasa a la libreria y nada mas: no aparece en pantalla, ni en el indice, ni en un log.

## COMO SE CORRE

  python sube_evidencia.py                 sube todos los lotes que existan
  python sube_evidencia.py galeria capas   solo esos lotes
  python sube_evidencia.py --lista         dice que hay ya subido y no sube nada

Interprete: el de cuda, que es el que tiene `huggingface_hub`.
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO = "EXCALORIO/fencingly-extraccion"
TIPO = "dataset"
VAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion")
COD = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion")

LOTES = {
    "galeria": {
        "local": VAL / "poses_inventadas",
        "repo": "evidencia/poses_inventadas",
        "patron": ["*.png"],
        "que": "Poses INVENTADAS, una imagen por caso, en tres paneles: ancla medida antes, la pose "
               "inventada en violeta sobre los pixeles reales, y ancla medida despues. Es lo que "
               "permite juzgar a ojo si lo inventado es una continuacion creible o esta puesto donde "
               "no hay nadie. El rotulo lleva el largo del hueco y el error esperado a ese largo.",
    },
    "cascada": {
        "local": VAL / "muestras" / "cascada_deteccion",
        "repo": "evidencia/cascada_deteccion",
        "patron": ["*.png"],
        "que": "COMO SE ENCUENTRA UNA POSE, RONDA A RONDA, con la cifra de cada una contada de los "
               "NPZ entregados: ronda 0 YOLO en el cuadro (96,92 %), ronda 1 YOLO en un RECORTE "
               "-- la redeteccion -- (1,15 %), ronda 2 la cabeza de pose de Sapiens 2 top-down "
               "(50 lados-cuadro, solo con hueco <= 24) y ronda 3 interpolar (2,73 %, y marcado). "
               "Incluye casos reales de redeteccion en dos paneles: el cuadro entero con la caja "
               "donde el detector no encontro a nadie, y el recorte donde la pose si salio.",
    },
    "tres_rondas": {
        "local": VAL / "poses_tres_rondas",
        "repo": "evidencia/poses_tres_rondas",
        "patron": ["*.png"],
        "que": "LA CASCADA DE TRES RONDAS, el MISMO cuadro resuelto por cada via, uno al lado de "
               "otro: (1) la ronda 1 que NO detecto, con su caja estimada en violeta y ninguna "
               "pose; (2) la ronda 2, la cabeza de pose de Sapiens 2 top-down sobre esa caja, con "
               "si pasa el filtro y con cuantos puntos, tambien cuando falla; (3) la ronda 3, la "
               "pose interpolada que se entrega hoy, con el error esperado a ese largo de hueco. "
               "Solo con hueco <= 24, porque con hueco >= 25 el filtro pasa 0,6 % con un tirador "
               "delante y 12,7 % sobre fondo vacio, o sea que ahi no discrimina. En el rotulo va "
               "ese falso positivo por tramo, para que «pasa el filtro» se pueda pesar.",
    },
    "videos": {
        "local": VAL / "videos_revision",
        "repo": "evidencia/videos_revision",
        "patron": ["*.mp4"],
        "que": "Videos de revision con las dos poses de 308 puntos pintadas: verde el tirador de la "
               "izquierda, naranja el de la derecha, violeta y rotuladas las poses inventadas, banda "
               "roja en los cuadros que la puerta de planos descarta (primer plano o repeticion).",
    },
    "capas": {
        "local": VAL / "muestras" / "capas",
        "repo": "evidencia/capas_densas",
        "patron": ["*.png", "*.jpg"],
        "que": "Capas densas de Sapiens 2 sobre los tiradores: segmentacion, normales, pointmap y "
               "matting. Son las capas que hay que limpiar para que no entre el arbitro, el publico "
               "ni los tiradores de las pistas del fondo.",
    },
    "capas_sapiens": {
        "local": VAL / "muestras" / "capas_sapiens",
        "repo": "evidencia/capas_densas_sapiens",
        "patron": ["*.png", "*.jpg"],
        "que": "Mas ejemplos de las cabezas densas de Sapiens 2, cuadro completo, antes de limpiar.",
    },
    "arma": {
        "local": VAL / "muestras" / "arma",
        "repo": "evidencia/florete",
        "patron": ["*.png", "*.jpg"],
        "que": "Deteccion del florete. El filo se ve en la capa `matting` con alfa muy bajo (0,0006), "
               "que es lo que permite sacar la hoja aunque en la imagen normal casi no exista.",
    },
    "cono": {
        "local": VAL / "muestras" / "cono",
        "repo": "evidencia/cono_del_florete",
        "patron": ["*.png", "*.jpg"],
        "que": "Region de amenaza del florete y su interseccion con el rival. El eje sale de la "
               "HOJA cuando se ve en el alfa del matting y, cuando no -- que es la inmensa "
               "mayoria de los cuadros --, del EJE DEL BRAZO (hombro -> raiz de la mano) con una "
               "envolvente que cubre el error medido de esa suposicion. El eje del PULGAR se "
               "probo y se descarto: la palanca son 2,4-4,8 px y dos estimadores de la misma "
               "recta discrepaban 67-134 grados.",
    },
    "planos": {
        "local": VAL / "muestras" / "planos",
        "repo": "evidencia/puerta_de_planos",
        "patron": ["*.png", "*.jpg"],
        "que": "Primeros planos y repeticiones que la puerta descarta, para poder mirar a ojo si "
               "descarta lo que debe y solo lo que debe.",
    },
    "zoom": {
        "local": VAL / "muestras" / "zoom",
        "repo": "evidencia/fundidos_y_zoom",
        "patron": ["*.png", "*.jpg"],
        "que": "Fundidos encadenados y zooms. Los dos dan la misma firma de mezcla de pixeles y lo "
               "que los separa es el residuo relativo al contraste (fundido 0,219, zoom 0,062).",
    },
    "redeteccion": {
        "local": VAL / "muestras" / "redeteccion",
        "repo": "evidencia/redeteccion_en_recorte",
        "patron": ["*.png", "*.jpg"],
        "que": "Cuadros donde faltaba uno de los dos y se recupero buscandolo en un recorte alrededor "
               "de su posicion estimada.",
    },
    "pareja": {
        "local": VAL / "muestras" / "ab_pareja",
        "repo": "evidencia/eleccion_de_pareja",
        "patron": ["*.png", "*.jpg"],
        "que": "Eleccion de la pareja FILMADA cuando en el plano hay varias pistas, arbitro y publico. "
               "Es la averia mas grave y la que mas se ha trabajado.",
    },
    "sospechosos": {
        "local": VAL / "muestras" / "sospechosos",
        "repo": "evidencia/casos_sospechosos",
        "patron": ["*.png", "*.jpg"],
        "que": "Casos que el buscador automatico de errores marco como sospechosos, con su sintoma: "
               "salto de caja, cajas que solapan, separacion imposible, o faltan cuadros.",
    },
    "pista": {
        "local": VAL / "pista",
        "repo": "evidencia/pista",
        "patron": ["*.png", "*.jpg", "*.csv", "*.mp4", "*.md"],
        "que": "Pista detectada y posicion de cada tirador en METROS sobre ella, con la trayectoria en "
               "2D (largo x ancho) y los eventos marcados: marcha, fondo, extension de brazo.",
    },
    "prioridad": {
        "local": VAL / "prioridad",
        "repo": "evidencia/prioridad",
        "patron": ["*.png", "*.jpg", "*.mp4", "*.csv", "*.md"],
        "que": "Prioridad segun el reglamento FIE, cuadro a cuadro, con la regla que la concede.",
    },
    "mosaicos": {
        "local": VAL / "mosaicos",
        "repo": "evidencia/mosaicos",
        "patron": ["*.mp4", "*.png"],
        "que": "Mosaico de seis celdas por clip: original con esqueleto, matting, segmentacion, "
               "normales, pointmap y la deteccion del florete, todo a la vez y sincronizado.",
    },
    "intrusos": {
        "local": VAL / "muestras" / "limite_intruso",
        "repo": "evidencia/intrusos_verificados",
        "patron": ["*.png"],
        "que": "Los casos que quedan AL LIMITE de las seniales de intruso, verificados a ojo uno por "
               "uno. Estan aqui los dos errores reales encontrados (el arbitro elegido como tirador "
               "en `PowerfulAllLarva` y el trozo de espectador del borde en `LavishDistantDikkops`) "
               "y tambien los CORRECTOS que cada regla candidata se llevaba de mas, que son los que "
               "impidieron convertir la blancura en una puerta automatica.",
    },
    "cruce": {
        "local": VAL / "muestras" / "cruce_lados",
        "repo": "evidencia/cruce_de_lados",
        "patron": ["*.png"],
        "que": "El clip donde el `tirador izquierdo` entregado era en realidad el ARBITRO, cuadro a "
               "cuadro. Es el caso que destapo que las tres puertas geometricas no pueden distinguir "
               "a un arbitro que camina junto a la pista.",
    },
    "capas_limpias": {
        "local": VAL / "muestras" / "capas_limpias",
        "repo": "evidencia/capas_limpias",
        "patron": ["*.png", "*.jpg"],
        "que": "Las cuatro capas densas ANTES y DESPUES de limpiarlas, para ver que solo quedan los "
               "dos floretistas y que el arbitro, el publico y las pistas del fondo desaparecen.",
    },
    "cono3d": {
        "local": VAL / "muestras" / "cono3d",
        "repo": "evidencia/cono_3d",
        "patron": ["*.png", "*.jpg"],
        "que": "El cono de amenaza del florete en 3D, con el eje sacado de la hoja vista en el alfa "
               "del matting cuando la hay y del eje del brazo cuando no, mas los casos donde el 2D "
               "y el 3D no dicen lo mismo. NO se usa el eje del pulgar: esa via se midio y se "
               "descarto.",
    },
    "cono_arreglado": {
        "local": VAL / "muestras" / "cono_arreglado",
        "repo": "evidencia/cono_arreglado",
        "patron": ["*.png", "*.jpg"],
        "que": "ANTES y DESPUES del cono en el MISMO fotograma, tras los tres defectos que senialo el "
               "usuario. Lo que parecia un florete recto ERA EL BRAZO, pintado del color del cono. El "
               "largo pasa de 0,90 m (la hoja desde la guarda) a 1,10 m (el alcance muniera-punta, "
               "que es de donde sale el cono). Y la apertura estaba al REVES de lo que parecia: los "
               "23,1 grados contenian solo el 22,6 % de las hojas medidas. El rotulo declara el "
               "convenio, que es semiangulo: la envolvente son +-33 grados, o sea 66 de cono "
               "completo, y por eso se veia gorda.",
    },
    "mano_dominante": {
        "local": VAL / "muestras" / "mano_dominante",
        "repo": "evidencia/mano_dominante",
        "patron": ["*.png", "*.jpg", "*.csv"],
        "que": "La mano ARMADA elegida por tirador, con la senial del guante. El criterio viejo era "
               "`la mas adelantada` y el usuario aviso de que fallaba a menudo. El guante es una "
               "senial independiente de la geometria, porque es obligatorio en la mano armada y la "
               "otra va desnuda.",
    },
    "cono_cono": {
        "local": VAL / "muestras" / "cono_cono",
        "repo": "evidencia/cono_contra_cono",
        "patron": ["*.png", "*.jpg", "*.csv"],
        "que": "La interseccion del cono de un tirador con el cono del OTRO, que es donde puede "
               "haber CONTACTO ENTRE LAS DOS HOJAS, con la region de solape sombreada. Es el "
               "planteamiento correcto y lo corrigio el usuario: la version anterior cruzaba el cono "
               "contra el TORSO del rival, que contesta la pregunta de la amenaza pero no esta. "
               "Interesa porque el contacto de hojas es la senial que falta para detectar PARADAS.",
    },
    "contacto": {
        "local": VAL / "muestras" / "contacto_hojas",
        "repo": "evidencia/contacto_de_hojas",
        "patron": ["*.png", "*.jpg", "*.csv"],
        "que": "Laminas anotadas A CIEGAS (sin conos, sin nombre de clip, en orden aleatorio) para "
               "intentar validar el solape contra verdad vista a ojo. El resultado es negativo y por "
               "eso estan aqui: de 16 laminas, CERO decidibles como `si hay contacto`. Los monitores "
               "del marcador tapan la region del cruce y el desenfoque de movimiento cae justo en "
               "los cuadros que importan. O sea que el contacto de hojas **no es verificable a ojo "
               "en este material**, y el proxy se queda como columna cruda sin validar.",
    },
    "checks": {
        "local": VAL,
        "repo": "documentacion",
        "patron": ["CHECKS.md", "DONDE_ESTA_TODO.md", "RESULTADOS.md", "DATOS_POR_CLIP.md"],
        "que": "La tabla de CHECKS: cada garantia de la tuberia con su umbral declarado, lo medido, "
               "el veredicto, el sitio del codigo que la cumple y donde esta la evidencia.",
    },
    "documentacion": {
        "local": pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO"),
        "repo": "documentacion",
        "patron": ["00_EMPIEZA_AQUI.md"],
        "que": "El documento maestro del proyecto, con todo lo medido y las averias resueltas.",
    },
}
"""Cada lote es una carpeta local, su sitio en el repo, los patrones que se suben y una frase de que
es. La frase NO es decoracion: acaba en el indice del repo, y una figura sin saber que se mira no es
evidencia de nada. Los lotes cuya carpeta local no existe todavia se saltan en silencio, para que
este mismo fichero valga cuando lleguen la pista y la prioridad."""


def _token():
    """QUE HACE: devuelve el token sin imprimirlo ni dejarlo escrito en ningun sitio."""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    p = pathlib.Path(r"C:\Users\Alejandro\Desktop\.env")
    if p.exists():
        for linea in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in linea:
                continue
            k, v = linea.split("=", 1)
            if k.strip().upper() in ("HUGGINFACE", "HUGGINGFACE", "HF_TOKEN"):
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    raise SystemExit("no hay token: ni en el entorno ni en la clave HUGGINFACE del .env")


def _mide(carpeta, patrones):
    fs = []
    for pat in patrones:
        fs.extend(carpeta.glob(pat))
        fs.extend(carpeta.glob("**/" + pat))
    fs = sorted(set(f for f in fs if f.is_file()))
    return fs, sum(f.stat().st_size for f in fs)


def indice(subidos):
    """QUE HACE: el `README.md` del repo, que es lo que hace que esto sea documentacion y no un
    vertedero de PNG."""
    l = ["---", "license: other", "pretty_name: FENCINGLY evidencia de extraccion", "---", "",
         "# FENCINGLY, evidencia visual de la extraccion", "",
         "Repo **privado**. Contiene fotogramas y clips de competicion con **atletas "
         "identificables**, asi que no se hace publico ni se comparte fuera del proyecto.", "",
         "Cada carpeta es un paso de la tuberia de extraccion, y cada figura esta aqui para poder "
         "**contradecir al algoritmo a ojo**: una cifra agregada de presencia del 99 % no dice si la "
         "pareja elegida es la correcta, y una imagen si.", "",
         "## Codigo de color, el mismo en todo el material", "",
         "| color | que es |", "|---|---|", "| verde | el tirador de la IZQUIERDA |",
         "| naranja | el tirador de la DERECHA |",
         "| violeta | pose INVENTADA (interpolada o sostenida), con la etiqueta puesta |",
         "| banda roja | cuadro descartado: primer plano o repeticion |", "",
         "## Que hay", ""]
    for nom, n, mb, que, dest in subidos:
        l.append("### `%s`" % dest)
        l.append("")
        l.append(que)
        l.append("")
        l.append("%d ficheros, %.1f MB." % (n, mb))
        l.append("")
    l.append("## Lo que este material NO dice")
    l.append("")
    l.append("Las poses inventadas se validan escondiendo poses **que si se conocen**, y ahi el error "
             "va del 2,8 % del torso con un cuadro de hueco al 25,7 % con 24. Pero los huecos que de "
             "verdad se rellenan son en buena parte cuadros donde el tirador **no esta en el plano**, "
             "y para esos no existe verdad con la que comparar: su error es, por construccion, no "
             "verificable. Por eso cada punto lleva su marca de origen (0 medido, 1 interpolado, "
             "2 sostenido, 3 redetectado) y esa marca no se debe perder.")
    l.append("")
    return "\n".join(l)


def main():
    from huggingface_hub import HfApi
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    solo_lista = "--lista" in sys.argv
    api = HfApi(token=_token())

    info = api.repo_info(repo_id=REPO, repo_type=TIPO)
    if not getattr(info, "private", False):
        raise SystemExit("ME NIEGO A SUBIR: %s no es privado y esto lleva atletas identificables"
                         % REPO)
    print("repo %s, privado, %s" % (REPO, TIPO))
    ya = set(api.list_repo_files(repo_id=REPO, repo_type=TIPO))
    print("ficheros que ya hay en el repo: %d" % len(ya))
    print()

    nombres = args if args else list(LOTES)
    plan = []
    for nom in nombres:
        if nom not in LOTES:
            print("   lote desconocido, se salta: %s" % nom)
            continue
        c = LOTES[nom]
        if not c["local"].exists():
            print("   %-16s todavia no existe (%s)" % (nom, c["local"].name))
            continue
        fs, tam = _mide(c["local"], c["patron"])
        if not fs:
            print("   %-16s carpeta vacia" % nom)
            continue
        nuevos = sum(1 for f in fs
                     if "%s/%s" % (c["repo"], f.relative_to(c["local"]).as_posix()) not in ya)
        print("   %-16s %4d ficheros  %7.1f MB  (%d sin subir aun)  -> %s" % (
            nom, len(fs), tam / 1e6, nuevos, c["repo"]))
        plan.append((nom, c, fs, tam))
    if solo_lista or not plan:
        print()
        print("no se sube nada" if solo_lista else "no hay nada que subir")
        return

    print()
    subidos = []
    for nom, c, fs, tam in plan:
        api.upload_folder(folder_path=str(c["local"]), path_in_repo=c["repo"], repo_id=REPO,
                          repo_type=TIPO, allow_patterns=c["patron"],
                          commit_message="evidencia: %s" % nom)
        print("subido %-16s %4d ficheros  %7.1f MB" % (nom, len(fs), tam / 1e6))
        subidos.append((nom, len(fs), tam / 1e6, c["que"], c["repo"]))

    if subidos:
        api.upload_file(path_or_fileobj=indice(subidos).encode("utf-8"), path_in_repo="README.md",
                        repo_id=REPO, repo_type=TIPO, commit_message="indice de la evidencia")
        print()
        print("total %d ficheros, %.1f MB" % (
            sum(s[1] for s in subidos), sum(s[2] for s in subidos)))
        print("https://huggingface.co/datasets/%s" % REPO)


if __name__ == "__main__":
    main()
