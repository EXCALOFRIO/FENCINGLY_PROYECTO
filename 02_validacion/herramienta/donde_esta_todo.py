"""Genera el indice de DONDE ESTA CADA PRUEBA, contando los ficheros de verdad.

LO QUE PIDIO EL USUARIO, literal: "haz una tabla y me lo pones al terminar con los checks y donde
estan las pruebas de todo para verificarlo".

`CHECKS.md` contesta la primera mitad: cada garantia con su umbral y su veredicto. Esto contesta la
segunda: que hay, cuanto hay, donde esta en local, donde esta en el repo privado, y **que se mira en
cada cosa**, que es lo que la vuelve verificable en vez de un vertedero de figuras.

Se genera contando el disco, no a mano. Un indice escrito a mano miente en cuanto alguien anade una
carpeta, y este documento existe justamente para poder comprobar.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sube_evidencia as SE

VAL = SE.VAL
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
SALIDA = VAL / "DONDE_ESTA_TODO.md"

QUE_MIRAR = {
    "galeria": "Que la pose VIOLETA del panel central sea una continuacion creible de las dos "
               "medidas de los lados, y no un esqueleto puesto donde no hay nadie. El rotulo dice el "
               "largo del hueco; los que van en ROJO estan fuera del rango donde el error esta "
               "medido y su error no es verificable.",
    "videos": "Que las cajas no cambien de persona, que no aparezca el arbitro ni el publico, y que "
              "las poses violeta esten donde deben. La banda roja marca los cuadros descartados.",
    "intrusos": "Los casos AL LIMITE de las seniales de intruso, verificados uno por uno. Estan los "
                "dos errores reales y tambien los CORRECTOS que cada regla candidata se llevaba de "
                "mas, que es lo que impidio convertir la blancura en puerta automatica.",
    "cruce": "El clip donde el `tirador izquierdo` entregado es el ARBITRO. Es el caso que demostro "
             "que las tres puertas geometricas no pueden distinguirlo.",
    "capas_limpias": "Las cuatro capas ANTES y DESPUES, en pares. Que el arbitro y el publico "
                     "desaparezcan, y que la mascara del tirador NO salga con agujeros: una persona "
                     "es una sola pieza.",
    "capas": "Capas densas sobre los tiradores, sin limpiar, que es la linea base que falla.",
    "capas_sapiens": "Mas ejemplos de las cabezas densas de cuadro completo.",
    "mosaicos": "Las seis celdas a la vez y sincronizadas: original con esqueleto, matting, seg, "
                "normal, pointmap y florete. Sirve para ver de un golpe si una capa se descuelga.",
    "arma": "El filo de la hoja en el alfa del matting.",
    "cono": "El cono de la hoja en 2D y su interseccion con el rival.",
    "cono3d": "El cono en 3D con vista de planta en centimetros y el circulo de 90 cm, que es el "
              "largo reglamentario de la hoja. Los ficheros `DISCREPA_` son los cuadros donde el 2D "
              "y el 3D no dicen lo mismo: se buscaron a proposito y resultaron ser RUIDO, porque "
              "barajando las profundidades al azar salen los mismos.",
    "prioridad": "La prioridad cuadro a cuadro con la regla FIE que la concede. El detector NO "
                 "funciona todavia (69,7 % contra 73,8 % de no contestar nunca): las laminas sirven "
                 "para ver DONDE falla.",
    "pista": "La pista a escala con sus lineas, la trayectoria en metros de los dos tiradores a lo "
             "largo y a lo ancho, la distancia entre ellos, y los eventos marcados con su "
             "desplazamiento en metros. Si el titulo dice `fisica NO PASA`, la homografia de ese "
             "clip no es valida todavia.",
    "planos": "Los primeros planos y repeticiones que la puerta descarta.",
    "zoom": "Fundidos encadenados y zooms, que dan la misma firma y se separan por el residuo.",
    "redeteccion": "Cuadros donde faltaba uno y se recupero buscandolo en un recorte.",
    "pareja": "La eleccion de la pareja FILMADA cuando hay varias pistas, arbitro y publico.",
    "sospechosos": "Lo que el buscador automatico marco, con su sintoma: salto, solapan, fundido, "
                   "recorta, sep, dpie, faltan, pocos, blancura, borde.",
    "checks": "La tabla de garantias: umbral declarado, lo medido, veredicto, el sitio del codigo "
              "que lo cumple y donde esta la evidencia.",
    "documentacion": "El documento maestro, con todo lo medido y las averias resueltas.",
}
"""Que se mira en cada carpeta. Sin esto un indice solo dice cuantos ficheros hay, y una figura de la
que no se sabe que juzgar no es evidencia de nada."""

HERRAMIENTAS = [
    ("02_validacion/herramienta/tabla_checks.py",
     "Ejecuta las garantias y escribe `CHECKS.md`. `cajas N` sin GPU, `poses` con el de cuda."),
    ("02_validacion/herramienta/busca_errores.py",
     "Busca SINTOMAS y saca imagenes de los cuadros senialados, ordenando por sospecha."),
    ("02_validacion/herramienta/galeria_inventadas.py",
     "La galeria de poses inventadas en tripanel."),
    ("02_validacion/herramienta/repara_npz_poses.py",
     "Aplica a los NPZ ya calculados los filtros anadidos despues, SIN GPU."),
    ("02_validacion/herramienta/tiempos.py",
     "Coste por clip paso a paso. `fase1` con ultralytics, `fase2` con cuda, `informe` junta."),
    ("02_validacion/herramienta/video_revision.py",
     "Los videos de revision y el calculo de las poses. Dos fases."),
    ("02_validacion/herramienta/sube_evidencia.py",
     "Sube la evidencia al repo PRIVADO, y se niega a subir si el repo no es privado."),
    ("01_extraccion/codigo/extrae_pareja.py",
     "La cadena completa que saca los dos tiradores. Entrada unica del proyecto."),
    ("01_extraccion/codigo/blancura_pareja.py",
     "Mide si la blancura sirve para cazar al arbitro elegido como tirador."),
]


def cuenta(carpeta, patrones):
    fs, tam = SE._mide(carpeta, patrones)
    return len(fs), tam


def main():
    l = ["# Donde esta cada prueba", "",
         "Generado contando el disco, no escrito a mano: un indice a mano miente en cuanto alguien "
         "anade una carpeta, y este documento existe para poder comprobar.", "",
         "La tabla de garantias con su veredicto esta en [`CHECKS.md`](CHECKS.md). Esto dice **que "
         "hay, donde esta y que se mira en cada cosa**.", "",
         "Todo lo visual esta ademas en el repo privado de Hugging Face `%s`. Va al privado y no al "
         "publico porque son **atletas identificables en competicion**." % SE.REPO, "",
         "| que es | ficheros | tamanio | en local | en el repo | que se mira |",
         "|---|---:|---:|---|---|---|"]
    tot_n, tot_b = 0, 0
    for nom, c in SE.LOTES.items():
        if not c["local"].exists():
            continue
        n, b = cuenta(c["local"], c["patron"])
        if not n:
            continue
        tot_n += n
        tot_b += b
        rel = str(c["local"]).replace(str(RAIZ) + "\\", "").replace("\\", "/")
        l.append("| %s | %d | %.1f MB | `%s` | `%s` | %s |" % (
            nom, n, b / 1e6, rel, c["repo"],
            QUE_MIRAR.get(nom, c["que"])[:400]))
    l.append("| **total** | **%d** | **%.0f MB** | | | |" % (tot_n, tot_b / 1e6))
    l.append("")
    l.append("## Las herramientas, para poder repetir cualquier cifra")
    l.append("")
    l.append("| script | que hace |")
    l.append("|---|---|")
    for r, q in HERRAMIENTAS:
        l.append("| `%s` | %s |" % (r, q))
    l.append("")
    l.append("### Los dos interpretes NO coexisten")
    l.append("")
    l.append("`ultralytics` y `transformers`+`torch` viven en entornos distintos, y por eso las "
             "herramientas van en dos fases. Correr un script contra el interprete equivocado da un "
             "`ModuleNotFoundError` y ya ha costado tiempo en este proyecto: comprueba con "
             "`-c \"import X\"` antes de una tirada larga.")
    l.append("")
    l.append("## Lo que sigue sin funcionar, dicho aqui para que no se pierda")
    l.append("")
    l.append("| que | estado medido |")
    l.append("|---|---|")
    l.append("| deteccion de PRIORIDAD | 69,7 % de acierto contra **73,8 % de no contestar nunca**, "
             "y 9,1 % en clips de control. **No funciona todavia.** |")
    l.append("| PARADAS | **0 anotadas** en 15,4 s de accion revisada. Sin verdad no hay cifra. |")
    l.append("| cono en 3D con `pointmap` | **no se puede**: todos los controles fisicos fallan, y "
             "las 11 discrepancias entre 2D y 3D son ruido (barajando las profundidades salen las "
             "mismas). |")
    l.append("| la HOJA a 720p | lo que se detecta **no es la hoja**: el angulo con el eje del brazo "
             "sale 60,9 grados, peor que los 45 del azar. Y el 81 % del corpus es 720p. |")
    l.append("| `PowerfulAllLarva` | sigue entregando al ARBITRO como tirador izquierdo. Marcado "
             "por el aviso `blancura` para revision; la tuberia no lo corrige sola. |")
    l.append("")
    SALIDA.write_text("\n".join(l), encoding="utf-8")
    print("\n".join(l))
    print()
    print("escrito en %s" % SALIDA)


if __name__ == "__main__":
    main()
