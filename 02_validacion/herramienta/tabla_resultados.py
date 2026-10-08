"""La tabla de RESULTADOS por capacidad: que funciona, con que cifra, y donde esta la prueba.

LO QUE PIDIO EL USUARIO, literal: "modifica de Hugging Face los ejemplos y me pasas la tabla que te
pedi con lo que hay y los resultados".

## POR QUE HACEN FALTA TRES DOCUMENTOS Y NO UNO

  `CHECKS.md`            las GARANTIAS: cada cosa que la tuberia dice cumplir, con su umbral
                         declarado y su veredicto apto o no apto. Se ejecuta.
  `DONDE_ESTA_TODO.md`   el INVENTARIO: que hay, cuanto, donde, y que se mira en cada cosa. Se
                         genera contando el disco.
  `RESULTADOS.md`        esto: por CAPACIDAD, si funciona o no y con que numero.

Las garantias y los resultados no son lo mismo, y confundirlos es el error que este proyecto lleva
evitando desde el principio. Una garantia puede estar APTA (ningun cuadro entregado tiene un salto de
caja) mientras la capacidad que la usa **no funciona** (la prioridad acierta menos que no contestar).
Los doce checks en verde no significan que el producto funcione: significan que lo que se entrega no
se contradice a si mismo.

## DE DONDE SALE CADA CIFRA

Cada fila lleva su `fuente`, que es el fichero o la seccion del documento maestro donde la cifra se
puede volver a sacar. **Una cifra sin fuente no entra en esta tabla.** Las que dependen de una tirada
se recalculan aqui leyendo el CSV o el JSON; las que vienen de una medida cerrada de un agente van
citadas con su seccion, porque reproducirlas exige volver a correr su script y no un simple `read_csv`.

## EL ORDEN DE LAS FILAS ES DELIBERADO

Primero lo que funciona, despues lo que funciona a medias con su cobertura, y al final **lo que no
funciona y los resultados negativos**. Los negativos no van escondidos abajo por vergueenza: van
juntos porque son los que dicen en que no hay que gastar tiempo, y ese es su valor.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

VAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion")
PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")
SALIDA = VAL / "RESULTADOS.md"
REPO = "EXCALORIO/fencingly-extraccion"

FUNCIONA = [
    ("Los dos tiradores, sin intrusos",
     "mediana **100 %** de cuadros con los dos; minimo 91 %; 12 de 30 clips sin ningun aviso",
     "`CHECKS.md` C1-C8, `busca_errores.py`", "evidencia/casos_sospechosos"),
    ("Siempre las dos poses",
     "**0** cuadros entregados sin pose, en 28 clips; 96,0 % de las poses son MEDIDAS",
     "`CHECKS.md` P2 y P5", "evidencia/videos_revision"),
    ("Pose inventada cuando falta",
     "error **2,2 % del torso** de mediana (p90 6,0 %), midiendo escondiendo poses conocidas; "
     "**0 sostenidas**, todas con ancla a los dos lados",
     "`CHECKS.md` P3, seccion 21", "evidencia/poses_inventadas"),
    ("Saber que parte NO esta validada",
     "cada pose inventada lleva el largo de su hueco; **2,2 %** del total viene de huecos mayores de "
     "24 cuadros, que es donde el error deja de estar medido",
     "`CHECKS.md` P5", "evidencia/poses_inventadas"),
    ("Capas densas limpias, solo los floretistas",
     "fuga del arbitro **0,0000**; recall de keypoints 0,995",
     "seccion 16, `capas_limpias.py`", "evidencia/capas_limpias"),
    ("Quitar repeticiones, primeros planos y fundidos",
     "puerta de planos y detector de fundido con su residuo relativo al contraste (fundido 0,219 "
     "contra zoom 0,062)",
     "secciones 14 y 15", "evidencia/puerta_de_planos, evidencia/fundidos_y_zoom"),
    ("Mano armada (dominante)",
     "el criterio viejo fallaba **1 de cada 4** (0,741 sobre los 27 primeros; 0,831 sobre los 59), "
     "como dijo el usuario. El nuevo es una **cascada** y su primer peldanio es el **pie "
     "adelantado**, que es una correspondencia dentro del cuerpo y no depende de donde este el "
     "rival. **Validado en un conjunto RESERVADO** de 19 tiradores que no participaron en elegir la "
     "regla, con el criterio declarado antes de anotarlos, y con **la cifra y la cobertura por "
     "separado**: la cascada entera **acierta 1,000 con cobertura 19 de 19** (0,949 sobre los 59); "
     "**el pie con voto firme, 12 de 12 sin fallo y cobertura 12 de 19**; y el criterio del pie sin "
     "esa puerta, 0,895 sobre 19 y 0,881 sobre 59. El 1,000 es de la cascada y la cobertura del pie "
     "es 12 de 19: una redaccion anterior fundia las dos cosas en una frase. Lo que no decide se "
     "marca `dudosa` en vez de adivinar",
     "secciones 25.19 a 25.29", "evidencia/mano_dominante"),
    ("Coste por clip",
     "**17,3 s** de fase 1 por clip sobre 50 clips (56 ms por cuadro), y **YOLO se lleva el 95 %**; "
     "los filtros que preocupaban cuestan casi nada",
     "`tiempos_fase1.csv`", "documentacion"),
]

A_MEDIAS = [
    ("Eventos de movimiento: marcha, romper, a fondo, alto, brazo",
     "**650 eventos en 22 clips**. **Recall del 90 %** (44 de 49 zancadas contadas a ojo con convenio "
     "escrito antes). La confusion marcha/a fondo que vio el usuario: **arreglada**, matriz 5 de 6 "
     "con **`marcha -> fondo` = 0**, y los solapes de tipos distintos de **73 a 0** con aserto",
     "secciones 26 y 30", "evidencia/pista"),
    ("Posicion en la pista, en metros",
     "el trackeo **a lo largo** es creible (temblor p95 **0,025 m** contra 0,13-0,20 m/cuadro de un "
     "a fondo) y la excursion **transversal** es de **9 a 77 veces el ruido**, asi que su FORMA es "
     "senial. Lo que falla es la **ESCALA**: pasa la puerta fisica en el **2,6 %** de los clips, y "
     "**6 de cada 10 clips no tienen homografia** en absoluto",
     "secciones 23, 26 y 31", "evidencia/pista"),
    ("Cono de amenaza del florete",
     "eje corregido al antebrazo: los candidatos que caian al suelo bajan de **63,6 % a 15,7 %**, y "
     "el angulo contra la hoja medida pasa de 44,8 grados (el azar) a **14,4**. Largo 1,10 m y "
     "apertura declarada como **semiangulo** (+-33, o sea 66 de cono completo). Ver la hoja de verdad "
     "**estrecha el cono 3,8 veces**: semiangulo +-8,6 gr centrado en la hoja MEDIDA contra "
     "+-33,0 centrado en el antebrazo, los dos el p90 de un residuo medido. El 7,9x anterior "
     "comparaba los 33,0 con los 4,2 gr del temblor de la DIRECCION de la hoja, que es otra "
     "magnitud, y esta RETIRADO con su medida. Pero **pierde ~la mitad de las situaciones en "
     "alcance**",
     "secciones 22, 24 y 29", "evidencia/cono_arreglado"),
]

NO_FUNCIONA = [
    ("Prioridad segun la FIE",
     "acierta **69,7 %** contra **73,8 % de no contestar nunca**, y 9,1 % en clips de control. "
     "**No funciona.** Cinco de las doce reglas exigen ver la hoja y no son medibles",
     "seccion 19, `prioridad_fie.md`", "evidencia/prioridad"),
    ("Paradas",
     "**0 anotadas** en 15,4 s de accion revisada, y el contacto de hojas **no es verificable a ojo** "
     "en este material: de 16 laminas ciegas, cero decidibles. Sin verdad no hay cifra",
     "seccion 22", "evidencia/contacto_de_hojas"),
    ("Cono en 3D con `pointmap`",
     "**no se puede.** Todos los controles fisicos fallan (Z negativa en el 11,4 % de un clip, campo "
     "de vision implicado de 21 a 144 grados). Y las 11 discrepancias 2D contra 3D son **ruido**: "
     "barajando las profundidades al azar salen las mismas",
     "seccion 22", "evidencia/cono_3d"),
    ("La hoja del florete a 720p",
     "lo que se detecta **no es la hoja**: el angulo con el eje del brazo sale **60,9 grados**, peor "
     "que los 45 del azar. Y el **81 % del corpus es 720p**. A 1080p si hay senial (23,1 grados) pero "
     "la hoja solo se ve en el 3,0 % de los conos de la ventana de accion",
     "seccion 20", "evidencia/florete"),
    ("Cazar al ARBITRO entregado como tirador",
     "es el defecto recurrente sin resolver: **2 de 33 clips anotados = 6,1 %**, y **la pose del "
     "arbitro es perfectamente buena**, asi que ninguna metrica de calidad lo detectara nunca. Se han "
     "probado y **medido cuatro vias, y las cuatro fallan**: la blancura (la dispara el lame), la "
     "careta (acaba siendo un detector de oscuridad), la geometria (paso las tres puertas sin "
     "rozarlas) y **la conducta** (el arbitro dio tasa 0,524, por encima del p10 de los tiradores "
     "reales, porque el braceo de andar dispara el evento de brazo)",
     "secciones 18, 25.26 a 25.28 y 31", "evidencia/errores_encontrados"),
    ("El guante para reconocer la mano armada",
     "la idea era razonable y la medida la desmiente: separacion pareada **-0,002** y signo correcto "
     "0,455, o sea **al reves**. El rasgo lee la MANGA (pearson 0,681) y hay guantes oscuros. Ademas "
     "**ningun articulo del reglamento prohibe guante en la otra mano**",
     "seccion 25", "evidencia/mano_dominante"),
    ("El bote o la balestra",
     "**no se puede medir** y va declarado: el ruido del suelo es 3,5 cm, el umbral quedaria en 10,4 "
     "cm de despegue simultaneo y una balestra despega 5-8 cm. **0 eventos emitidos**",
     "seccion 26", "evidencia/pista"),
    ("Detectar pistas con plancha extra",
     "**no es observable**: en el encuadre entran solo **5,53 m de los 14**, asi que el largo total no "
     "se ve en un clip",
     "seccion 23", "evidencia/pista"),
]

ERRORES = [
    ("el arbitro elegido como tirador", "`PowerfulAllLarva`",
     "paso las tres puertas geometricas sin rozarlas. **Sin arreglar**: marcado por el aviso "
     "`blancura` para revision", "evidencia/cruce_de_lados"),
    ("un trozo de espectador del borde como tirador", "`LavishDistantDikkops`",
     "**arreglado**: 27,4 % a 100 %. Una caja cortada por el borde en el 100 % de los cuadros nunca "
     "vio a la persona entera", "evidencia/intrusos_verificados"),
    ("10.140 keypoints sobre otra persona", "28 clips",
     "**arreglado**: el filtro existia en la ruta de las capas densas y no en la de entrega",
     "evidencia/capas_densas"),
    ("8 poses sostenidas en los bordes", "45 casos de galeria",
     "**arreglado**: recortar los extremos DESPUES de filtrar poses vacias. Quedan 0",
     "evidencia/poses_inventadas"),
    ("el detector de pasos no detectaba pasos", "11 clips",
     "**arreglado**: picos de 12,33 m/s que eran saltos de keypoint. La marcha pasa de 0,139 a 0,313 m",
     "evidencia/pista"),
    ("un a fondo dispara el detector de paso", "41 % de los pasos",
     "**marcado, no filtrado**: hay columna `solapa_fondo`. Encontrado mirando un recorte, no en "
     "ninguna cifra agregada", "evidencia/pista"),
    ("el CSV agregado tenia un solo clip", "284 filas contra 40",
     "**arreglado de raiz**: los agregados se reconstruyen leyendo el disco, asi que una pasada de un "
     "clip suelto es inofensiva por construccion", "evidencia/pista"),
    ("la puerta de publicacion no hacia lo que decia", "12,8 % contra 2,6 %",
     "comparaba la mediana agrupada de los dos tiradores en vez de cada uno. **Corregido** con "
     "`publicable_estricto`", "evidencia/pista"),
]


def cifra_tiempos():
    f = PRUEBAS / "tiempos_fase1.csv"
    if not f.exists():
        return None
    import pandas as pd
    d = pd.read_csv(f)
    return len(d), d["t_fase1"].median(), d["T"].median()


def cuenta_repo():
    try:
        import sube_evidencia as SE
        from huggingface_hub import HfApi
        api = HfApi(token=SE._token())
        fs = api.list_repo_files(repo_id=SE.REPO, repo_type=SE.TIPO)
        return len([f for f in fs if f.startswith("evidencia/")]), len(fs)
    except Exception:
        return None, None


def main():
    ev, tot = cuenta_repo()
    t = cifra_tiempos()
    l = ["# Resultados por capacidad", "",
         "Que funciona, con que cifra, y donde esta la prueba. Cada fila lleva su **fuente**: una "
         "cifra sin fuente no entra en esta tabla.", "",
         "Este documento no sustituye a los otros dos y no dicen lo mismo:", "",
         "| documento | que contesta |", "|---|---|",
         "| [`CHECKS.md`](CHECKS.md) | las **garantias**: lo que la tuberia dice cumplir, con su "
         "umbral y su veredicto. Se ejecuta. |",
         "| [`DONDE_ESTA_TODO.md`](DONDE_ESTA_TODO.md) | el **inventario**: que hay, cuanto, donde, y "
         "que se mira en cada cosa. |",
         "| este | los **resultados**: si cada capacidad funciona o no. |", "",
         "**Y la distincion importa.** Los doce checks estan en verde y eso **no** significa que el "
         "producto funcione: significa que lo que se entrega no se contradice a si mismo. Una "
         "garantia puede estar apta mientras la capacidad que la usa no funciona, y el caso mas claro "
         "esta abajo: la prioridad acierta menos que no contestar nunca.", ""]
    if ev:
        l.append("Evidencia en el repo **privado** `%s`: %d ficheros bajo `evidencia/`, %d en total. "
                 "Privado porque son **atletas identificables en competicion**." % (REPO, ev, tot))
        l.append("")

    def bloque(titulo, intro, filas, cab):
        l.append("## %s" % titulo)
        l.append("")
        if intro:
            l.append(intro)
            l.append("")
        l.append("| %s |" % " | ".join(cab))
        l.append("|%s|" % "|".join("---" for _ in cab))
        for f in filas:
            l.append("| %s |" % " | ".join(str(x) for x in f))
        l.append("")

    bloque("Lo que funciona", "", FUNCIONA, ["capacidad", "cifra medida", "fuente", "evidencia"])
    bloque("Lo que funciona a medias, con su cobertura declarada",
           "Estas tres se pueden usar **si se respeta el subconjunto donde valen**. La cobertura no "
           "es una pega menor: es parte del resultado.",
           A_MEDIAS, ["capacidad", "cifra medida", "fuente", "evidencia"])
    bloque("Lo que NO funciona, y por que",
           "Estos son los mas utiles de la tabla, porque dicen **en que no hay que gastar tiempo**. "
           "Cuatro de los siete son resultados negativos medidos con su control, no cosas a medio "
           "hacer.",
           NO_FUNCIONA, ["capacidad", "cifra medida", "fuente", "evidencia"])
    bloque("Averias encontradas y en que quedaron",
           "Ocho averias que **ninguna cifra agregada veia**. Tres se encontraron mirando una imagen "
           "o un recorte, no una tabla.",
           ERRORES, ["averia", "donde", "estado", "evidencia"])

    if t:
        l.append("## Coste, medido")
        l.append("")
        l.append("Sobre **%d clips** de %d cuadros de mediana: **%.1f s por clip** de fase 1, o sea "
                 "%.0f ms por cuadro. YOLO de cuadro completo se lleva el 95 %% de eso; la puerta de "
                 "planos, la de fundidos, las trayectorias, la eleccion de pareja y toda la limpieza "
                 "suman menos de 3 s." % (t[0], int(t[2]), t[1], 1000 * t[1] / max(t[2], 1)))
        l.append("")
        l.append("La medida con la GPU compartida sale inflada y no vale: la contencion **solo puede "
                 "sumar** tiempo. Por eso `tiempos.py` avisa de la ocupacion al empezar e informa el "
                 "p10 al lado de la mediana.")
        l.append("")

    l.append("## Lo que esta tabla NO dice")
    l.append("")
    l.append("Las cifras de cajas salen de 30 a 60 clips y las de poses de 28, **no del corpus de "
             "8.559**. Las verdades de referencia de la mano armada, de la prioridad y del contacto "
             "de hojas **las he construido yo a ojo**, no un arbitro FIE, y estan en fichero para que "
             "se puedan contradecir. Y varias medidas se apoyan en muestras pequenias que estan "
             "senialadas en su seccion: el careo del eje del cono con la hoja tiene 24 de sus 31 "
             "detecciones en un solo asalto, y el subconjunto firme de la mano armada son 13 casos.")
    l.append("")
    SALIDA.write_text("\n".join(l), encoding="utf-8")
    print("\n".join(l))
    print()
    print("escrito en %s" % SALIDA)


if __name__ == "__main__":
    main()
