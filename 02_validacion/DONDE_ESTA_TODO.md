# Donde esta cada prueba

Generado contando el disco, no escrito a mano: un indice a mano miente en cuanto alguien anade una carpeta, y este documento existe para poder comprobar.

La tabla de garantias con su veredicto esta en [`CHECKS.md`](CHECKS.md). Esto dice **que hay, donde esta y que se mira en cada cosa**.

Todo lo visual esta ademas en el repo privado de Hugging Face `EXCALORIO/fencingly-extraccion`. Va al privado y no al publico porque son **atletas identificables en competicion**.

| que es | ficheros | tamanio | en local | en el repo | que se mira |
|---|---:|---:|---|---|---|
| galeria | 44 | 50.8 MB | `02_validacion/poses_inventadas` | `evidencia/poses_inventadas` | Que la pose VIOLETA del panel central sea una continuacion creible de las dos medidas de los lados, y no un esqueleto puesto donde no hay nadie. El rotulo dice el largo del hueco; los que van en ROJO estan fuera del rango donde el error esta medido y su error no es verificable. |
| videos | 26 | 324.0 MB | `02_validacion/videos_revision` | `evidencia/videos_revision` | Que las cajas no cambien de persona, que no aparezca el arbitro ni el publico, y que las poses violeta esten donde deben. La banda roja marca los cuadros descartados. |
| capas | 3 | 0.9 MB | `02_validacion/muestras/capas` | `evidencia/capas_densas` | Capas densas sobre los tiradores, sin limpiar, que es la linea base que falla. |
| capas_sapiens | 4 | 3.5 MB | `02_validacion/muestras/capas_sapiens` | `evidencia/capas_densas_sapiens` | Mas ejemplos de las cabezas densas de cuadro completo. |
| arma | 6 | 5.9 MB | `02_validacion/muestras/arma` | `evidencia/florete` | El filo de la hoja en el alfa del matting. |
| cono | 10 | 7.2 MB | `02_validacion/muestras/cono` | `evidencia/cono_del_florete` | El cono de la hoja en 2D y su interseccion con el rival. |
| planos | 8 | 1.3 MB | `02_validacion/muestras/planos` | `evidencia/puerta_de_planos` | Los primeros planos y repeticiones que la puerta descarta. |
| zoom | 9 | 5.9 MB | `02_validacion/muestras/zoom` | `evidencia/fundidos_y_zoom` | Fundidos encadenados y zooms, que dan la misma firma y se separan por el residuo. |
| redeteccion | 12 | 8.5 MB | `02_validacion/muestras/redeteccion` | `evidencia/redeteccion_en_recorte` | Cuadros donde faltaba uno y se recupero buscandolo en un recorte. |
| pareja | 30 | 20.3 MB | `02_validacion/muestras/ab_pareja` | `evidencia/eleccion_de_pareja` | La eleccion de la pareja FILMADA cuando hay varias pistas, arbitro y publico. |
| sospechosos | 51 | 37.1 MB | `02_validacion/muestras/sospechosos` | `evidencia/casos_sospechosos` | Lo que el buscador automatico marco, con su sintoma: salto, solapan, fundido, recorta, sep, dpie, faltan, pocos, blancura, borde. |
| pista | 313 | 250.5 MB | `02_validacion/pista` | `evidencia/pista` | La pista a escala con sus lineas, la trayectoria en metros de los dos tiradores a lo largo y a lo ancho, la distancia entre ellos, y los eventos marcados con su desplazamiento en metros. Si el titulo dice `fisica NO PASA`, la homografia de ese clip no es valida todavia. |
| prioridad | 86 | 326.8 MB | `02_validacion/prioridad` | `evidencia/prioridad` | La prioridad cuadro a cuadro con la regla FIE que la concede. El detector NO funciona todavia (69,7 % contra 73,8 % de no contestar nunca): las laminas sirven para ver DONDE falla. |
| mosaicos | 82 | 69.9 MB | `02_validacion/mosaicos` | `evidencia/mosaicos` | Las seis celdas a la vez y sincronizadas: original con esqueleto, matting, seg, normal, pointmap y florete. Sirve para ver de un golpe si una capa se descuelga. |
| intrusos | 28 | 22.4 MB | `02_validacion/muestras/limite_intruso` | `evidencia/intrusos_verificados` | Los casos AL LIMITE de las seniales de intruso, verificados uno por uno. Estan los dos errores reales y tambien los CORRECTOS que cada regla candidata se llevaba de mas, que es lo que impidio convertir la blancura en puerta automatica. |
| cruce | 7 | 4.0 MB | `02_validacion/muestras/cruce_lados` | `evidencia/cruce_de_lados` | El clip donde el `tirador izquierdo` entregado es el ARBITRO. Es el caso que demostro que las tres puertas geometricas no pueden distinguirlo. |
| capas_limpias | 30 | 39.3 MB | `02_validacion/muestras/capas_limpias` | `evidencia/capas_limpias` | Las cuatro capas ANTES y DESPUES, en pares. Que el arbitro y el publico desaparezcan, y que la mascara del tirador NO salga con agujeros: una persona es una sola pieza. |
| cono3d | 9 | 3.7 MB | `02_validacion/muestras/cono3d` | `evidencia/cono_3d` | El cono en 3D con vista de planta en centimetros y el circulo de 90 cm, que es el largo reglamentario de la hoja. Los ficheros `DISCREPA_` son los cuadros donde el 2D y el 3D no dicen lo mismo: se buscaron a proposito y resultaron ser RUIDO, porque barajando las profundidades al azar salen los mismos. |
| cono_arreglado | 10 | 7.8 MB | `02_validacion/muestras/cono_arreglado` | `evidencia/cono_arreglado` | ANTES y DESPUES del cono en el MISMO fotograma, tras los tres defectos que senialo el usuario. Lo que parecia un florete recto ERA EL BRAZO, pintado del color del cono. El largo pasa de 0,90 m (la hoja desde la guarda) a 1,10 m (el alcance muniera-punta, que es de donde sale el cono). Y la apertura estaba al REVES de lo que parecia: los 23,1 grados contenian solo el 22,6 % de las hojas medidas. El |
| mano_dominante | 106 | 120.9 MB | `02_validacion/muestras/mano_dominante` | `evidencia/mano_dominante` | La mano ARMADA elegida por tirador, con la senial del guante. El criterio viejo era `la mas adelantada` y el usuario aviso de que fallaba a menudo. El guante es una senial independiente de la geometria, porque es obligatorio en la mano armada y la otra va desnuda. |
| cono_cono | 32 | 24.4 MB | `02_validacion/muestras/cono_cono` | `evidencia/cono_contra_cono` | La interseccion del cono de un tirador con el cono del OTRO, que es donde puede haber CONTACTO ENTRE LAS DOS HOJAS, con la region de solape sombreada. Es el planteamiento correcto y lo corrigio el usuario: la version anterior cruzaba el cono contra el TORSO del rival, que contesta la pregunta de la amenaza pero no esta. Interesa porque el contacto de hojas es la senial que falta para detectar PARA |
| contacto | 23 | 56.6 MB | `02_validacion/muestras/contacto_hojas` | `evidencia/contacto_de_hojas` | Laminas anotadas A CIEGAS (sin conos, sin nombre de clip, en orden aleatorio) para intentar validar el solape contra verdad vista a ojo. El resultado es negativo y por eso estan aqui: de 16 laminas, CERO decidibles como `si hay contacto`. Los monitores del marcador tapan la region del cruce y el desenfoque de movimiento cae justo en los cuadros que importan. O sea que el contacto de hojas **no es  |
| checks | 2 | 0.0 MB | `02_validacion` | `documentacion` | La tabla de garantias: umbral declarado, lo medido, veredicto, el sitio del codigo que lo cumple y donde esta la evidencia. |
| documentacion | 1 | 0.4 MB | `C:/Users/Alejandro/Desktop/FENCINGLY_PROYECTO` | `documentacion` | El documento maestro, con todo lo medido y las averias resueltas. |
| **total** | **932** | **1392 MB** | | | |

## Las herramientas, para poder repetir cualquier cifra

| script | que hace |
|---|---|
| `02_validacion/herramienta/tabla_checks.py` | Ejecuta las garantias y escribe `CHECKS.md`. `cajas N` sin GPU, `poses` con el de cuda. |
| `02_validacion/herramienta/busca_errores.py` | Busca SINTOMAS y saca imagenes de los cuadros senialados, ordenando por sospecha. |
| `02_validacion/herramienta/galeria_inventadas.py` | La galeria de poses inventadas en tripanel. |
| `02_validacion/herramienta/repara_npz_poses.py` | Aplica a los NPZ ya calculados los filtros anadidos despues, SIN GPU. |
| `02_validacion/herramienta/tiempos.py` | Coste por clip paso a paso. `fase1` con ultralytics, `fase2` con cuda, `informe` junta. |
| `02_validacion/herramienta/video_revision.py` | Los videos de revision y el calculo de las poses. Dos fases. |
| `02_validacion/herramienta/sube_evidencia.py` | Sube la evidencia al repo PRIVADO, y se niega a subir si el repo no es privado. |
| `01_extraccion/codigo/extrae_pareja.py` | La cadena completa que saca los dos tiradores. Entrada unica del proyecto. |
| `01_extraccion/codigo/blancura_pareja.py` | Mide si la blancura sirve para cazar al arbitro elegido como tirador. |

### Los dos interpretes NO coexisten

`ultralytics` y `transformers`+`torch` viven en entornos distintos, y por eso las herramientas van en dos fases. Correr un script contra el interprete equivocado da un `ModuleNotFoundError` y ya ha costado tiempo en este proyecto: comprueba con `-c "import X"` antes de una tirada larga.

## Lo que sigue sin funcionar, dicho aqui para que no se pierda

| que | estado medido |
|---|---|
| deteccion de PRIORIDAD | 69,7 % de acierto contra **73,8 % de no contestar nunca**, y 9,1 % en clips de control. **No funciona todavia.** |
| PARADAS | **0 anotadas** en 15,4 s de accion revisada. Sin verdad no hay cifra. |
| cono en 3D con `pointmap` | **no se puede**: todos los controles fisicos fallan, y las 11 discrepancias entre 2D y 3D son ruido (barajando las profundidades salen las mismas). |
| la HOJA a 720p | lo que se detecta **no es la hoja**: el angulo con el eje del brazo sale 60,9 grados, peor que los 45 del azar. Y el 81 % del corpus es 720p. |
| `PowerfulAllLarva` | sigue entregando al ARBITRO como tirador izquierdo. Marcado por el aviso `blancura` para revision; la tuberia no lo corrige sola. |
