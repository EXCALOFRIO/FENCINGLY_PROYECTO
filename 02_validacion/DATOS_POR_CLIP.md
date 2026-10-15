# Todos los datos que sacamos de cada clip

**Esto es lo que se usara para entrenar.** Cada fila es un campo que sale de cada clip, con su forma real leida del disco, que significa, en que unidad, **si esta verificado y con que error medido**.

Un tensor de poses sin su error es una promesa. Con su error es un dato. Por eso la ultima columna es la que importa.

Las formas y los nombres **se leen de los ficheros**, no estan escritos a mano: si alguien anade o quita un campo, la tabla lo dice en vez de quedarse mintiendo. Un campo que aparezca en disco sin descripcion sale como `SIN DESCRIBIR`.

Estado hoy: **39 clips** con poses de 308 puntos, de un corpus de 8.559.

### Las tres marcas de procedencia, que son lo que no se puede recuperar despues

`origen` (medida o inventada), `hueco` (de que hueco sale lo inventado) y `lado_valido` (si ese lado es un tirador de verdad) **no son metadatos: son parte del dato**. Si se entrena mezclando poses medidas e inventadas sin distinguirlas, el modelo aprende el ruido de la interpolacion y **nadie se enteraria nunca**.

## 1 · Poses y cajas · `poses_<clip>.npz`

Un fichero por clip. `T` es el numero de cuadros y el 2 son los dos tiradores, lado 0 el de la izquierda de la imagen y lado 1 el de la derecha.

| campo | forma | que es | unidad | verificado | error medido |
|---|---|---|---|---|---|
| `KP` | `(316, 2, 308, 2)` | los 308 keypoints de los dos tiradores, en pixeles | px | SI, a ojo en video y con la galeria de tripaneles | **2,26 px** de error de torso por bajar la resolucion de 1024 a 384x288 (medido). **96,0 %** de las poses entregadas son MEDIDAS |
| `SC` | `(316, 2, 308)` | la confianza de cada keypoint | 0-1 | SI, indirectamente | se usa con umbral 0,3. Los puntos fuera de la caja de su persona van a **cero**: eran **10.140** y estaban sobre otra persona |
| `origen` | `(316, 2)` | como se obtuvo la pose: 0 medida, 1 interpolada, 2 sostenida, -1 nada | codigo | SI, es la marca que permitio auditar todo lo demas | **0 sostenidas**. Sin este campo no se puede pesar el dato y NO se puede recuperar despues |
| `hueco` | `(316, 2)` | largo en cuadros del hueco del que sale cada pose inventada | cuadros | SI, contra la tabla de error por largo de hueco | interpolar con hueco <= 24 cuadros da **2,2 % del torso** (p90 6,0 %). Por encima de 24 **NO hay medida**, y ese es el **2,2 %** de lo entregado |
| `B` | `(316, 2, 4)` | caja de cada tirador, normalizada 0-1, en xyxy | fraccion del cuadro | SI, 60 clips y verificacion a ojo | ningun cuadro entregado salta mas de **0,60 anchos de caja** (peor medido 0,563). Ninguna caja es una tira del borde ni pertenece a alguien nunca visto entero |
| `Be` | `(316, 2, 4)` | caja ESTIMADA donde deberia estar el tirador cuando falta | fraccion del cuadro | parcial | se usa para buscar en un recorte; su error no se ha medido aparte |
| `pres` | `(316, 2)` | si la pose de ese tirador en ese cuadro es MEDIDA | bool | SI | **0** cuadros entregados sin los dos, en 28 clips |
| `mask` | `(316,)` | cuadros utilizables: fuera primeros planos, repeticiones y fundidos | bool | SI | la puerta de fundidos separa fundido de zoom por el residuo relativo al contraste (0,219 contra 0,062) |
| `W` | `()` | ancho del video | px | SI | **81 %** del corpus es 1280x720 |
| `H` | `()` | alto del video | px | SI | 18 % es 1920x1080, 0,8 % es 854x480 |
| `T` | `()` | cuadros del clip | cuadros | SI | mediana **307**; el viejo guardaba solo **129** |
| `motivo` | `()` | por que se eligio esa pareja, con sus cifras | texto | SI | permite contradecir la eleccion sin recalcular |
| `video` | `()` | ruta al mp4 de origen | texto | - | - |
| `filtro_vacias` | `()` | cuantas poses medidas se tiraron por venir vacias | cuenta | SI | una pose con menos de **20 puntos utiles de 63** no vale como ancla |
| `puntos_min` | `()` | umbral de puntos utiles usado en este fichero | cuenta | - | va guardado para que el fichero diga COMO se hizo |

## 2 · Eventos de movimiento · `serie_eventos.csv`

Una fila por EVENTO, no por cuadro. Es la serie temporal de bajo nivel que pedia el usuario: mas facil de leer e interpretar que las poses crudas. **650 eventos en 22 clips.** Se consulta con una linea: `ev.query("lado==1 and tipo=='marcha' and indice_del_tipo==3")`.

| campo | forma | que es | unidad | verificado | error medido |
|---|---|---|---|---|---|
| `clip` | `columna` | el clip del que sale el evento | texto | SI | 22 clips |
| `lado` | `columna` | 0 el tirador de la izquierda de la imagen, 1 el de la derecha | 0/1 | SI | **0 cruces en 2.896 cuadros**: el lado 0 esta a la izquierda en 11 de 11 clips |
| `lado_valido` | `columna` | si ese lado es un tirador de verdad | bool | PARCIAL | en **6,1 %** de los clips el lado entregado es el **ARBITRO**, y las **cuatro** vias probadas para cazarlo han fallado |
| `tipo` | `columna` | marcha, romper, fondo, alto, brazo, accion, bote | categoria | SI, matriz de confusion contra 6 gestos a ojo | **5 de 6**, y **`marcha -> fondo` = 0**. `bote` va con **0 eventos**: no es medible, y hay un aserto de gravedad que lo impide |
| `indice_del_tipo` | `columna` | el numero del evento dentro de su tipo y su lado | cuenta | SI | es lo que permite consultar «la tercera marcha del tirador derecho» en una linea, y es **el mismo numero que sale en el contador del video** |
| `cuadro_inicio` | `columna` | cuadro en que empieza el evento | cuadro | PARCIAL | desfase medido **+0,5 cuadros (+17 ms)** sobre 62 emparejamientos, sin sesgo. El **error de FRONTERA esta sin medir**: es lo que falta anotar |
| `cuadro_fin` | `columna` | cuadro en que acaba | cuadro | PARCIAL | igual que el inicio |
| `cuadro_pico` | `columna` | cuadro del maximo de la senial dentro del evento | cuadro | PARCIAL | es donde se miden los rasgos de arbitraje, no donde empieza el gesto |
| `t_inicio_s` | `columna` | segundo en que empieza | s | PARCIAL | el mismo desfase que `cuadro_inicio` |
| `t_fin_s` | `columna` | segundo en que acaba | s | PARCIAL | idem |
| `duracion_s` | `columna` | cuanto dura el evento | s | PARCIAL | marcha ~0,10-0,40 s. El **bote** se rechaza con esto y la gravedad: 8,5 cm exigen 7,9 cuadros de vuelo y el candidato duraba 1 |
| `x_inicio_m` | `columna` | posicion a lo largo de la pista al empezar | m desde la linea central | PARCIAL | temblor p95 **0,025 m**. Convenio: **x crece hacia la derecha de la imagen** |
| `x_fin_m` | `columna` | posicion a lo largo al acabar | m | PARCIAL | igual que `x_inicio_m` |
| `magnitud_m` | `columna` | cuanto se desplazo, en metros | m | PARCIAL | marcha p50 **0,343 m**, a fondo p50 0,602. Depende de la escala de la pista, que solo pasa la puerta fisica en el **2,6 %** de los clips |
| `unidad_magnitud` | `columna` | si la magnitud va en metros o en altos de caja | categoria | SI | cuando no hay homografia la magnitud **no se convierte**, se declara en otra unidad |
| `velocidad_m_s` | `columna` | magnitud dividida por duracion | m/s | PARCIAL | hereda el error de la escala. Referencia: un a fondo real son 4-6 m/s |
| `y_inicio_m` | `columna` | posicion a lo ancho de la pista | m | NO la escala, SI la forma | la excursion es de **9 a 77 veces el ruido**, asi que la FORMA es senial; la **escala esta mal** y el control fisico pasa 14 de 22 |
| `y_fin_m` | `columna` | posicion a lo ancho al acabar | m | NO la escala | igual que `y_inicio_m` |
| `transversal_creible` | `columna` | si la excursion a lo ancho supera el ruido | bool | SI | **22 de 22** la superan: de 9 a 77 veces el ruido |
| `pies_movidos` | `columna` | cuantos pies cambiaron de apoyo en el evento | cuenta | SI | es lo que separa marcha (los dos) de a fondo (solo el de delante) |
| `completa` | `columna` | si el ciclo del gesto se cerro dentro del tramo utilizable | bool | SI | un gesto cortado por el borde del clip no se puede medir entero |
| `ext_torso` | `columna` | extension del brazo en fracciones de torso | razon | SI | usa `hombro -> raiz de la mano`, que es la que CRECE al extenderse (cv 0,226 contra 0,108 del antebrazo) |
| `solapa_fondo` | `columna` | si el evento coincide en el tiempo con un a fondo | bool | SI | antes del arbitraje el **41 %** de los pasos solapaban un a fondo; ahora los solapes de tipos distintos son **0** |
| `arbitrado` | `columna` | si hubo que decidir entre dos tipos solapados | bool | SI | los solapes de tipos distintos pasan de **73 a 0**, con aserto |
| `razon_pie_atras` | `columna` | desplazamiento del pie de atras sobre el de delante | razon | SI | es **el** discriminador marcha/fondo: frontera 0,45, medido en la fase de extension y no en toda la ventana |
| `pico_dpies_guardia` | `columna` | separacion de pies en el pico, sobre la guardia del clip | razon | SI | frontera 1,25. Es el segundo rasgo del arbitraje marcha/fondo |
| `margen_arbitraje` | `columna` | cuanta holgura tuvo la decision del arbitraje | 0-1 | SI | p50 **0,286**; solo **2 de 48** quedan en la frontera |
| `guardia_m` | `columna` | separacion de pies de referencia de ese tirador en ese clip | m | PARCIAL | de los 17 lados con guardia, **10 caen en los 0,50-0,90 m** de un floretista y 7 no: el resto hereda la escala mala |
| `guardia_plausible` | `columna` | si esa guardia cae en el rango de un floretista | bool | SI | 10 de 17 |
| `lado_no_valido_por` | `columna` | por que ese lado no es un tirador | texto | SI | hoy solo toma el valor de arbitro, en 2 clips de 33 |
| `via_accion` | `columna` | si la accion se midio en metros o en altos de caja | categoria | SI | las dos vias dan el mismo resultado en la puerta de la mano: 8 de 8 y 11 de 11 |
| `magnitud_altos_caja` | `columna` | el desplazamiento medido en altos de caja | razon | SI | es la via **sin homografia**, y da el mismo resultado que la metrica en la puerta de la mano armada |
| `levanta_frac_caja` | `columna` | cuanto se despega el pie, en fraccion del alto de caja | razon | SI | es la senial del bote, y por eso el bote sale a **0**: el ruido del suelo es 3,5 cm y una balestra despega 5-8 cm |
| `sobre_ruido` | `columna` | cuantas veces el ruido medido supera la senial del evento | razon | SI | criterio declarado de 5x; la transversal sale entre 9 y 77 |
| `tipo_establecido` | `columna` | si el tipo se decidio con los dos rasgos o solo con uno | bool | SI | distingue una clasificacion solida de una apretada |
| `sep_pies_m` | `columna` | separacion de los pies durante el evento | m | PARCIAL | hereda la escala. En guardia deberia ser 0,50-0,90 m |
| `umbral_quieto_m` | `columna` | umbral adaptativo por clip para considerar que se detiene | m | SI | es **adaptativo y va guardado**: dos umbrales fijos fallaron su propia prueba de margen |
| `mano` | `columna` | cual es la mano armada | A/B | SI, conjunto reservado | la **cascada** acierta **1,000 con cobertura 19 de 19** en clips que no participaron en elegir la regla, y 0,949 sobre los 59 con verdad. Su peldanio del **pie adelantado** solo cubre **12 de esos 19, sin ningun fallo**. Lo que no decide se marca dudoso |
| `eje` | `columna` | que eje se uso para el brazo en ese evento | texto | SI | `hombro -> raiz` para EXTENSION; el antebrazo es para ORIENTAR la hoja. Confundirlos fue un error real de este proyecto |
| `origen_max` | `columna` | la peor procedencia de pose usada en el evento | codigo | SI | un evento apoyado en pose interpolada es mas debil y aqui se ve |
| `pose_fiable` | `columna` | si las poses del evento son todas medidas | bool | SI | es el resumen de `origen_max` y `hueco_sin_medir` |
| `hueco_max` | `columna` | el hueco mas largo de las poses que usa el evento | cuadros | SI | por encima de 24 el error de la pose **no esta medido** |
| `hueco_sin_medir` | `columna` | si el evento usa una pose de hueco mayor de 24 cuadros | bool | SI | marca el dato cuyo error no es verificable |
| `publicable` | `columna` | si la escala metrica del clip pasa el control fisico **y** el evento pasa la puerta fisica por evento | bool | SI | **2,6 %** de los clips por tirador; la puerta por evento rebaja **156 de 16.202** filas publicables (0,0096) que tenian x o v imposibles (2026-10-09) |
| `publicable_clip` | `columna` | el `publicable` del clip, antes de la puerta por evento | bool | SI | es el que habia hasta el 2026-10-09 |
| `fisica_evento` | `columna` | si el evento pasa la puerta fisica: v media <= 8 m/s, desplazamiento/duracion <= 8 m/s, abs(x) <= 16 m | bool o None | SI | **None** si la fila no trae magnitudes en metros (sin homografia): no medido no es aprobado |
| `motivo_no_publicable` | `columna` | que comprobaciones fallan (`velocidad`, `desplazamiento`, `fuera_de_pista`, unidas por `+`) | texto o None | SI | `ETAPA7_PUERTAS_2026-10-09.md` |
| `puerta_dos_lados` | `columna` | si los DOS tiradores del clip pasan el control fisico | bool | SI | la puerta vieja comparaba la mediana **agrupada** y daba 12,8 %; por tirador es **2,6 %** |
| `ancla_deriva_m` | `columna` | cuanto deriva el origen del mapa a lo largo del clip | m | SI | utilizable (< 1,00 m) en **2 de 11 clips**; el resto lleva el aviso |
| `ancla_fiable` | `columna` | si el origen del mapa se puede creer | bool | SI | 2 de 11 |
| `x_origen_m` | `columna` | donde se puso el cero del eje longitudinal | m | SI | es una **convencion declarada**, no una medida |
| `sin_homografia` | `columna` | si este clip no tiene escala metrica | bool | SI | **6 de cada 10 clips** no la tienen. Cuando es cierto, los campos en metros vienen a **None y NO a cero** |

## 3 · Cono del florete · `cono3d_completo_<clip>_accion.npz`

Solo en la ventana de accion. El cono es la region donde puede estar la hoja.

| campo | forma | que es | unidad | verificado | error medido |
|---|---|---|---|---|---|
| `eje2` | `(300, 2, 2)` | direccion del arma proyectada en la imagen | vector unitario | SI | sale del **antebrazo**: los candidatos que caian al suelo bajan de **63,6 % a 15,7 %**, y el angulo contra la hoja medida pasa de 44,8 grados (el azar) a **14,4** |
| `apert` | `(300, 2)` | apertura del cono, en SEMIANGULO | grados | SI, contra 153 hojas de 12 asaltos | envolvente **+-33,0** (o sea 66 de cono completo). Con la muestra grande el p90 sale 34,3 [24,3, 40,9]. **No se puede afinar mas**: la dispersion entre asaltos (6,3) ya supera el objetivo de 5,0 |
| `L` | `(300, 2)` | alcance del cono desde la muniera | px | SI, contra el reglamento | **1,10 m**: es el alcance muniera-punta (m.8.4 y m.7), no los 0,90 m de la hoja sola desde la guarda |
| `origen` | `(300, 2)` | si la direccion viene de la HOJA VISTA o se supone del brazo | 0 vista, 1 supuesta | SI | **el 97 % de los conos lleva direccion supuesta**. Ver la hoja de verdad **estrecha el cono 3,8 veces**: semiangulo **+-8,6 gr** centrado en la hoja medida contra **+-33,0 gr** centrado en el antebrazo, los dos el p90 de un residuo medido. El **7,9x** que decia antes esta RETIRADO: comparaba los 33,0 con los 4,2 gr del temblor de la DIRECCION de la hoja, que es otra magnitud y con la que el cono no contenia la hoja que centraba en 21 de 60 cuadros |
| `am2` | `(300, 2)` | si el cono amenaza al rival en 2D | bool | PARCIAL | la distancia lateral en 2D es **cota inferior** de la 3D, asi que el cono 2D **nunca pierde una amenaza real, solo le sobran**. Pierde ~la mitad de las situaciones en alcance por vertices mal puestos |
| `am3` | `(300, 2)` | si amenaza en 3D | bool | NO, descartado | `pointmap` **no da profundidad utilizable**: todos los controles fisicos fallan y las 11 discrepancias 2D/3D son **ruido** (barajando las profundidades salen las mismas) |
| `w2` | `(300, 2)` | peso del solape de los dos conos | 0-1 | NO validable | el contacto de hojas **no es verificable a ojo** en este material: de 16 laminas ciegas, **cero decidibles**. El solape 2D es **cota superior** del 3D |

## 4 · Donde fue el tocado · `zona_<clip>.csv` (etapa 11, desde el 2026-10-08)

Una fila por clip. Medido contra `strip_location` en 582 clips anotados (`ESTADO_2026-10-08.md` §2).

| campo | forma | que es | unidad | verificado | error medido |
|---|---|---|---|---|---|
| `motivo` | `columna` | `ok`, `sin luz`, `sin homografia confiable...`, `tocado fuera de las poses` | texto | SI | contestan **0,380**: sin luz 0,40, sin homografia 0,22 (antes del respaldo de luces) |
| `cuadro_tocado` | `columna` | primer encendido del grupo del tocado | cuadro | PARCIAL | sale de la etapa 9 y hereda su error |
| `via_luz` | `columna` | `roi_sede`, `general` o `general_tras_roi` | categoria | SI | el respaldo coincide con la ROI a +-3 cuadros en el **64 %** de 70 clips |
| `cuadro_usado` | `columna` | cuadro con homografia confiable mas cercano al tocado, en +-0,5 s | cuadro | SI | - |
| `x_izq_m / x_der_m` | `columna` | posicion de cada tirador a lo largo de la pista | m desde la linea central | PARCIAL | convenio: x crece hacia la derecha de la imagen |
| `x_tocado_m` | `columna` | la media de las dos | m | **SI, contra verdad humana** | correlacion con la zona **0,387** (n=220); la x de la etapa 7 da **-0,003** |
| `zona` | `columna` | la zona de `x_tocado_m` con +-2 / +-5 m | categoria | **ADOPTADA** (2026-10-08) | 1.313 clips: 0,476 contra 0,353 de `middle`, +0,123 IC95 [+0,090, +0,155]; contra la etapa 7 en 694 clips +0,141 IC95 [+0,094, +0,187]. Con 582 clips no pasaba (IC95 [-0,005, +0,150]) |
| `conf_h / metodo_h` | `columna` | confianza de la homografia y si viene de rayas o propagada con la camara | 0-1 / texto | SI | rayas 0,538 contra propagada 0,508 de acierto |
| `frac_cuadros_con_h` | `columna` | fraccion de cuadros del clip con homografia confiable | 0-1 | SI | - |

## 4b · Posicion de los tiradores cuadro a cuadro · `pista_h_<clip>.npz` (etapa 11) y `scout_<clip>.npz` (scouting v2, desde el 2026-10-15)

`pista_h_`: la homografia de CADA cuadro (`H`, `conf`), los pies de cada tirador en pixeles (`pies`),
`fps`, `cuadro_tocado`, `min_conf`. `scout_` (`codigo/scouting_v2.py`, fase `scouting_v2` del director):
de ahi, la posicion de cada tirador en metros. Validado en los 4.236 anotados de florete
(`04_documentacion/informes/SCOUTING_V2_2026-10-14.md`).

| campo | forma | que es | unidad | verificado | error medido |
|---|---|---|---|---|---|
| `x`, `y` | `(T, 2)` | posicion de cada tirador (0 izq, 1 der) en la pista, medida o interpolada | m desde la linea central / desde el eje | **SI** | \|x\| <= 7,5 m en 98,7 %; zona del tocado 0,495 (etapa 11 0,492, `middle` 0,399; 3.963 clips) |
| `estado` | `(T, 2)` | 0 medido, 1 interpolado (huecos <= 0,5 s), -1 sin dato | categoria | SI | con los dos tiradores en >= 50 % de los cuadros utiles: 56,0 % de los clips |
| `motivo` | `(T, 2)` | por que falta: indice en `motivos` (`sin_h`, `sin_pies`, `fuera`, `salto`, `velocidad`, `cruce`) | categoria | SI | sin homografia el 36 % de los pares cuadro-tirador |
| `x_suave`, `vel` | `(T, 2)` | x suavizada 0,33 s y su velocidad | m, m/s | SI | \|v\| <= 6 m/s en 99,5 % |
| `sep` | `(T,)` | separacion der - izq | m | SI | en [0,5, 14] m en 99,6 % |
| `conf_h`, `util`, `fps`, `cuadro_tocado` | | lo que viene de la etapa 11 y de las poses | | - | - |

Para DONDE esta cada tirador, `scout_`; para cuanto se mueve y los eventos, la etapa 7 (sus
desplazamientos cuadran con los de `scout_`: mismo sentido en el 88 % de 21.417 eventos). La x de la
etapa 7 como posicion acierta menos que contestar `middle`.

## 5 · Luces · `lights_<clip>.csv` (etapa 9), el respaldo

`via = general_tras_roi`: la ROI de sede no encontro ninguna luz y la encontro el detector general.
Recupera **63 de 80** clips que la ROI dejaba sin luz (anaheimfoil2018, qatarfoil2021,
katowicefoil2019 sobre todo), con el cuadro a +-3 de la ROI en el **64 %** de los casos en que
contestan las dos. Menos fiable: separarlo por `via` antes de agregar.

## 6 · Scouting · `03_datos/scouting/` (por corpus, no por clip)

`clip_tirador.csv` (clip, lado), `tiradores.csv` (perfil) y `asaltos.csv` (asalto, tirador).
Reunen lo de arriba con el catalogo y la ficha FIE; **no miden nada nuevo**. Los metros llevan
`frac_escala_publicable`; los conteos y las duraciones no dependen de la escala.

## Lo que NO esta en esta tabla, y por que

| dato | estado |
|---|---|
| **prioridad FIE** instante a instante | el detector acierta **69,7 %** contra **73,8 % de no contestar nunca**. No esta listo para entrenar |
| **paradas** (el parry de esgrima) | **0 anotadas**, y el contacto de hojas no es verificable a ojo en este material. `alto` NO es una parada: es dejar de andar |
| **luces** del aparato | ya existen del pipeline anterior, pero **8,11 %** de 25.139 clips no tienen ninguna, y en una sede falla el **36,78 %**. Hay que auditarlas antes de usarlas |
| **capas densas** (seg, normal, matting) | se extraen y se limpian (fuga de arbitro 0,0000) pero **no van en el NPZ de poses**: cuestan 67 h de GPU sobre el corpus |
| **profundidad** | `pointmap` medido como **inutil**. La `y del pie` ordena mejor y es gratis |

## Lo que esta tabla NO dice

Los errores salen de muestras de **19 a 60 clips** segun el campo, no del corpus de 8.559. Las verdades de referencia de la mano armada, de los pasos y del contacto de hojas **se construyeron a ojo en este proyecto**, no por un arbitro FIE, y estan en fichero para que se puedan contradecir. Y el **error de frontera** de los eventos sigue **sin medir**: es lo que falta anotar.