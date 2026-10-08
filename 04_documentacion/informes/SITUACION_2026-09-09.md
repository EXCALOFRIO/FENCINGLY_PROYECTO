# FENCINGLY — situación completa y próximos pasos

**2026-09-09, tarde.** Este documento es la puerta de entrada. Los demás quedan como anexos:
`QUE_VAMOS_A_DETECTAR.md` (señal por señal), `ANTES_DE_COLAB.md` (comprobaciones), `NOCHE_2026-09-09.md`
(bitácora de la noche), `FENCINGLY_resultados.pdf` (el informe científico).

---

## 1. La respuesta corta

**El objetivo de hoy está cumplido: la extracción definitiva está preparada y probada.** El cuaderno existe,
sus siete pasos se han ejecutado aquí de punta a punta, y todo está en un repositorio privado de Hugging Face
del que se puede partir si Colab se corta.

**Y la respuesta honesta sobre el 73,6 %: sabemos por qué está ahí, y no es un problema de modelo.** Es que el
modelo está aprendiendo un atajo. Eso es lo importante de estas dos semanas, más que cualquier décima.

---

## 2. Por qué el modelo toca techo: el atajo del orden de luces

Esta es la parte que cambia el plan, así que va primero.

**El orden en que se encienden las luces predice el ganador el 63,51 % de las veces por sí solo**, sin mirar
nada más. Y el modelo lo ha aprendido: predice «el que encendió segundo» en el **72,71 %** de los clips.

Partiendo los clips según quién encendió primero:

| | clips | acierto |
|---|---:|---:|
| gana el que encendió **segundo** | 1.671 | **83,00 %** |
| gana el que encendió **primero** | 960 | **45,21 %** |

**Una diferencia de 37,80 puntos.** Y crece con el desfase entre luces: **+26,59 / +38,16 / +44,45** según el
desfase sea corto, medio o largo. El contraste largo-menos-corto es **+17,39 pp [+4,96, +29,41], p = 0,003**.

Los 960 clips donde gana el que encendió primero están **por debajo del azar**. El modelo no es que no sepa:
es que está prediciendo activamente al revés en ese tercio del corpus.

### El premio, y el riesgo

- Subir esos 960 al **60 %** da **+5,40 pp** globales. Al **75 %**, **+10,95 pp**.
- Pero si al arreglarlo se pierden 5 puntos en los otros 1.671, eso cuesta **−3,18 pp**. No es gratis.

### Y un giro que hay que entender

Encontré **14 columnas de inicio de movimiento mal definidas** (el segundo tirador se medía contra la luz del
otro). Al arreglarlas, **la diferencia no bajó: subió +2,94 pp.** Porque una medida más limpia de quién se
movió primero **correlaciona mejor** con quién encendió primero.

**O sea que parte del «atajo» es el reglamento.** En florete, quien inicia la acción correctamente suele tener
la prioridad, y la luz refleja eso. Destruir la señal de la luz destruiría también información legítima. Por
eso las tres técnicas de eliminación de atajos que me pasaste no son un botón que apretar.

### El límite de potencia, que conviene saber

Con el corpus actual, **una mejora de +1,6 pp es indetectable**: se confunde con el ruido. El suelo de ruido
medido es **0,8 pp**, calibrado con un bloque nulo que llega a **ganar +0,69 pp** sin información.

Para medir mejoras de ese tamaño harían falta **entre 12 y 20 campeonatos más**. Los vídeos que estás
consiguiendo no son un extra: son el camino.

---

## 3. Lo que está listo para Colab

### El cuaderno definitivo

`FENCINGLY_extraccion_completa.ipynb` — 18 celdas, 8 de código. **El cuaderno no contiene lógica**: llama a
`pasos.py`, que se ha ejecutado aquí completo. Lo que corra en Colab es lo que se ha medido aquí.

| paso | qué comprueba antes de seguir |
|---|---|
| 0 puertas | versiones, huella del productor, parche de ventana, GPU — **aborta si falla** |
| 1 manifiesto | esquema por fuente, **`weapon == "foil"` explícito**, admite JSON o URL nuevos |
| 2 estado | `index.csv` ∩ `.npz` en disco — **nunca un contador** |
| 3 extracción | por clip, `f0 == 0` y `f1 == n_frames` (clip entero, sin recortar) |
| 4 índice | 119 columnas, fila solo si el `.npz` existe, y luego **el lector de verdad** |
| 5 rodaja | **mando separado, `aplicar=False`** — informa y no toca nada |
| 6 tabla | Parquet por `(clip_id, frame_idx)`, `NaN` = indefinido |
| 7 huecos | lo que no se detecta, con su cifra |

Verificado por mí, no solo por el agente: **puertas OK con `producer_ok=True`**, el filtro de florete rechaza
espada y los clips sin arma, la reanudación dice **0 pendientes**, el lector reporta **`invisibles=[]`** y la
tabla sale con 1.062 filas.

### Tres fallos que la prueba local cazó y Colab habría escondido

1. **La puerta dio un OK falso con el árbol congelado.** Devolvía una huella distinta (`fd2421b3…` en vez de
   `3e1dbe0a…`) **sin dar error**. Ahora toma dos raíces separadas: la de extracción y la de medida.
2. **La reanudación con 0 pendientes rompía todo lo siguiente** con un `KeyError`. Y esa ruta se ejecuta
   **cada vez** que se relanza el cuaderno después de acabar.
3. **La verificación al reanudar era vacía**: comprobaba contra un índice vacío, o sea un OK que no comprueba
   nada. La misma familia que el contador que decía «completado = 3» habiendo 6 ficheros.

### El respaldo en Hugging Face

`EXCALORIO/fencingly-extraccion` — **dataset privado, 20 ficheros**, verificado privado en cada subida.
Estructura: `notebooks/`, `docs/`, `data/{poses,crops,tables,index}/`. El token se lee del `.env` y **no
aparece en ningún cuaderno, log ni informe**; en Colab va como secreto.

Privado a propósito: los recortes son fotogramas de retransmisión y los índices llevan nombres de tiradores.
Caben en los 100 GB gratuitos, así que **no hace falta publicar nada**.

---

## 4. Lo que se ha arreglado hoy

**El bug de la ventana de guardia, aplicado y verificado.** Recupera **483 de 488 clips** bloqueados
(bloqueados 1.341 → 858; entran 14.110 → 14.593), y quedan **+5 frente a hoy**, no 0 — no lo redondeo.

Lo importante: **la huella del productor no se mueve.** Medida ahora mismo:
`3e1dbe0ad8a4c706…9827` y movimiento `6c9fa66b2f4614c1…3ce7`. Así que **los 23.091 artefactos siguen
válidos** y no hay que rematerializar. Batería idéntica antes y después: **1.321 / 0 fallos / 0 errores**.

**La cobertura de vídeos.** De 23.567 a **29.787 de 34.374 (86,7 %)**. Tiradores con el 100 % de sus vídeos:
de 14 a **47**. Mediana por tirador: 74,4 % → **88,2 %**. Parejas de entrenamiento: 4.958 → **6.465 (+30,4 %)**.

---

## 5. Tres trampas nuevas, y una es grave

**1. Parchear un módulo en caliente ENVENENA EL ALMACÉN PARA SIEMPRE.** `build_store` graba en `store.json` un
hash de las funciones vivas de `dataset.py`. Un envoltorio sobre `load_sample` deja congelado un hash que
**ningún proceso limpio reproduce**, y el almacén queda inabrible con los datos perfectos dentro. Verificado en
el código: `dataset.py` está en `SEMANTIC_SOURCE_FILES`, se hashea, `store.py` lo graba y lo valida al abrir.

**2. La cadencia nominal no es el intervalo de muestreo.** La banda de 30 fps es **telecine 25→30 en 8.503 de
11.211 clips (75,8 %)**: un fotograma repetido **cada seis exactos**, que da velocidad cero en un paso y doble
en el siguiente. A **50 fps no hay ni un clip duplicado**, así que su paso de 20 ms es real. La cámara lenta de
verdad son **54 clips (0,35 %)**, con ratios de hasta **12,0**.

**3. Las repeticiones de jugada contaminan las poses sin activar ninguna alarma.** Un clip que abre con
primeros planos de repetición marca cobertura de pose **1,000 en los dos tiradores**: el detector los encuentra
porque **están ahí**, solo con otro encuadre. **Cuatro vías probadas, ninguna discrimina.** Va como hueco
declarado, con `escala_r` de columna cruda.

Y sobre esa última: **no se bajó el umbral de 1,60 a 1,55 para que el control positivo pasara.** Habría sido
inventar la medida.

---

## 6. Lo que NO funciona, con su cifra

Está aquí para que no se gaste tiempo otra vez. Cada línea es una vía cerrada **medida**.

| vía | resultado |
|---|---|
| Profundidad monocular para escala | **46-51 % de error** sobre un vano de 4,00 m |
| Detección de líneas de pista | 3,8 % contra 4,3 % de una referencia **asistida por humano** |
| Transformador de vídeo (VideoMAE) | **+0,08 pp** [−0,51, +0,79] |
| Vorticidad del flujo óptico | +0,60 pp [−0,07, +1,33] tras su nulo |
| Vacilación en la preparación | **AUC 0,4997** |
| Aterrizaje del pie delantero | **−1,12 pp**, peor que sus dos nulos |
| Centro de masas | señal **real** (+0,39 pp), cerrada por redundancia (r = 0,877) |
| Choque de hojas por audio | 20-28 % con 10 % de falsa alarma |
| Panorama estéreo del audio | ρ = **+0,002** (p = 0,829) |
| Segmentación de hoja | 8 de 9 métodos clásicos detectan **0,00**; SAM2, 2 de 15 |

**Un cierre no significa que la señal sea falsa.** El centro de masas le gana a una copia de sí mismo con el
lado destruido: su señal es real, solo que el modelo ya la tenía por otra columna. **Se extrae y se guarda
igual**, que es tu criterio y es el correcto.

---

## 7. Próximos pasos, por orden

1. **Lanzar la extracción definitiva en Colab.** Está listo. Es el paso que desbloquea todo lo demás.
2. **Las tres señales nuevas** (en curso): tiempo de deliberación del árbitro, velocidad angular del antebrazo,
   y eje del antebrazo contra el rival como sustituto de «punta en línea». La tercera es la interesante y la
   que más riesgo de circularidad tiene: hay que compararla **contra el atajo**, no contra el azar.
3. **Mapear `lotes/<sede>/` a la estructura de HF** y probar la reanudación contra el remoto. La subida
   autenticada está probada; **la reanudación contra el remoto no**. Es cambio de rutas, no de lógica.
4. **Sacar el proyecto de `AppData\Local\Temp`.** Sigue viviendo en un directorio temporal de otra sesión.
   Hay respaldo en `D:\FENCINGLY_respaldo_2026-09-09` (3,35 GB), pero esto es deuda que hay que pagar.
5. **Fijar el envenenamiento del almacén con una prueba** para que nadie lo resucite.
6. **Los campeonatos nuevos.** Son la única vía conocida para medir mejoras por debajo de 1,6 pp.

---

## 8. Lo que no te voy a vender

- **Ninguna de las ideas probadas ha movido el acierto de forma medible todavía.** Lo que ha cambiado es que
  ahora sabemos **dónde está el techo y por qué**, y eso vale más que una décima que no se sostenga.
- **Publiqué dos cifras que luego tuve que retirar** (un ≈72,9 con una sola causa cuando eran dos, y un
  +1,03 pp como «la primera mejora real»). Las dos están corregidas en el informe.
- **La región del +11,3 pp no se puede usar** para justificar una decisión: está definida por el resultado en
  las dos mitades, así que no puede tasar un rasgo ni una puerta.
- **`GROUND_HORIZON_Y = 0.14` está en producción sin registro de la corrida que lo fijó.** Deuda declarada.
