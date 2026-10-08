# Incidente del 2026-09-23: la noche de extraccion que duro 25 minutos

Se dejo la extraccion corriendo toda la noche. Proceso **25 minutos** y se paro. Esto es que paso,
con hora y prueba, y el arreglo.

---

## El resumen

**El PC no se suspendio ni se apago solo.** Estuvo encendido toda la noche. Lo que paro la
extraccion fue un **borrado recursivo del arbol temporal donde vivian los interpretes de Python**,
a las 23:32. El reinicio de Windows Update de las 05:46 es real pero **llego seis horas tarde**:
para entonces ya no se estaba procesando nada.

## La linea de tiempo, con la prueba de cada linea

| hora | que paso | como se sabe |
|---|---|---|
| 23:14 | se lanzan 5 trabajadores (3 de GPU, 2 de CPU) | `logs_noche/*.log` |
| 23:14-23:39 | **funcionan bien**: 51 poses nuevas, 29 `revision`, 28 series | ficheros con `mtime` en esa franja |
| | ritmo de gpu1 y gpu2: **54,0 y 53,4 clips/h cada uno** | su propia linea `VUELTA 1` |
| ~23:27 | empieza el borrado: `cuda/.venv/Scripts` pasa a tener **1 fichero** de ~20 | `mtime` 23:27:39 |
| 23:32:40 | muere gpu0 con *"una vuelta entera sin completar ni un clip"* | `gpu0.log` |
| 23:33:00 | el borrado esta dentro de `site-packages` | `mtime` del directorio |
| **23:39:26** | **mueren gpu1 y gpu2. Ultimo fichero escrito: 23:39:23** | `poses_BogusSleepyKitfox.npz` |
| 00:19:22 | el borrado llega a los tres `pyvenv.cfg` | `mtime` de los `.venv` |
| 23:39 → 05:46 | cpu0 y cpu1 siguen vivos **girando en vacio**: su entrada ya no se producia | 126 vueltas registradas, 22.807 s |
| **05:46:05** | `MoUsoCoreWorker.exe` reinicia el equipo: *"Sistema operativo: service pack (planeado)"* | Event ID **1074** |
| **05:47:19** | `TrustedInstaller.exe` reinicia otra vez: *"actualizacion (planeada)"* | Event ID **1074** |

**Que el PC estuvo encendido toda la noche lo demuestran los trabajadores de CPU**: 126 vueltas
consecutivas a 180 s cada una son 6,3 h continuas de reloj. Si el equipo se hubiera suspendido,
esa cuenta se habria cortado.

## La causa: un borrado, no un fallo

La prueba de que fue un **borrado** y no una corrupcion:

- todos los directorios que sobreviven tienen `mtime` **23:32-23:33** -- que es lo que pasa cuando
  se borran ficheros de dentro;
- en `cuda/.venv/Scripts` queda **solo `python.exe`**, de unos veinte que habia (`pip`, `activate`,
  los lanzadores...);
- en la raiz del scratchpad **no queda ni un fichero**;
- el numero de directorios de sesion bajo `%TEMP%\claude\C--Users-Alejandro-Desktop\` paso de
  **16 a 5**.

**Sobrevivio exactamente lo que mis tres procesos tenian abierto** -- `python.exe`, `torch`,
`transformers`, `numpy`, `cv2` -- que es justo lo que Windows no deja borrar mientras esta en uso.
Y lo que quedo esta **destripado por dentro**: `import torch` funciona pero `torch.cuda` no existe,
porque se borraron los submodulos que nadie tenia mapeados en ese instante.

Storage Sense **no esta configurado** en esta maquina, asi que la limpieza no es de Windows.

## Por que era fragil, y estaba avisado

Los tres interpretes vivian aqui:

    %TEMP%\claude\C--Users-Alejandro-Desktop\73a45131-...\scratchpad\{fz\.venv, fz\.venv-gpu, cuda\.venv}

O sea: **en el directorio temporal de OTRA sesion**. La memoria del proyecto ya lo decia con todas
las letras -- *"el arbol NO esta en el Escritorio. La base es ...\Temp\claude\...\73a45131..."* --
y se trato como una curiosidad de configuracion en vez de como lo que era: **toda la capacidad de
computo del proyecto colgando de un directorio que cualquiera puede limpiar**.

## El arreglo

**Un interprete duradero, fuera de cualquier temporal:**

    D:\FENCINGLY_PY\venv

Y **uno solo con todo** en vez de tres. Esto ultimo es posible desde que `pipeline.capacidades()`
pregunta "que puedo importar" en vez de "que interprete soy": la separacion en dos venv existia
para que cada etapa supiera si podia correr, y esa pregunta ya se contesta mirando los imports.
Es ademas la configuracion de Colab, asi que Windows y Colab dejan de ser dos caminos distintos.

`rutas.interprete()` lo busca en este orden, y **el temporal queda el ultimo**:

1. `FENCINGLY_PY_YOLO` / `FENCINGLY_PY_HF` si estan puestas
2. `D:\FENCINGLY_PY\venv` (o `FENCINGLY_VENV`)
3. el interprete actual
4. los venv del scratchpad temporal -- solo como ultimo recurso, y **avisando**

## Lo que hay que cambiar ademas, y no es codigo

`noche.py` lanzo los trabajadores y **no se entero de que habian muerto**. Estuvieron seis horas
sin hacer nada y nadie lo dijo. Dos cosas que faltan:

- **un latido**: si el conjunto de ficheros hechos no crece en N vueltas, avisar de forma visible
  en vez de seguir girando en silencio -- los de CPU hicieron exactamente eso, 126 veces;
- **una comprobacion de arranque**: antes de lanzar la noche, verificar que el interprete importa
  lo que hace falta. Hubiera cazado el problema en el segundo 1 en vez de a los 18 minutos.

## Lo que se hizo el 2026-09-23, y lo que salio de rebote

**El entorno nuevo, `D:\FENCINGLY_PY\venv`**, con las versiones clavadas donde importaba:
`numpy 2.5.2` y `torch 2.14.0+cu126` como antes, `transformers 5.16.1` como antes, y lo que no se
pudo clavar apuntado: `pandas 3.0.5 -> 3.0.6`, `ultralytics 8.4.142 -> 8.4.160`,
`sklearn 1.9.0 -> 1.9.1`.

**Y se comprobo que reproduce antes de seguir extrayendo**, con `verifica_reproduce.py`: se
reextraen clips ya hechos en un directorio aparte y se comparan campo a campo. Resultado sobre 4
clips: **34 campos identicos, 2 dentro del ruido de fp16, 0 distintos**. `pres`, `mask` y `origen`
identicos, que era lo que no admitia tolerancia. Mezclar dos entornos en un mismo corpus no se
arregla despues, asi que esto va antes y no despues.

**Un detalle de esa comprobacion que merece quedar escrito.** A la primera dio `KP` con mediana
**0,000000** y maximo **724 px**. No era el entorno: es que el modelo de pose devuelve coordenadas
finitas y absurdas para puntos de confianza casi nula, y un bit de diferencia en fp16 los manda a
otro sitio absurdo. Filtrando por `SC >= 0,3` -- el umbral que usa todo lo que consume el NPZ -- el
maximo baja a **0,000000**. Comparar puntos que nadie usa es medir basura, y habria hecho parar una
corrida buena.

**Y la premisa que se cayo de rebote.** Al instalar todo junto resulto que `ultralytics` y
`transformers` **conviven sin problema** en el mismo proceso. El proyecto llevaba meses afirmando
lo contrario -- *"esa es la realidad instalada, no una preferencia"* -- y de ahi salian los dos
venv, las tres pasadas del playbook y media complicacion de `lotes.py`. Era un accidente de
instalacion que nadie volvio a comprobar. **Windows y Colab son ahora el mismo camino.**

## Lo que NO se perdio

Nada de lo ya extraido. El estado se lee del disco, asi que la reanudacion sigue exacta: los **186
clips** con poses siguen ahi y la corrida continua donde se quedo. Lo unico perdido son las horas
de GPU que no se usaron.
