# Reanudar tras el reinicio (2026-09-25)

El reinicio mata todos los procesos en marcha pero **no pierde nada**: el estado de qué clip está
hecho en qué etapa vive en disco (los ficheros `poses_*.npz`, `serie_eventos_*.csv`, etc.), nunca en
memoria de un proceso. Al relanzar, cada script vuelve a mirar el disco y sigue donde estaba.

## Qué estaba corriendo y hay que relanzar

Abre una terminal en `C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion` y lanza, cada uno
en su propia ventana (o con `&` en background):

### 1. Extracción de poses (etapa 1+4) — la prioritaria, usa GPU

```
D:\FENCINGLY_PY\venv\Scripts\python.exe -u bucle.py --conjunto anotados --etapas 1,4 --desde 0 --cuantos 5856 --lote 40 --espera 180 --vacias-max 400
```

Estado a las 08:57 del 25/09: **75,3 % hecho (4.407 de 5.856)**. Esta es la que se había parado sin
avisar la noche del 23-24/09 (proceso huérfano de una sesión anterior) — por eso el número clave a
vigilar tras relanzar es que las poses en disco SIGAN subiendo, no solo que el proceso esté "vivo".

Verificar que avanza:
```
D:\FENCINGLY_PY\venv\Scripts\python.exe -c "
import sys, glob, os, time
sys.path.insert(0,'codigo')
import rutas as RU
ps=glob.glob(str(RU.TRABAJO/'poses_*.npz'))
print('hechos:', len(ps))
"
```
(el directorio correcto es `01_extraccion/pruebas/`, NO `D:/FENCINGLY_TRABAJO` — es donde vive
`RU.TRABAJO` de verdad, confirmarlo con `rutas.py` si hay dudas).

### 2. Etapa 7 (eventos + pista), en dos mitades

```
D:\FENCINGLY_PY\venv\Scripts\python.exe -u bucle.py --conjunto anotados --etapas 7 --desde 0 --cuantos 2928 --lote 40 --espera 180 --vacias-max 400
D:\FENCINGLY_PY\venv\Scripts\python.exe -u bucle.py --conjunto anotados --etapas 7 --desde 2928 --cuantos 2928 --lote 40 --espera 180 --vacias-max 400
```

Estos consumen las poses ya extraídas por la etapa 4; no compiten fuerte por GPU (son CPU-bound), y
pueden convivir con la etapa 4 en paralelo sin relanzar en orden especial.

### 3. Lo que NO hace falta relanzar

- `zona_datos.py` — ya terminó (4.888 de 4.889 clips, fotogramas escritos en `D:/FENCINGLY_ZONA/`).
- `zona_modelo.py` (por fotograma) — ya entrenó, pesos en
  `03_datos/modelos/zona_pista.pt` (acierto 0,430 en el reservado).
- `zona_clip.py` (por clip, con atención) — **si no llegó a entrenar antes del reinicio**, relanzar:
  ```
  D:\FENCINGLY_PY\venv\Scripts\python.exe -u zona_clip.py --epocas 10
  ```
  Comprobar si ya existe `03_datos/modelos/zona_clip.pt` antes de relanzar.

## Qué NO relanzar

- `porclip.py` (el script del scratchpad que compara acierto por-fotograma vs por-clip) — es
  diagnóstico puntual, no hace falta que sobreviva al reinicio. Si se quiere repetir el número,
  usar `num_workers=0` en el `DataLoader` (el bug de multiprocessing en Windows sin
  `if __name__ == "__main__"` ya se parcheó así una vez).

## Verificación tras relanzar todo

```
D:\FENCINGLY_PY\venv\Scripts\python.exe -c "
import sys, glob, os, time
sys.path.insert(0,'codigo')
import rutas as RU, catalogo as CA
ahora=time.time()
ps=glob.glob(str(RU.TRABAJO/'poses_*.npz'))
tot=len(CA.anotados())
print('poses: %d de %d (%.1f%%)'%(len(ps),tot,100*len(ps)/tot))
"
```

Esperar ~15-20 min tras el arranque (carga de modelos YOLO/Sapiens) y volver a mirar que el número
haya subido. Si no sube en 30 min, la etapa 4 se ha vuelto a parar y hay que mirar el log.
