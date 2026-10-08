# Colab: lo que bloquea el arranque, por orden

**Auditoría del 2026-09-11. Se conserva tal cual para no volver a auditarlo.** El usuario decidió
hacer primero las figuras (el cono, el brazo armado, la cascada de tres rondas, el texto y los GIF) y
dejar Colab para la tanda siguiente. Esto es el informe de entonces, con la fecha puesta: **antes de
tocar nada, comprobar que cada punto sigue siendo verdad**, porque un informe de hace días sobre un
árbol que otros agentes escriben es un indicador, no la cosa.

---

## Lo que bloquea el arranque

1. **`lotes.py` NO EXISTE.** La sección 3 entera del playbook documenta una herramienta que nunca se
   escribió: no hay recorrido del corpus, ni `--desde`, ni `--sube`, ni fichero de estado, ni
   reanudar. En Colab eso es lo que salva el lote cuando te echan de la sesión; hoy se pierde entero.

2. **Cinco etapas son no-ops silenciosos, y no fallan: informan éxito.**
   - La **5** (capas densas) no está cableada.
   - La **7** hace `GP.main(...) if hasattr(GP, "main")` y **`grafica_pista` no tiene `main`**, así
     que **`events_<clip>.csv` no lo escribe nunca el pipeline**.
   - Las **8** y **9** sólo devuelven una cadena: **`lights_<clip>.csv` tampoco existe**.

   O sea que **tres de las cinco salidas de la tabla del playbook no se producen**. Hay que cablear
   las etapas **o quitarlas de la tabla**. Lo segundo es tan válido como lo primero, pero una de las
   dos hay que hacerla: prometer un fichero que no se escribe y devolver éxito es la peor de las tres
   opciones.

3. **`cual_interprete()` devuelve `"ultralytics"` en cuanto `import ultralytics` funciona**, así que
   en un Colab unificado —donde los dos juegos de dependencias conviven— **saltaría para siempre las
   etapas 4-9**. Esto no da error: da un lote entero a medias.

4. **Los pesos de Sapiens se cargan por ruta local sin recurso al hub**: en Colab sería
   `from_pretrained("None")`. Los cinco modelos son públicos, así que es un arreglo de rutas, no de
   licencias.

5. **216 rutas absolutas de Windows**: 138 en `01_extraccion/codigo`, 74 en
   `02_validacion/herramienta`, 4 en `pipeline.py`. Y el corpus está en `D:/FENCINGLY_VIDEOS_RAW`
   con la lista de los 8.559 ids **fuera del repo**, en el scratchpad. Hace falta una configuración
   y un `requirements.txt`.

6. **Las tres pasadas NO son una limitación de las librerías.** No hay en todo el árbol **ni un
   conflicto de dependencias medido**: los dos venvs corren **el mismo torch 2.14.0** y sólo cambia
   la rueda de CUDA. En un Colab único debería ser **una sola pasada**. Las tres pasadas son una
   herencia de esta máquina, no un requisito.

---

## El coste está donde no se ha medido nada

**El 83 % de la pasada 1 es YOLO**, cuadro a cuadro, en **fp32**, de uno en uno, a `imgsz 960`. Y
**no existe una sola cifra** sobre saltar cuadros, lotear, bajar `imgsz`, `half=True` ni `yolo11s`.

Todo lo que está optimizado está en la **pasada 2**, que son **28 s** y que además se puede no
correr. Es el sitio clásico: se optimizó lo que era fácil de medir, no lo que cuesta.

`torch.compile` está bloqueado en Windows por falta de `triton` y **en Colab sí hay**; el propio
fichero del proyecto estima **1,3-2x** en un ViT.

### Ya probado y muerto, con su cifra — no repetirlo

| intento | resultado |
|---|---|
| lotes de cuadros en Sapiens | **8 %** y se satura |
| `sdpa` forzado | **7,5x más lento** |
| `bf16` | igual |
| bajar resolución en las densas | **`pointmap` sólo funciona a 1024x768**: su `scale_head` es de tamaño fijo |

---

## El orden, por lo que bloquea

1. `lotes.py` de verdad: recorrido del corpus, `--desde`, `--cuantos`, `--sube`, fichero de estado y
   **reanudar**.
2. Cablear las etapas 5/7/8/9 **o quitarlas de la tabla del playbook**.
3. `cual_interprete()`, que hoy saltaría las etapas 4-9 para siempre en un Colab unificado.
4. Recurso al hub para los pesos, las 216 rutas absolutas a una configuración, `requirements.txt`.
5. **Y sólo entonces el 83 %**: paso del detector, lotes de imágenes, `imgsz`, `half`, `yolo11s`,
   `torch.compile` — cada uno con su criterio declarado **antes** de medir.

---

## Un aviso para cuando se llegue ahí

En GPU dedicada el **`p10` y la mediana de `tiempos.py` tienen que converger**. Hoy no convergen
porque la tarjeta está compartida: `matting` dio **11,9 s** en unos clips y **356,6 s** en otro, un
factor **30** que no es del modelo.

**Si en Colab siguen sin converger, la medida no vale** y hay que buscar por qué antes de optimizar
nada. Optimizar contra un cronómetro que mide otra cosa es cómo se gastan tardes enteras.

---

## Cómo se comprueba que Colab está listo

Un cuaderno limpio que **procese 5 clips y suba**, y que el `p10` y la mediana de los tiempos
**converjan**. Las dos condiciones, no una.
