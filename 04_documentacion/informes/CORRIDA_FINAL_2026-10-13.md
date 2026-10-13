# La corrida final de florete: el flujo, preparado y SIN lanzar (2026-10-13)

La corrida **NO se lanza** hasta que el usuario diga que se acabaron las pruebas. Aqui esta todo lo
que hara, en que orden, cuanto cuesta y que tiene que decidir el usuario antes (seccion 6). Cada
pieza que entra tiene su cifra en su informe (`ESTADO_2026-10-10.md` §1). Lo que no paso va
apagado, con su numero (§2 del mismo).

## 1. Que sale de cada clip (106.432 clips de florete)

| etapa | donde | que | estado |
|---|---|---|---|
| 1 | GPU | YOLO11m-pose en cada cuadro, la pareja de tiradores, planos utiles | entra |
| 4 | GPU | Sapiens 2, 308 puntos por tirador y cuadro | entra |
| 7 | CPU | marcha, retroceso, fondo, brazo, mano armada, posicion; puerta fisica por evento | entra |
| 9 | CPU | las luces del tocado y su cuadro (`fencingly_v2` primero) | entra |
| 11 | CPU | la x del tocado en metros y su zona (rayas pintadas + movimiento de camara) | entra |
| 11 + | CPU | **la homografia de cada cuadro** (`FENCINGLY_GUARDA_H=1`, `pista_h_<clip>.npz`) | **lo decide el usuario** |
| V-JEPA 2 | GPU | resumen de video para quien inicia y la decision del arbitro | entra; **la receta la decide el usuario** |
| DINOv2 | GPU | rasgos cuadro a cuadro | pendiente de su sonda (D0, mie 14/10) |
| despues | CPU | `zona_fusion.py` por torneo (zona final), etiquetador entrenado y etiquetas | entra, al final |

Los interruptores ya estan en su valor adoptado por defecto (`FENCINGLY_LUCES_V2=1`,
`FENCINGLY_PIERNA_K=0.5`, `FENCINGLY_PUERTA_EVENTO=1`, `FENCINGLY_ZONA_FUSION=1`; los que no pasaron,
a 0). Los videos se quedan en `D:\FENCINGLY_VIDEOS_RAW\_descargados` (`--borra-video` solo borra la
copia de `pruebas/videos`), asi que todo lo que sale del video se puede rehacer despues sin bajarlo
otra vez.

## 2. El orden propuesto: los clips con dos luces PRIMERO

La pregunta que mas importa del proyecto es **la decision del arbitro** (`prioridad`). Con los 735
clips de entrenamiento de los anotados no se pudo medir (V-JEPA no se distingue del etiquetador). La
medida buena necesita los ~21.000 clips de florete con las dos luces, y en el orden de siempre
(barajado) estarian completos al final, a las ~7 semanas. La etapa 9 no depende de ninguna otra, asi
que se pasa primero sobre todo el florete. Con eso sale la lista de clips con dos luces, que van
delante. **La medida llega hacia los ~13-14 dias (fases 1, 2A y 3A), no a las ~7 semanas, y sin
calcular nada de mas.**

## 3. Las fases, con su orden de comandos (desde `01_extraccion`, con `D:\FENCINGLY_PY\venv`)

Un solo trabajo de GPU a la vez en todas las fases (`gpu-one-job-at-a-time`). Todo sin ventanas
(`CREATE_NO_WINDOW`) y reanudable: el estado es el disco.

**Fase 0, prevuelo** (minutos):

    python noche.py --prevuelo
    python ../02_validacion/herramienta/tabla_checks.py tabla

**Fase 1, luces de todo el florete** (CPU y red, ~1,5 dias; la GPU queda libre):

    python noche.py --conjunto foil --gpu 3 --etapas-gpu 9 --cpu 0 --descarga

Baja los ~106.000 mp4 (~2,3 MB cada uno, ~240 GB en D:) y escribe `lights_<clip>.csv`. Los
"trabajadores gpu" de este comando solo corren la etapa 9, que es CPU.

**Fase 2A, los clips con dos luces** (GPU, ~21.000 clips a 96 clips/h medidos, **~9 dias**):

    python lista_dos_luces.py --conjunto foil --salida ../03_datos/catalogo/foil_dos_luces.txt
    set FENCINGLY_GUARDA_H=1        (si el usuario dice que si, §6)
    python noche.py --conjunto ../03_datos/catalogo/foil_dos_luces.txt --gpu 3 --cpu 2 --descarga

**Fase 3A, V-JEPA sobre esos clips** (GPU, ~21.000 x ~6,4 s, **~1,6 dias ESTIMADOS**):

    python noche.py --para
    python codigo/vjepa_variantes.py --todos --conjunto ../03_datos/catalogo/foil_dos_luces.txt ^
        --modelo vjepa2-vitg-fpc64-256 --salidas rec4,mic4 --destino D:/FENCINGLY_VJEPA/g2

Y la sonda de `prioridad` con las ~21.000 decisiones (GPU, ~1-2 dias). Con el protocolo de
`VJEPA2_2026-10-09.md` pregunta 2: mismo reparto por torneo, misma rejilla y criterio, declarado
antes de mirar.

**Fase 2B, el resto del florete** (~85.000 clips, **~37 dias**):

    python noche.py --conjunto foil --gpu 3 --cpu 2 --descarga

(salta solo lo ya hecho).

**Fase 3B, V-JEPA sobre el resto**: G2 (~6,3 dias) o V0 (~1,2 dias), **segun lo que diga la
`prioridad` medida en la fase 3A**.

**Fase 4, al final** (CPU, horas): `python zona_fusion.py --conjunto foil --escribe`, entrenar el
etiquetador y etiquetar, y las comprobaciones de `PLAYBOOK.md` §4 (`tabla_checks.py`,
`busca_errores.py`).

El respaldo a HF privado (`respalda.py`, cada 2 h) va con `noche.py`, como en los anotados. Los
embeddings de V-JEPA se quedan en D: (~120 GB solo los de dos luces; se pueden rehacer desde los
videos).

## 4. Cuanto tarda, todo junto

| fase | dias | GPU |
|---|---:|---|
| 1 luces | ~1,5 | libre |
| 2A dos luces | ~9 | etapas 1 + 4 |
| 3A V-JEPA G2 + sonda `prioridad` | ~3 | V-JEPA y sonda |
| 2B resto | ~37 | etapas 1 + 4 |
| 3B V-JEPA resto | ~1-6 | V-JEPA |
| **total** | **~52-57 (7,5-8 semanas)** | |

Si DINOv2 pasa su sonda: +15-28 dias. Disco: C: ~150 GB (poses), con 350 libres; D: ~240 GB de
videos + 120-610 GB de V-JEPA, con 1.410 libres.

## 5. Lo preparado hoy (sin cambiar ninguna cifra)

- `vjepa_variantes.py`: salida `mic4` (el micro a 4x4, lo que midio G2; la cuarta parte de disco
  que `mic8`). Se comprueba que `mic4` sea el 2x2 de `mic8` en la medicion de coste de G2 (200
  clips, al final de la cola).
- `zona_tocado.py`: `FENCINGLY_GUARDA_H` (por defecto 0). Probado en 6 anotados sin escribir en
  `pruebas`: los campos medidos de `zona_` son identicos (la unica diferencia, `via_h`, es que esos
  CSV son de antes de existir la columna). 6-15 KB por clip, ~1,5 GB en florete, 0 s de mas. En los
  4 clips con respuesta hay homografia confiable en 225-255 cuadros de cada clip hasta el tocado.
- `lista_dos_luces.py`: la regla de `sonda_vjepa.filas_prioridad`. En los anotados: 1.119 de 5.856.

## 6. Lo que tiene que decidir el usuario ANTES de lanzar

1. **Cuando**: la cola de pruebas acaba con DINOv2 (mie 14/10) y G4 (jue 15 - vie 16/10).
2. **El orden** del §2: las dos luces primero. Recomendado: si.
3. **V-JEPA**: G2 (ViT-g, `rec4 + mic4`) en los de dos luces, y el resto se decide con la
   `prioridad` medida. Recomendado: si.
4. **La homografia de cada cuadro** (`FENCINGLY_GUARDA_H=1`). Recomendado: si. No cuesta nada y es
   lo unico que permite despues la posicion en metros cuadro a cuadro y el registro entre clips sin
   volver a leer 106.000 videos.
5. **G4** (ViT-g a 384, solo el ojo meso): correrla o quitarla. Quitarla ahorra ~1 dia de GPU. Es
   poco probable que cambie la receta: en el ojo meso el modelo grande no aporto (G0 contra V0
   +0,001), y a 384 cuesta ~3 veces mas.
