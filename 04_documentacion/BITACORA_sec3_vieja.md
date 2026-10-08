## 3. La cifra que manda: 23,5 millones de pasadas

Medido sobre 15.452 clips: **342 cuadros de media** por clip. Para 34.374 clips, todos los
fotogramas, dos tiradores, y siendo top-down (una pasada **por persona y por cuadro**):

| | |
|---|---:|
| cuadros totales | **12 millones** |
| **pasadas de Sapiens** | **23,5 millones** |

| si rinde | tarda |
|---|---|
| 5 pasadas/s | **54 días** |
| 10 pasadas/s | 27 días |
| 40 pasadas/s | 6,8 días |
| 150 pasadas/s | 1,8 días |

Un ViT de mil millones de parámetros a 1024×768 en una 3060 estará en la banda baja, así que **el 1b
sobre todo el corpus no cabe en esta máquina**. Por eso el plan empieza midiendo el rendimiento real
**con un vídeo** y con el **0.4b**, no con el 1b.

