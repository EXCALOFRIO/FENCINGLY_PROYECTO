# Calidad de las poses: la auditoría que faltaba

Medido el **2026-09-09** sobre **10.840 clips** de 21 sedes y **1.776.368 pares de cuadros**. Nadie
había puesto una cifra a esto, y «los mecanismos existen» no es una validación.

---

## 1. Lo que envenena el entrenamiento

Estos tres son los que corrompen la etiqueta o la geometría, no los que solo quitan datos.

| detector | clips | lectura |
|---|---:|---|
| **Cambio de identidad** (≥2 cambios de signo y >5 % invertido) | **0,43 %** | el peor fallo posible, y es raro |
| **Teletransportes** (≥3 saltos por encima de 10 m/s) | 6,80 % | **pero mira el §2: es de cuadros** |
| **Huesos inestables** (cv hombro-cadera > 0,35) | **0,52 %** | el esqueleto casi nunca salta de cuerpo |

El cambio de identidad es el que más importa aunque sea el más raro: **la etiqueta depende del lado**,
así que intercambiar los tiradores a mitad de clip invierte la respuesta correcta. Que esté en el
0,43 % es la mejor noticia de esta auditoría.

## 2. El teletransporte es de CUADROS, no de clips

Esta distinción decide qué se hace con él, y decirlo mal invitaría a tirar 700 clips.

| | |
|---|---:|
| pares de cuadros medidos | **1.776.368** |
| teletransportes | 4.468 |
| **fracción de CUADROS** | **0,2515 %** |
| clips con ≥ 3 | 6,80 % |
| fracción afectada por clip, **mediana** | **0,0000 %** |
| p90 / p99 / máximo por clip | 0,85 % / 4,02 % / 16,3 % |

**Uno de cada 400 cuadros.** El arreglo es descartar esos cuadros, no esos clips.

Y la distribución de velocidad lo confirma: mediana **3,4 m/s** (plausible para un tirador),
**p90 16,1** y **p99 37,9 m/s**. Lo primero es esgrima; lo segundo es imposible y por eso se detecta.

**Salvedad honesta:** la velocidad usa la escala supuesta (caja del tirador ≈ 0,388 del alto de
imagen). Un error de escala inflaría el número — pero **no seis veces**, así que el p99 no se explica
por eso.

## 3. Lo que solo reduce datos

| | |
|---|---:|
| más de 1 s seguido sin pose útil | 5,09 % |
| más de 2 s seguidos | 2,33 % |
| **fracción útil, mediana** | **1,000** |
| un solo tirador, fracción mediana | 0,000 |
| segundos perdidos, p90 / p99 | 0,40 s / 2,88 s |

**La mediana de fracción útil es 1,000**: en el clip típico, *todos* los cuadros tienen los dos
tiradores con al menos 5 de los 8 puntos que usan las medidas de prioridad. Eso es lo que hay que
retener: el clip típico está limpio y los problemas están en una cola.

## 4. Y el hallazgo que vino de otro sitio

Generando la muestra de cazoleta apareció algo que esta auditoría no mide: en varias celdas **el
recorte no contenía ninguna cazoleta, solo fondo vacío**. El recorte se centra en la muñeca de la
pose, así que **eso significa que la muñeca estaba en el fondo** — un punto con confianza suficiente
para pasar el filtro y colocado donde no hay nada.

Es un modo de fallo distinto de los cinco de arriba: no es que falte la pose ni que salte, es que
**está donde no debe con confianza normal**. Ninguno de los detectores actuales lo caza, y no hay
cifra para él todavía. Queda como hueco declarado.

## 5. Qué hacer con esto

- **Descartar cuadros, no clips**, para los teletransportes: son el 0,25 % de los cuadros.
- **Marcar y excluir** los del 0,43 % con cambio de identidad: ahí la etiqueta está mal, no ruidosa.
- Los 5,09 % con más de un segundo perdido **sirven igual** si la ventana de entrenamiento no cae
  dentro del hueco; es una decisión de la rodaja, no del clip.
- Y **ninguno de estos umbrales borra nada por su cuenta**: todos salen como columnas para que quien
  entrene elija, porque un umbral puesto aquí sería un umbral inventado.

## 6. Las peores sedes, por si hay que priorizar

| sede | n | clips graves |
|---|---:|---:|
| eurochamps2022 | 487 | 32,2 % |
| bonnfoil2022 | 549 | 25,9 % |
| tokyoolympics2020 | 370 | 22,4 % |
| serbiafoil2022 | 803 | 19,7 % |
| katowicefoil2019 | 375 | 17,6 % |

Ojo con leer esto como «calidad de la sede»: ya nos pasó que una tabla por sede medía **cuánto se
había rebuscado en el residuo** y no la sede. Aquí el sesgo posible es distinto —número de pistas
visibles, distancia de cámara— pero el aviso vale igual.
