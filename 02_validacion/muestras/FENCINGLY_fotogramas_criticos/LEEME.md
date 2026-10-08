# Fotogramas criticos

Una carpeta por modo de fallo. Cada imagen lleva el **cuadro donde el fallo es maximo**, los dos
esqueletos en colores distintos, sus cajas, la banda de pista activa en gris, y arriba los
numeros medidos que metieron ese clip en esa carpeta.

**Empieza por `0_LIMPIOS_busca_mis_fallos`.** Son clips que mi algoritmo declara limpios: si
encuentras ahi un arbitro, un espectador, alguien de otra pista o una pose mal puesta, es un
**falso negativo mio**, y eso vale mas que confirmar cualquiera de mis positivos.

En `INDICE_y_tus_veredictos.csv` tienes una fila por imagen con `mi_veredicto`. Rellena
`tu_veredicto` con `deacuerdo`, `falso_positivo` o `es_peor` y usa `comentario` para lo que veas.

Lo que ninguna de estas carpetas caza, y lo digo porque es un hueco real: una pose **colocada en
el fondo con confianza normal**. No falta y no salta, simplemente esta donde no hay nadie. Si la
ves, dimelo, porque no tengo detector para eso todavia.
