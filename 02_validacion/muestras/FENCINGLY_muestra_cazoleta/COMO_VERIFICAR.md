# Como verificar la muestra

Abre `muestra.png`. Cada celda lleva un numero, un **circulo amarillo** donde el prior dice
que deberia estar la cazoleta, y una **caja roja** que es la propuesta del modelo.

En `verificar.csv`, para cada `n` pon en `veredicto`:

  `si`     la caja roja esta sobre la cazoleta (o sobre cazoleta + guante-empunadura)
  `no`     la caja roja NO esta sobre la cazoleta
  `dudoso` no se ve bastante para decidir

**No mires `CLAVE_no_mirar.csv`.** La muestra lleva **controles mezclados**: algunas cajas
las genero un prompt senuelo y otras estan puestas al azar. Sirven para medir si la
verificacion es fiable, y solo funcionan si no sabes cuales son.

Marca lo que veas, sin intentar ser generoso. Un `no` vale igual que un `si`.
