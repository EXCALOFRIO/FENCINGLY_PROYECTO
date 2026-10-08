# Convenio de FRONTERA, declarado ANTES de que se anote

Fecha: 2026-09-11. Nada está anotado todavía cuando se escribe este fichero.

El usuario ofreció: *«igual me pides que te diga dónde empieza y acaba una marcha, o un romper, o un
fondo, para que lo aprendas mejor»*. **Sí, hace falta**, y hay un hueco concreto: lo medido hasta
ahora es **detección** (recall 85-90 %) y **clasificación** (5 de 6), pero **nadie ha medido si las
FRONTERAS son correctas**. Un evento puede existir, ser del tipo correcto, y tener el inicio 8 cuadros
tarde.

## LA PREGUNTA QUE HAY QUE FIJAR ANTES: ¿cuándo EMPIEZA una marcha?

Hay dos respuestas defendibles y **difieren en 2-4 cuadros**:

  **(A) cuando el pie SE DESPEGA del suelo** — el instante en que se pierde el contacto.
  **(B) cuando el pie EMPIEZA A MOVERSE** — el instante en que arranca, que es antes: el talón se
  levanta y el pie rota sobre la punta sin haber despegado todavía.

**SE ADOPTA (A): la frontera es el DESPEGUE, y el fin es el ATERRIZAJE.** Tres razones:

1. Es un **instante discreto y observable**: se ve si la suela toca o no toca. (B) exige juzgar
   cuándo un movimiento «empieza», que es un umbral y vuelve a meter el problema que este cambio de
   primitiva quita.
2. Es lo que mide el detector nuevo (`posicion.contactos`), así que la anotación y la detección
   miden **la misma cosa**. Si se anotara (B) y se detectara (A), la diferencia sistemática de 2-4
   cuadros se leería como error del detector y no lo es.
3. En biomecánica es la convención estándar (*toe-off* / *foot-strike*).

**Y ESTO NO ES GRATIS, hay que saberlo:** con (A) la marcha «empieza» un poco después de que el
tirador haya iniciado el gesto. Si lo que se quiere es el instante de INTENCIÓN, (A) llega tarde por
construcción. Se adopta (A) porque es medible, no porque sea la única verdad.

## Qué se anota exactamente

Por cada acción de la tira, tres números y un tipo:

| campo | qué es |
|---|---|
| `cuadro_inicio` | el primer cuadro en que **el pie ya no toca** el suelo |
| `cuadro_fin` | el primer cuadro en que **el pie vuelve a tocar** |
| `tipo` | `marcha`, `romper`, `fondo` o `alto` |
| `pie` | `delante` o `atras`, el que se mueve (si se mueven los dos, el que empieza) |

Reglas de desempate, para que la anotación sea reproducible:

- **Si hay duda de un cuadro, se anota el MENOR** (el más temprano). Así el sesgo de la anotación
  tiene un signo conocido en vez de ser aleatorio.
- **Si el pie no se ve** (tapado, fuera de cuadro, movimiento borroso), se escribe `?` en ese campo y
  esa acción no entra en la cifra de error de frontera. Un `?` vale más que un número inventado.
- **`fondo`**: se anota el despegue y el aterrizaje del pie de DELANTE. El de atrás no despega, y si
  despega no es un fondo.
- **`alto`**: los dos pies apoyados. `cuadro_inicio` es el aterrizaje del último pie que se movió y
  `cuadro_fin` el despegue del primero que se mueva después.

## Qué se va a calcular con esto

  **error de frontera de inicio** = `cuadro_inicio` detectado − anotado, en cuadros y en ms
  **error de frontera de fin** = ídem
  **sesgo** (la mediana, que dice si el detector llega tarde o temprano de forma sistemática) y
  **dispersión** (p10-p90, que dice cuánto se puede fiar de un evento suelto)

Si el sesgo es sistemático **se corrige restándolo**, y se publica el antes y el después. Si es cero,
se dice que no hay nada que corregir, como ya pasó con el desfase de detección (+0,5 cuadros).

## Dónde están las tiras y el fichero

  tiras numeradas: `02_validacion/pista/anotar/accion_<clip>_l<lado>_c<cuadro>.png`
  fichero a rellenar: `02_validacion/pista/anotar/fronteras.csv` (ya tiene las columnas y una fila
  por acción, sólo hay que escribir los números)

Cada tira lleva **todos los cuadros de la acción, numerados, uno a uno**, con el marcador de cada pie
en un color distinto, sobre el fotograma ESTABILIZADO para que el suelo no se mueva. Con eso se puede
leer el número del cuadro en que la suela deja de tocar.
