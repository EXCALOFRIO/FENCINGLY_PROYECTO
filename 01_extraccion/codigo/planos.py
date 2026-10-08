"""Detecta PRIMEROS PLANOS y REPETICIONES, y devuelve la mascara de cuadros utilizables del clip.

LOS TRES DEFECTOS QUE EL USUARIO VIO en `02_validacion/muestras/pareja/` son EL MISMO:
  1. el primer clip empieza con una REPETICION en primer plano de la cabeza de un tirador
  2. en algunos cuadros solo se detecta a una persona
  3. en el ultimo no se detecta a nadie
No son fallos del selector: son cuadros de OTRO plano de camara, donde no hay dos tiradores que
encontrar.

TRES MEDIDAS DESCARTADAS, y por que. No las reintentes:

  `escala_r`, altura de caja normalizada por la mediana del clip: fallo su control positivo dos veces
  (1,58 contra 1,32 de un clip limpio, solapado). El defecto era de disenio: la repeticion es la
  mitad del clip, asi que contamina el normalizador con lo que pretende detectar.

  CORRELACION DE HISTOGRAMA HSV entre cuadros consecutivos: **no ve estos cortes**. Medido en
  `HarshExemplaryGermanspaniel`, su pico maximo (z robusto 464, 286 veces la mediana) resulto al
  mirarlo ser un cambio de FONDO dentro del mismo primer plano, y el corte real al plano en vivo, que
  esta en c109, no lo detecto. El fondo negro domina el histograma en los dos planos.

  UMBRAL ABSOLUTO DE TALLA, `alto de caja > 0,60`, que es lo que habia aqui antes: falla al reves.
  En `RichDefensiveBushbaby` la camara es fija y ancha, y el tirador del lado cercano ocupa 0,609 del
  alto de imagen estando en plano en vivo. El umbral tiraba los cuadros 354-368, que son el final del
  asalto con LOS DOS tiradores a la vista, y esos son exactamente los cuadros donde el usuario vio
  "no detecta a nadie". Una talla absoluta no puede valer para todos los clips porque el encuadre
  cambia de un video a otro.

  Y UNA CUARTA, que es la anterior a esta y tampoco vale: preguntar si LA CAJA MAS GRANDE tiene
  companiera a su profundidad. Marco dos tramos en vivo como si fueran primer plano, y al mirarlos
  los dos eran lo mismo: un ARBITRO o un espectador en primer plano, que es la caja mas grande del
  cuadro y no tiene companiera porque esta solo, con los dos tiradores perfectamente visibles detras
  (`RichDefensiveBushbaby` c72-79 y `defeatedHarlequinRabbitSkinkLizard` c316-321). La pregunta
  estaba mal planteada: da igual si la caja grande esta sola.

  Y UNA QUINTA: preguntar solo si HAY ALGUNA PAREJA PLAUSIBLE en el cuadro, sin mirar la talla.
  Falla al reves otra vez, y en grande: tira 152 cuadros de `RichDefensiveBushbaby` y 168 de
  `defeatedHarlequin`, entre ellos casi todo el asalto (c99-228). El motivo es que la geometria de
  la pareja no se cumple en TODOS los cuadros aunque los dos tiradores esten ahi: una caja cortada
  por el borde del encuadre pierde alto y rompe la razon de tamanio, y en pleno fondo se pasan de
  2,6 alturas de separacion. Con eso, "no hay pareja en este cuadro" no implica "no hay asalto".

LO QUE SE IMPLEMENTA: LAS DOS CONDICIONES A LA VEZ, que es lo unico que separa en lo medido.
Un cuadro es primer plano solo si

  1. la caja mayor pasa de 0,75 del alto de imagen, Y
  2. no hay en el cuadro ninguna pareja que cumpla la geometria de un asalto

Medido sobre los cinco clips, el alto de la caja mayor:

  | tramo                                  |   p50 |   max |
  |----------------------------------------|------:|------:|
  | `Harsh` c0-108, la repeticion           | 0,860 | 0,883 |
  | `Harsh` c109+, en vivo                  | 0,320 | 0,351 |
  | `RichDefensive`, camara fija y ancha    | 0,551 | 0,713 |
  | `defeatedHarlequin`, arbitro delante    | 0,495 | 0,570 |
  | `ZanyMeager`, cuatro pistas             | 0,416 | 0,519 |
  | `VainAching`                            | 0,395 | 0,399 |

El minimo de la repeticion es 0,789 y el maximo en vivo 0,713: **el margen de la talla sola es de
1,11x**, que es poco, y por eso NO se usa sola. La segunda condicion es la que da el margen: en la
repeticion el 73 % de los cuadros tienen UNA sola persona detectada, y en los cinco tramos en vivo
ese porcentaje es **0 %** en todos.
"""
from __future__ import annotations

import numpy as np

ALTO_PRIMER_PLANO = 0.75
"""Alto de caja por encima del cual el cuadro se examina. Cae entre el maximo medido en vivo (0,713)
y el minimo medido en la repeticion (0,789). No decide solo: hace falta ademas que no haya pareja."""
DPIE_PAREJA = 0.30
"""Diferencia de y del pie admitida al buscar pareja en un cuadro. Aqui se usa el tope ABSOLUTO y no
el relativo de `pareja_v2`, porque esta puerta solo tiene que decidir si el cuadro es un primer
plano, y para eso vale ser generoso: el caso que hay que marcar es una cabeza sola ocupando 0,86 del
alto, que no tiene pareja de ningun tipo. Medido: la pareja verificada mas extrema, con la camara a
lo largo de la pista, da 0,220."""
RAZON_TALLA = 1.60
"""Cuanto puede diferir el tamanio de los dos y seguir siendo la pareja. Medido en el caso limite de
`RichDefensiveBushbaby` c360: 0,609 contra 0,512, razon 1,19."""
SEP_ALTURAS = (0.15, 3.20)
"""Separacion entre los dos, en alturas de cuerpo. Los mismos valores y por la misma medida que
`pareja_v2.SEP_ALTURAS`: hay parejas verificadas a ojo a 2,84 alturas y la pareja cruzada entre dos
pistas distintas se va a 5,58."""
MIN_TRAMO = 6
"""Un tramo de menos de 6 cuadros no es un plano de camara: es un salto de deteccion o alguien
cruzando por delante."""
HUECO_UNE = 4
"""Dos tramos separados por menos de esto son el mismo plano con un cuadro perdido en medio."""


def perfil_talla(cajas, cuadros, H, T) -> np.ndarray:
    """QUE HACE: alto de la caja MAS GRANDE de cada cuadro, en fraccion del alto de imagen.

    Se conserva porque sirve para informar y para las graficas, pero YA NO decide sola: decidir por
    talla absoluta es el fallo documentado arriba.
    """
    alto = np.zeros(T, dtype="float64")
    c = np.asarray(cajas, dtype="float64")
    k = np.asarray(cuadros, dtype="int64")
    for i in range(len(k)):
        if 0 <= k[i] < T:
            alto[k[i]] = max(alto[k[i]], (c[i, 3] - c[i, 1]) / H)
    return alto


def _agrupa(idx, T) -> list:
    """QUE HACE: pasa de indices sueltos a tramos continuos, uniendo huecos y tirando los breves."""
    if not len(idx):
        return []
    bruto, ini, ant = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        if int(v) - ant > HUECO_UNE:
            bruto.append((ini, ant))
            ini = int(v)
        ant = int(v)
    bruto.append((ini, ant))
    return [(a, b) for a, b in bruto if b - a + 1 >= MIN_TRAMO]


def _extiende(tramos, sospecha, T) -> list:
    """QUE HACE: alarga cada tramo marcado mientras la caja enorme siga ahi.

    POR QUE HACE FALTA, medido en `HarshExemplaryGermanspaniel`: la condicion de "no hay pareja" se
    apaga en c97, pero la repeticion sigue hasta c108. En c97 aparecen dos personas de fondo que por
    casualidad cumplen la geometria de un asalto, y el cuadro se cuela. La caja enorme, en cambio,
    no se apaga: va de 0,861 en c97 a 0,789 en c108 y cae a 0,290 en c109. El corte de plano esta
    donde desaparece la caja enorme, con un escalon de 2,7x, no donde se apaga la pareja.
    """
    fuera = []
    for a, b in tramos:
        while a > 0 and sospecha[a - 1] > ALTO_PRIMER_PLANO:
            a -= 1
        while b < T - 1 and sospecha[b + 1] > ALTO_PRIMER_PLANO:
            b += 1
        fuera.append((a, b))
    """Se vuelven a unir por si dos tramos crecieron hasta tocarse."""
    unido = []
    for a, b in sorted(fuera):
        if unido and a <= unido[-1][1] + 1:
            unido[-1] = (unido[-1][0], max(unido[-1][1], b))
        else:
            unido.append((a, b))
    return unido


def marca_primer_plano(cajas, cuadros, W, H, T) -> dict:
    """QUE HACE: por cada cuadro decide si HAY alguna pareja plausible; si no la hay, es primer plano.

    QUE DEVUELVE: `sospecha` (el alto de la caja mayor, informativo), `sola` (cuadros sin ninguna
    pareja posible), `tramos` y la `mask` de cuadros utilizables.
    """
    c = np.asarray(cajas, dtype="float64")
    ks = np.asarray(cuadros, dtype="int64")
    alto = (c[:, 3] - c[:, 1]) / H
    pie = c[:, 3] / H
    cx = 0.5 * (c[:, 0] + c[:, 2]) / W

    sospecha = np.zeros(T, dtype="float64")
    sola = np.zeros(T, dtype=bool)
    for k in range(T):
        en = np.nonzero(ks == k)[0]
        if not len(en):
            continue
        sospecha[k] = float(alto[en].max())
        if sospecha[k] <= ALTO_PRIMER_PLANO:
            continue
        if len(en) < 2:
            sola[k] = True
            continue
        a = alto[en][:, None]
        b = alto[en][None, :]
        razon = np.maximum(a / np.maximum(b, 1e-6), b / np.maximum(a, 1e-6))
        dpie = np.abs(pie[en][:, None] - pie[en][None, :])
        alt = 0.5 * (a + b) * H
        sepalt = np.abs(cx[en][:, None] - cx[en][None, :]) * W / np.maximum(alt, 1e-6)
        """La separacion minima no es cosmetica: dos cajas encajadas una dentro de otra son la misma
        persona detectada dos veces, y aceptarlas convertiria cualquier primer plano en utilizable."""
        vale = ((dpie <= DPIE_PAREJA) & (razon <= RAZON_TALLA)
                & (sepalt >= SEP_ALTURAS[0]) & (sepalt <= SEP_ALTURAS[1]))
        np.fill_diagonal(vale, False)
        sola[k] = not bool(vale.any())

    tramos = _agrupa(np.nonzero(sola)[0], T)
    tramos = _extiende(tramos, sospecha, T)
    mask = np.ones(T, dtype=bool)
    for a, b in tramos:
        mask[a:b + 1] = False
    fuera = int((~mask).sum())
    return {"mask": mask, "tramos": tramos, "fuera": fuera,
            "frac_fuera": fuera / float(max(T, 1)),
            "sospecha": sospecha, "sola": sola}


def tramo_utilizable(cajas, cuadros, W, H, T, pres=None) -> dict:
    """QUE HACE: mascara de cuadros utilizables, y cuanto mejora la presencia al aplicarla.

    No parte el clip en trozos: devuelve una MASCARA por cuadro, porque los primeros planos pueden
    estar en medio y partirlo en dos obligaria a elegir mitad, perdiendo datos buenos.
    """
    r = marca_primer_plano(cajas, cuadros, W, H, T)
    if pres is not None:
        m = r["mask"]
        amb_todo = float((pres.sum(axis=1) == 2).mean())
        amb_util = float((pres[m].sum(axis=1) == 2).mean()) if m.any() else float("nan")
        r["ambos_todo"] = amb_todo
        r["ambos_utilizable"] = amb_util
        r["ganancia_pp"] = 100.0 * (amb_util - amb_todo)
    return r
