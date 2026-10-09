# El angulo del borde lejano JUNTO a la homografia: protocolo DECLARADO ANTES de medir

Escrito el 2026-10-09 (tarde), antes de cruzar con `strip_location` ninguna de las reglas de abajo.
Lo que se cambie despues de ver cifras se apunta abajo como cambio, con su fecha.

## Punto de partida

`RAYAS_ID_2026-10-09.md` (agente H): el angulo del borde LEJANO de la pista en el cuadro del tocado,
con un modelo por torneo `x = a + b*tan(ang) + c*desplazamiento de pies` (Huber) ajustado sobre los
clips del mismo torneo que SI tienen homografia (la salida de produccion, sin etiquetas), da la x
con mediana 0,73 m (AJUSTE) / 0,67 m (PRUEBA) dejando uno fuera. Informativo en AJUSTE: en esos
clips CON homografia, la zona sacada solo del angulo acerto 0,511, la de la homografia 0,485. Como
RESPALDO en los clips sin homografia no paso en PRUEBA. Aqui se prueban sus dos propuestas.

Rasgos: los del agente H (`02_validacion/auditoria/rayas_id_2026-10-09/rasgos.jsonl.gz`, 5.525 clips,
el cuadro del tocado y los dos anteriores) y su modelo (`modelo.py`: Huber, con desplazamiento,
>= 8 referencias por torneo).

## Reparto y lo que de PRUEBA ya se ha mirado

El de `ZONA_COBERTURA_2026-10-09.md`, sin tocarlo (`torneos_ajuste.txt` = AJUSTE, el resto PRUEBA).
De PRUEBA ya se miro contra etiquetas, en agregado: 79 clips sin homografia (ZONA_COBERTURA, regla
B(0,25)+p) y 533 clips sin homografia (RAYAS_ID, G(8, inf)); y, sin etiquetas, el error en metros
del modelo del angulo en sus referencias. Tambien el acierto global de PRODUCCION de la zona por
arma y via (`mide_zona.py`, las dos mitades juntas). **Ninguna fusion ni ningun prior de los de
abajo se ha cruzado con etiquetas de PRUEBA.**

## Idea 1: FUSION en los clips que ya contestan (barata: no rehace la etapa 11)

Solo actua en clips con `motivo = ok` cuyo torneo tiene modelo del angulo (>= 8 referencias SIN
contar el propio clip) y con angulo y desplazamiento medidos. **El modelo se ajusta dejando fuera el
propio clip**, tambien en produccion: si no, la x del angulo de un clip estaria ajustada a su propia
x de homografia. `x_h` = `x_tocado_m` de produccion, `x_a` = la del angulo. Familia fijada ahora
(21 candidatas):

- `W(w)`: `x = w * x_h + (1 - w) * x_a`, con w en {0; 0,25; 0,5; 0,75}
- `D(d)`: `x = x_a` si `|x_h - x_a| > d` m, si no `x_h`, con d en {1; 2; 3}
- cada una con la puerta de calidad del torneo `e_T <= e`, e en {1,0; 1,5; inf}, donde `e_T` es la
  mediana de `|x_a - x_h|` dejando uno fuera en las referencias del torneo (no usa etiquetas).

La zona sale de esa x con la particion de siempre (+-2, +-5).

## Idea 2: el angulo como PRIOR dentro de `hypothesize` (rehace la etapa 11)

En `pista_lineas._scale_prior`, el prior de "los tiradores suelen estar cerca del centro"
(`-0,5 (x/5)^2` por tirador) se SUSTITUYE, en cada cuadro con angulo del borde lejano y torneo con
modelo, por `-0,5 ((x_medio - x_a(t)) / s)^2` con `x_medio` la media de los pies de la hipotesis y
`x_a(t)` la x del angulo en ESE cuadro (el modelo del torneo, dejando fuera el clip). Familia:
`P(s)`, s en {1,5; 3,0} m. Sin angulo o sin modelo, el prior de siempre. Cambia las semillas, asi que
puede cambiar clips que ya contestaban y subir la cobertura. Cuesta rehacer la etapa 11 (~12 s/clip),
asi que se mide en una MUESTRA al azar (semilla 20261009): 600 clips con etiqueta de AJUSTE para
elegir `s`, y 1.000 de PRUEBA para evaluar (o todos los de PRUEBA si da tiempo; se dira cual).

## Como se elige y que decide

- En AJUSTE, de cada idea, la candidata de mayor acierto TOTAL de zona (todos los clips con etiqueta
  y `zona_` de la mitad; los que no contestan cuentan como fallo en las dos reglas por igual, porque
  no contestan en ninguna) frente a produccion. A igualdad, la que menos clips cambia. Solo va a
  PRUEBA si en AJUSTE su ganancia tiene IC95 por clip por encima de cero; si no, PRUEBA no se mira.
- **Criterio en PRUEBA**, una vez por idea: el acierto TOTAL sube frente a produccion con IC por
  clip (bootstrap pareado, 2.000) que no cruza cero, y **no baja en ningun arma mas de 0,02**. Como
  son DOS ideas evaluadas en PRUEBA, cada una tiene que pasar con IC al **97,5 %** (Bonferroni).
- Informativo: por via de la luz, por arma, IC por torneo, cuantos clips cambian de zona y la
  discrepancia en metros.

Lo que no pase queda APAGADO tras su variable de entorno, con su numero.

**Cambio 1 (2026-10-09, con la idea 1 ya medida y ANTES de ver ninguna cifra de la idea 2):** si
pasan las dos, se adopta solo la de mayor ganancia en PRUEBA; la combinacion de las dos no esta
medida y queda para otro protocolo. La idea 1 en produccion usa el angulo calculado EXACTAMENTE como
en la medida (el codigo de `rasgos.py`/`modelo.py` del agente H, movido a un modulo, y comprobado
contra `rasgos.jsonl`), no una version nueva.

## Resultados de la idea 1 (fusion)

Script `fusion.py` (copiado a `02_validacion/auditoria/angulo_fusion_2026-10-09/`). Acierto TOTAL =
sobre todos los clips con etiqueta y `zona_` de la mitad; los que no contestan cuentan como fallo en
las dos reglas. Las 21 candidatas en AJUSTE (3.395 clips; contestan 2.045; con x del angulo 1.849),
en `fusion_ajuste.txt`. Las mejores:

| candidata | cambian | produccion | nueva | dif | IC95 por clip |
|---|---:|---:|---:|---:|---|
| W(0) e<=inf (solo angulo) | 649 | 0,2863 | 0,2999 | +0,0135 | [+0,0015, +0,0256] |
| D(1) e<=inf | 552 | 0,2863 | 0,3063 | +0,0200 | [+0,0088, +0,0306] |
| **D(2) e<=inf** | 442 | 0,2863 | **0,3066** | **+0,0203** | [+0,0106, +0,0303] |
| D(3) e<=inf | 313 | 0,2863 | 0,3043 | +0,0180 | [+0,0100, +0,0262] |
| W(0,5) e<=inf | 387 | 0,2863 | 0,2892 | +0,0029 | [-0,0059, +0,0118] |

Las medias ponderadas (W) apenas mejoran; el DESEMPATE (D) si: cuando homografia y angulo discrepan
mucho, casi siempre se equivoca la homografia (identifico mal las rayas). La puerta de calidad del
torneo (e <= 1,0 / 1,5) quita ganancia. **Elegida por la regla declarada: D(2) e<=inf**; su IC en
AJUSTE esta por encima de cero, asi que va a PRUEBA. Informativo en AJUSTE: espada +0,026, florete
+0,022, **sable -0,019** (n=207); IC95 por torneo [-0,002, +0,035].

**PRUEBA, una vez** (2.130 clips; contestan 1.353; con x del angulo 1.207; cambian de zona 263):

    produccion 0,3089   D(2) 0,3338   +0,0249   IC97,5 por clip [+0,0113, +0,0380]
    IC97,5 por torneo [+0,0062, +0,0476]   discrepancia |x_h - x_a| mediana 0,66 m
    espada n=503 +0,006 | florete n=1.282 +0,019 | sable n=345 +0,075     (ninguna baja)
    via v2_fie n=1.324 +0,036 | v2_generic n=699 +0,004

**LA IDEA 1 PASA** (IC al 97,5 % por encima de cero y ningun arma baja). En los clips que contestan,
el acierto pasa de ~0,486 a ~0,525.

### En produccion

- `01_extraccion/codigo/angulo_lejano.py` (nuevo): `rasgos()` y `resume()` son `rasgos.py:uno` y
  `modelo.py:carga` sin cambios; `x_angulo()` (modelo del torneo sin el clip, >= 8 referencias) y
  `decide()` (D(2)).
- `01_extraccion/codigo/zona_tocado.py`: con `FENCINGLY_ZONA_FUSION=1` (defecto) la etapa 11 escribe
  ademas `ang_lejano`, `desp_pies_m`, `dos_pies` (~0,4 s mas por clip) y deja `x_final_m`/`zona_final`
  = la homografia con `via_zona = pendiente_torneo`.
- `01_extraccion/zona_fusion.py` (nuevo): el paso POR TORNEO despues de la etapa 11; escribe
  `x_angulo_m`, `x_final_m`, `zona_final`, `via_zona` (`homografia`, `angulo_torneo`,
  `homografia_sin_modelo`). `zona` y `x_tocado_m` no se tocan. `--cache rasgos.jsonl.gz` toma los
  rasgos ya medidos en vez de decodificar.
- `02_validacion/herramienta/mide_zona.py`: mide tambien `zona_final` si existe.

Comprobado: (1) con los rasgos de `rasgos.jsonl`, la logica de produccion da las MISMAS zonas que
`fusion.py` D(2) en los 2.045 + 1.353 clips que contestan (0 distintas, x del angulo igual a 1e-13);
(2) `angulo_lejano.rasgos` decodificando de nuevo reproduce `rasgos.jsonl` en 40 de 40 clips al azar;
(3) la etapa 11 con las columnas nuevas, en un directorio aparte, da los mismos rasgos que la
medida; (4) `zona_fusion.py` sobre una COPIA de los 5.837 `zona_` de anotados (con la cache) y
`mide_zona.py` sobre esa copia: zona final **0,516** contra la homografia 0,480, +0,036 IC95 por clip
[+0,023, +0,048] (n = 3.398 que contestan; en produccion las referencias incluyen tambien los clips sin
etiqueta, por eso no es exactamente la cifra de PRUEBA).

## Resultados de la idea 2 (prior del angulo en `hypothesize`)

`prior.py` rehace la etapa 11 con los rasgos decodificados una vez y `calibrate` tres veces: el
prior de siempre (control `base`) y P(1,5), P(3,0). El control reproduce el `zona_` de disco en
**todos** los clips de la muestra (0 distintos de 548 calculados).

**AJUSTE**, muestra al azar de 600 (semilla 20261009; 52 sin modelo de torneo, que no cambian):

| candidata | cambian | nuevos que contestan | produccion | nueva | dif | IC95 por clip | IC95 por torneo |
|---|---:|---:|---:|---:|---:|---|---|
| **P(1,5)** | 70 | 12 | 0,2633 | 0,2933 | **+0,0300** | [+0,0083, +0,0517] | [+0,0057, +0,0532] |
| P(3,0) | 33 | 4 | 0,2633 | 0,2733 | +0,0100 | [-0,0033, +0,0233] | [-0,0022, +0,0246] |

Por arma (P(1,5)): espada +0,051 (n=98), florete +0,026 (n=466), sable +0,028 (n=36). Casi toda la
ganancia es CORREGIR zonas que ya se contestaban (la identificacion de rayas), no cobertura nueva.
**Elegida P(1,5)**; su IC esta por encima de cero: va a PRUEBA, muestra al azar de 1.000.

**PRUEBA, una vez**, muestra al azar de 1.000 (semilla 20261009; 107 sin modelo de torneo, que no
cambian; control `base` = disco en todos):

    produccion 0,3060   P(1,5) 0,3230   +0,0170   IC97,5 por clip [+0,0020, +0,0320]
    IC97,5 por torneo [+0,0038, +0,0375]   cambian 81 (nuevos que contestan 10)
    espada n=235 +0,026 | florete n=594 +0,012 | sable n=171 +0,023          (ninguna baja)

**La idea 2 tambien pasa**, por poco.

## Veredicto

Pasan las dos. Por el cambio 1 (declarado antes de ver la idea 2) se adopta SOLO la de mayor
ganancia en PRUEBA. En los MISMOS 1.000 clips de PRUEBA (`compara_ideas.py`):

    idea 1, D(2)     +0,0270  IC97,5 [+0,0090, +0,0460]
    idea 2, P(1,5)   +0,0170  IC97,5 [+0,0020, +0,0320]
    idea 1 - idea 2  +0,0100  IC95 [-0,0060, +0,0260]

**ADOPTADA LA IDEA 1 (fusion D(2))**, `FENCINGLY_ZONA_FUSION=1` por defecto. La idea 2 NO entra en
produccion: queda medida (+0,017 en PRUEBA) y su codigo en `02_validacion/auditoria/
angulo_fusion_2026-10-09/prior.py`. Ademas seria cara de montar: su prior necesita el modelo del
torneo DENTRO de la etapa 11, o sea dos pasadas de la etapa 11 (~12 s/clip cada una). Combinar las
dos no esta medido (otro protocolo).

## Que hay que hacer para aplicarlo

- **Anotados** (no se ha escrito nada en `01_extraccion/pruebas`): con la noche parada,
  `python 01_extraccion/zona_fusion.py --conjunto anotados --cache 02_validacion/auditoria/rayas_id_2026-10-09/rasgos.jsonl.gz --escribe`
  (la cache cubre 5.369 clips; decodifica los ~270 restantes, ~0,4 s cada uno). Reescribe los 5.837
  `zona_`, asi que `respalda.py` volvera a subir esos clips. Despues `mide_zona.py` da la zona final.
- **Florete**: la etapa 11 ya guarda los rasgos del angulo (+~0,4 s de CPU por clip); hay que correr
  `zona_fusion.py --conjunto foil --escribe` al final (y, si se quiere, de vez en cuando durante la
  corrida: el modelo de cada torneo mejora con mas clips). El paso por torneo cuesta segundos por
  torneo (un ajuste Huber por clip).
