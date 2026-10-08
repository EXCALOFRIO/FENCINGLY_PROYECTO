"""Barras horizontales dibujadas a mano con cv2, para las figuras publicas de PRIORIDAD.

POR QUE NO MATPLOTLIB. Se intento y el resultado tuvo cinco colisiones de rotulo: los titulos se
comian los subtitulos, las notas al pie caian sobre el eje y la caja de aviso tapaba una barra.
Colocar texto en coordenadas de datos es adivinar donde acaba una cadena. Aqui cada banda de texto
tiene su alto calculado y se APILA con `vstack`, que es la misma regla que el resto de las figuras:
una banda nunca se pinta encima de un panel ya compuesto.

El sistema de color y las reglas de marca son los de la casa: superficie oscura, rejilla de un tono
sobre la superficie, marcas finas, extremos redondeados, un solo eje, valor rotulado en el extremo
de cada barra y color de estado SOLO cuando el color significa bueno o malo.
"""
from __future__ import annotations

import cv2
import numpy as np

from comun import F, FD, SinPisarse, banda, marco, texto


def bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


SUP = bgr("#1a1a19")
TINTA = bgr("#f2f2ef")
TINTA2 = bgr("#b9b9b2")
TINTA3 = bgr("#83837c")
REJILLA = bgr("#2e2e2c")
AZUL = bgr("#3987e5")
NARANJA = bgr("#d95926")
AQUA = bgr("#199e70")
CRITICO = bgr("#d03b3b")
AVISO = bgr("#fab219")
BUENO = bgr("#0ca30c")


class Coloca:
    """Rotulos que no se pisan, con el ANCHO REAL de cada cadena.

    POR QUE NO ES `panel_publico.SinPisarse` AQUI. Esa clase es la correcta sobre un fotograma:
    ahi los rotulos son de largo parecido ("LEFT (side 0)", "RIGHT (side 1)") y un ancho fijo de
    230 px los separa bien. En una grafica los rotulos van de "3.41" (32 px) a
    "HarshExemplaryGermanspaniel l1 1.00" (330 px), y un ancho fijo hace las dos cosas mal a la
    vez: separa de mas los cortos, que caben, y de menos los largos, que no. Asi que se guarda el
    rectangulo de cada uno y se compara de verdad.
    """

    def __init__(self, alto, paso=17, margen=4):
        self.r, self.alto, self.paso, self.margen = [], alto, paso, margen

    def reserva(self, x0, y0, x1, y1):
        self.r.append((x0, y0, x1, y1))

    def sitio(self, x, y, an, al=15):
        for signo in (-1, 1):
            yy = y
            for _ in range(16):
                caja = (x - self.margen, yy - al - self.margen,
                        x + an + self.margen, yy + self.margen)
                if 12 <= yy <= self.alto - 4 and not any(
                        caja[0] < b[2] and b[0] < caja[2] and caja[1] < b[3] and b[1] < caja[3]
                        for b in self.r):
                    self.r.append(caja)
                    return yy
                yy += signo * self.paso
        self.r.append((x, y - al, x + an, y))
        return y


def _cap(img, x0, y0, x1, y1, col, r=4):
    """Barra con el extremo de dato redondeado y el otro anclado a la linea base, cuadrado."""
    if x1 - x0 < 2 * r:
        cv2.rectangle(img, (x0, y0), (max(x0 + 1, x1), y1), col, -1)
        return
    cv2.rectangle(img, (x0, y0), (x1 - r, y1), col, -1)
    cv2.rectangle(img, (x1 - r, y0 + r), (x1, y1 - r), col, -1)
    for cy in (y0 + r, y1 - r):
        cv2.circle(img, (x1 - r, cy), r, col, -1, cv2.LINE_AA)


def tile(ancho, titulo, sub, n, filas, pies=(), xmax=100.0, etq=350, mar_der=190,
         alto_fila=62, grosor=26, ref=None, leyenda=()):
    """Un panel de barras completo: titulo, n, rejilla, barras, leyenda y notas al pie.

    `filas` son dicts con `et` (lista de 1-2 lineas), `v`, `col`, y opcionalmente `destaca`.
    `ref` es (valor, color, lineas de texto) para la linea de referencia vertical.
    """
    cab = banda(([(titulo, TINTA, 0.66)]
                 + ([(n, AVISO, 0.46)] if n else [])
                 + ([(sub, TINTA3, 0.46)] if sub else [])),
                ancho, alto_linea=25, margen=12, fondo=SUP)

    huecos = [int(f.get("hueco", 0)) for f in filas]
    techo = 0 if ref is None else 22 * len(ref[2]) + 6
    """EL ROTULO DE LA LINEA DE REFERENCIA TIENE SU PROPIA FRANJA RESERVADA ARRIBA, y el alto del
    panel crece para dársela. Se probaron las dos alternativas y las dos fallan: al lado de la
    primera barra pisa su valor, que es justo el que la linea esta juzgando; al lado de la barra mas
    corta pisa el valor de esa, porque en un panel estrecho el rotulo es mas ancho que el hueco que
    queda. Una franja propia no puede chocar con nada por construccion."""
    alto = alto_fila * len(filas) + sum(huecos) + 34 + techo
    g = np.full((alto, ancho, 3), SUP, dtype="uint8")
    x0, x1 = etq, ancho - mar_der

    def px(v):
        return int(round(x0 + (x1 - x0) * min(max(v, 0.0), xmax) / xmax))

    """La rejilla se adapta a la escala. Con `xmax = 1` (las tasas de cambio por segundo) la lista
    fija de 0-100 dejaba el panel con UN SOLO numero en el eje, el cero, y sin nada con que medir
    las barras a ojo."""
    ticks = ([0, 20, 40, 60, 80, 100] if xmax > 5
             else [round(0.2 * j * xmax, 3) for j in range(6)])
    for t in ticks:
        if t > xmax:
            continue
        xt = px(t)
        cv2.line(g, (xt, techo + 4), (xt, alto - 26), REJILLA, 1)
        et = "%d" % t if xmax > 5 else ("%.1f" % t)
        (tw, _th), _b = cv2.getTextSize(et, F, 0.42, 1)
        texto(g, et, (xt - tw // 2, alto - 8), 0.42, TINTA3, 1, sombra=False)
    cv2.line(g, (x0, alto - 26), (x1, alto - 26), REJILLA, 1)

    if ref is not None:
        rv, rc, _rt = ref
        cv2.line(g, (px(rv), techo + 2), (px(rv), alto - 26), rc, 2)

    cur = 10 + techo
    for i, f in enumerate(filas):
        cur += huecos[i]
        cy = cur + alto_fila // 2
        cur += alto_fila
        y0, y1 = cy - grosor // 2, cy + grosor // 2
        _cap(g, x0, y0, px(f["v"]), y1, f["col"])
        ls = f["et"] if isinstance(f["et"], (list, tuple)) else [f["et"]]
        yy = cy - 8 * (len(ls) - 1)
        for s in ls:
            if s:
                (tw, _th), _b = cv2.getTextSize(s, F, 0.46, 1)
                texto(g, s, (x0 - 14 - tw, yy + 5), 0.46, TINTA2, 1, sombra=False)
            yy += 19
        v = f.get("txt") or ("%.1f %%" % f["v"] if abs(f["v"] * 10 % 10) > 1e-9
                             else "%.0f %%" % f["v"])
        texto(g, v, (px(f["v"]) + 12, cy + 6), 0.62, TINTA,
              2 if f.get("destaca") else 1, FD, sombra=False)

    if ref is not None:
        rv, rc, rt = ref
        """EL ROTULO DE LA LINEA DE REFERENCIA VA EN LA BANDA DE LA BARRA MAS CORTA, calculada, no
        arriba ni abajo por convenio. Arriba comparte pixeles con el valor de la primera barra, que
        es justo la que la linea esta juzgando; abajo choca si la ultima barra es la mas larga. La
        fila mas corta es la unica en la que el hueco a la derecha del umbral esta libre seguro."""
        anchos = [cv2.getTextSize(s, FD, 0.5, 1)[0][0] for s in rt]
        ex = max(8, min(px(rv) + 14, ancho - 8 - max(anchos)))
        cv2.line(g, (px(rv), 4), (px(rv), techo), rc, 2)
        yy = 20
        for s in rt:
            texto(g, s, (ex, yy), 0.5, rc, 1, FD, sombra=False)
            yy += 22

    trozos = [cab, g]
    if leyenda:
        lg = np.full((30, ancho, 3), SUP, dtype="uint8")
        cx = etq
        for et, col in leyenda:
            cv2.rectangle(lg, (cx, 10), (cx + 22, 21), col, -1)
            texto(lg, et, (cx + 30, 21), 0.46, TINTA2, 1, sombra=False)
            (tw, _th), _b = cv2.getTextSize(et, F, 0.46, 1)
            cx += 30 + tw + 34
        trozos.append(lg)
    if pies:
        trozos.append(banda([(s, c, 0.455) for s, c in pies], ancho, alto_linea=22, margen=10,
                            fondo=SUP))
    out = np.vstack(trozos)
    marco(out, REJILLA, 1)
    return out


def strip(ancho, titulo, sub, n, xlo, xhi, marcas, umbral=None, pies=(), etq=360, mar_der=120,
          alto=236, unidad=""):
    """Una sola recta numerica con los valores puestos encima. La forma correcta cuando el dato
    son unos pocos numeros y un umbral, y lo que hay que ver es SI SE SOLAPAN.

    `marcas` son dicts con `v`, `col`, `et`, `fila` (0 arriba, 1 abajo) y `r` (radio).
    """
    cab = banda(([(titulo, TINTA, 0.66)]
                 + ([(n, AVISO, 0.46)] if n else [])
                 + ([(sub, TINTA3, 0.46)] if sub else [])),
                ancho, alto_linea=25, margen=12, fondo=SUP)
    g = np.full((alto, ancho, 3), SUP, dtype="uint8")
    x0, x1 = etq, ancho - mar_der
    ejey = alto - 46

    def px(v):
        return int(round(x0 + (x1 - x0) * (min(max(v, xlo), xhi) - xlo) / float(xhi - xlo)))

    paso = 0.5 if (xhi - xlo) <= 5 else 1.0
    t = xlo
    while t <= xhi + 1e-9:
        cv2.line(g, (px(t), 18), (px(t), ejey), REJILLA, 1)
        texto(g, ("%g" % round(t, 2)), (px(t) - 9, ejey + 20), 0.42, TINTA3, 1, sombra=False)
        t += paso
    cv2.line(g, (x0, ejey), (x1, ejey), REJILLA, 1)
    if unidad:
        texto(g, unidad, (x1 + 12, ejey + 6), 0.44, TINTA3, 1, sombra=False)

    co = Coloca(alto)
    if umbral is not None:
        uv, uc, ut = umbral
        cv2.line(g, (px(uv), 14), (px(uv), ejey), uc, 2)
        yy = 30
        for s in ut:
            (tw, _th), _b = cv2.getTextSize(s, F, 0.47, 1)
            texto(g, s, (px(uv) + 12, yy), 0.47, uc, 1, FD, sombra=False)
            co.reserva(px(uv) + 6, yy - 16, px(uv) + 18 + tw, yy + 4)
            yy += 21

    for m in marcas:
        cy = ejey - 40 - 44 * int(m.get("fila", 0))
        r = int(m.get("r", 8))
        cv2.circle(g, (px(m["v"]), cy), r + 2, SUP, -1, cv2.LINE_AA)
        cv2.circle(g, (px(m["v"]), cy), r, m["col"], -1, cv2.LINE_AA)
        co.reserva(px(m["v"]) - r, cy - r, px(m["v"]) + r, cy + r)
    for m in marcas:
        if not m.get("et"):
            continue
        r = int(m.get("r", 8))
        cy = ejey - 40 - 44 * int(m.get("fila", 0))
        (tw, _th), _b = cv2.getTextSize(m["et"], F, 0.44, 1)
        ex = max(4, min(px(m["v"]) - tw // 2, ancho - tw - 6))
        ey = co.sitio(ex, cy - r - 8, tw)
        texto(g, m["et"], (ex, ey), 0.44, m["col"], 1, sombra=True)

    trozos = [cab, g]
    if pies:
        trozos.append(banda([(s, c, 0.455) for s, c in pies], ancho, alto_linea=22, margen=10,
                            fondo=SUP))
    out = np.vstack(trozos)
    marco(out, REJILLA, 1)
    return out


def nube(ancho, titulo, sub, n, xlo, xhi, valores, umbral=None, pies=(), etq=60, mar_der=120,
         alto=210, destaca=()):
    """Nube de puntos de UNA variable: cada clip x lado es un punto, con jitter vertical fijo.

    Es la forma que contesta «viaja este umbral entre tiradores»: se ve la dispersion entera y
    cuantos puntos caen del lado malo de la raya, que es el numero que importa.
    """
    cab = banda(([(titulo, TINTA, 0.66)]
                 + ([(n, AVISO, 0.46)] if n else [])
                 + ([(sub, TINTA3, 0.46)] if sub else [])),
                ancho, alto_linea=25, margen=12, fondo=SUP)
    g = np.full((alto, ancho, 3), SUP, dtype="uint8")
    x0, x1 = etq, ancho - mar_der
    ejey = alto - 34

    def px(v):
        return int(round(x0 + (x1 - x0) * (min(max(v, xlo), xhi) - xlo) / float(xhi - xlo)))

    t = xlo
    while t <= xhi + 1e-9:
        cv2.line(g, (px(t), 10), (px(t), ejey), REJILLA, 1)
        texto(g, "%.1f" % t, (px(t) - 12, ejey + 20), 0.42, TINTA3, 1, sombra=False)
        t += 0.25
    cv2.line(g, (x0, ejey), (x1, ejey), REJILLA, 1)
    co = Coloca(ejey - 4)
    if umbral is not None:
        uv, uc, ut = umbral
        cv2.line(g, (px(uv), 8), (px(uv), ejey), uc, 2)
        yy = 24
        for s in ut:
            (tw, _th), _b = cv2.getTextSize(s, F, 0.47, 1)
            texto(g, s, (px(uv) + 12, yy), 0.47, uc, 1, FD, sombra=False)
            co.reserva(px(uv) + 6, yy - 16, px(uv) + 18 + tw, yy + 4)
            yy += 21
    rng = np.random.default_rng(7)
    puntos = []
    for i, (et, v, col) in enumerate(valores):
        cy = int(ejey - 22 - (i % 7) * 15 - rng.integers(0, 4))
        cv2.circle(g, (px(v), cy), 7, SUP, -1, cv2.LINE_AA)
        cv2.circle(g, (px(v), cy), 5, col, -1, cv2.LINE_AA)
        co.reserva(px(v) - 7, cy - 7, px(v) + 7, cy + 7)
        puntos.append((et, v, col, px(v), cy))
    for et, v, col, x, cy in puntos:
        if et not in destaca:
            continue
        s = "%s  %.3f" % (et, v)
        (tw, _th), _b = cv2.getTextSize(s, F, 0.44, 1)
        """Si el rotulo no cabe a la derecha del punto se pone a la IZQUIERDA. Sin esto el ultimo
        punto de la nube se sale del panel y su cifra queda cortada por el borde."""
        ex = x + 12 if x + 12 + tw <= ancho - 8 else x - 12 - tw
        ey = co.sitio(ex, cy + 5, tw)
        cv2.line(g, (x, cy), (ex + (0 if ex > x else tw), ey - 5), col, 1, cv2.LINE_AA)
        texto(g, s, (ex, ey), 0.44, col, 1)
    trozos = [cab, g]
    if pies:
        trozos.append(banda([(s, c, 0.455) for s, c in pies], ancho, alto_linea=22, margen=10,
                            fondo=SUP))
    out = np.vstack(trozos)
    marco(out, REJILLA, 1)
    return out


def nota(ancho, titulo, lineas, borde, fondo, hero=None, alto_min=0):
    """Una caja de texto con borde de color. `hero` es (numero, color) para el numero grande."""
    trozos = [banda([(titulo, TINTA, 0.66)], ancho, alto_linea=26, margen=12, fondo=fondo)]
    if hero:
        h = np.full((104, ancho, 3), fondo, dtype="uint8")
        texto(h, hero[0], (26, 78), 2.5, hero[1], 5, FD, sombra=False)
        (tw, _th), _b = cv2.getTextSize(hero[0], FD, 2.5, 5)
        yy = 44
        for s, c in hero[2]:
            texto(h, s, (26 + tw + 30, yy), 0.54, c, 1, FD, sombra=False)
            yy += 26
        trozos.append(h)
    trozos.append(banda([(s, c, 0.455) for s, c in lineas], ancho, alto_linea=22, margen=12,
                        fondo=fondo))
    out = np.vstack(trozos)
    if out.shape[0] < alto_min:
        out = np.vstack([out, np.full((alto_min - out.shape[0], ancho, 3), fondo, dtype="uint8")])
    marco(out, borde, 2)
    return out
