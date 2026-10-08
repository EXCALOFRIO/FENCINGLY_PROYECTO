"""Barra de progreso con el API de tqdm y respaldo propio cuando tqdm no esta.

Por que existe: `tqdm` NO esta instalado en `scratchpad/fz/.venv` ni en `.venv-gpu`
(comprobado el 2026-09-09), y `pipeline.process_many` hace

    try:    from tqdm.auto import tqdm ; bar = tqdm(...)
    except: bar = None

o sea que **en local la barra de la extraccion no existe y no avisa**. En Colab tqdm si
esta preinstalado, asi que ahi se usa el de verdad. Este modulo elige uno u otro y
mantiene el mismo API, para que el mismo codigo de extraccion sirva en los dos sitios y
para que la ausencia de tqdm no vuelva a ser un fallo silencioso.

API soportado: `Barra(total=, desc=, unit=, leave=, dynamic_ncols=)`, `.update(n)`,
`.set_postfix(**kw)`, `.set_description(txt)`, `.write(txt)`, `.close()`, y `with`.
"""
from __future__ import annotations

import shutil
import sys
import time

try:                                          # Colab: el de verdad
    from tqdm.auto import tqdm as _tqdm
    TQDM = True
except Exception:                             # local: el respaldo de abajo
    _tqdm = None
    TQDM = False


def _hms(s: float) -> str:
    if s is None or s != s or s in (float("inf"), float("-inf")):
        return "--:--"
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class _Propia:
    """Respaldo minimo: misma linea que tqdm, con ritmo y tiempo restante."""

    def __init__(self, total=None, desc="", unit="it", leave=True, dynamic_ncols=True,
                 stream=None, min_intervalo=0.2, **_):
        self.total = None if total is None else int(total)
        self.desc, self.unit, self.leave = desc, unit, leave
        self.dynamic = dynamic_ncols
        self.n = 0
        self.postfix = ""
        self.t0 = time.perf_counter()
        self._ultimo = 0.0
        self._stream = stream or sys.stderr
        self._min = min_intervalo
        self._cerrada = False
        self._pinta(forzar=True)

    # -- API tqdm --
    def update(self, n=1):
        self.n += int(n)
        self._pinta()

    def set_postfix(self, **kw):
        self.postfix = " ".join(f"{k}={v}" for k, v in kw.items())
        self._pinta()

    def set_postfix_str(self, s=""):
        self.postfix = str(s)
        self._pinta()

    def set_description(self, desc=""):
        self.desc = str(desc)
        self._pinta()

    def write(self, txt=""):
        self._stream.write("\r" + " " * self._ancho() + "\r")
        self._stream.write(str(txt) + "\n")
        self._stream.flush()
        self._pinta(forzar=True)

    def close(self):
        if self._cerrada:
            return
        self._cerrada = True
        self._pinta(forzar=True)
        self._stream.write("\n" if self.leave else "\r" + " " * self._ancho() + "\r")
        self._stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __iter__(self):
        return self

    # -- pintado --
    def _ancho(self):
        if not self.dynamic:
            return 80
        try:
            return max(40, shutil.get_terminal_size((100, 24)).columns - 1)
        except Exception:
            return 100

    def _pinta(self, forzar=False):
        ahora = time.perf_counter()
        if not forzar and ahora - self._ultimo < self._min:
            return
        self._ultimo = ahora
        dt = ahora - self.t0
        ritmo = self.n / dt if dt > 0 else 0.0
        izq = f"{self.desc}: " if self.desc else ""
        if self.total:
            frac = min(1.0, self.n / self.total)
            resto = (self.total - self.n) / ritmo if ritmo > 0 else float("nan")
            der = (f" {self.n}/{self.total} [{_hms(dt)}<{_hms(resto)}, "
                   f"{ritmo:.2f} {self.unit}/s]")
            if self.postfix:
                der += " " + self.postfix
            hueco = self._ancho() - len(izq) - len(der) - 8
            nb = max(6, hueco)
            lleno = int(nb * frac)
            barra = "#" * lleno + "-" * (nb - lleno)
            linea = f"{izq}{frac*100:3.0f}%|{barra}|{der}"
        else:
            linea = f"{izq}{self.n} {self.unit} [{_hms(dt)}, {ritmo:.2f} {self.unit}/s]"
            if self.postfix:
                linea += " " + self.postfix
        linea = linea[: self._ancho()]
        self._stream.write("\r" + linea + " " * max(0, self._ancho() - len(linea)))
        self._stream.flush()


def Barra(*a, **kw):
    """tqdm si esta; si no, el respaldo. `dynamic_ncols=True` por omision."""
    kw.setdefault("dynamic_ncols", True)
    if TQDM:
        kw.pop("stream", None)
        kw.pop("min_intervalo", None)
        return _tqdm(*a, **kw)
    return _Propia(*a, **kw)


def motor() -> str:
    return "tqdm" if TQDM else "respaldo_local"
