"""El guardia de las rutas: que la RUTA CRITICA no tenga ninguna ruta absoluta de esta maquina.

## QUE PROBLEMA RESUELVE, Y CUAL NO

En el arbol hay del orden de **225 rutas absolutas**. Convertirlas todas seria tocar unos 80
scripts de experimento (`a2_*`, `e1`-`e29`, `f1`-`f17`, `capas_limpias_*`) **cuyas salidas son las
cifras publicadas** en `RESULTADOS.md`. Tocarlos arriesga esas cifras a cambio de cero beneficio en
Colab, porque el pipeline no los importa.

Asi que la decision es: se convierten los **nueve de la ruta critica** y se pone este guardia para
que sigan limpios. Lo demas se deja, a proposito y por escrito.

La forma en que esto falla en Colab merece decirse: una ruta absoluta que no existe **no da un
error**. `os.walk` de un directorio inexistente devuelve vacio, asi que el lote imprime
"NO VIDEO, skipped" quinientas veces y se lee como un problema de corpus. Y `snap()` devolvia
`None`, con lo que `from_pretrained(str(None))` pedia al hub un repositorio llamado literalmente
`"None"` y la traza acusaba a Hugging Face. Por eso hace falta un guardia y no basta con mirar.

    python verifica_rutas.py          sale 1 si la ruta critica esta sucia
    python verifica_rutas.py --todo   ademas cuenta las del resto del arbol, sin exigir nada
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]

CRITICOS = [
    "01_extraccion/pipeline.py",
    "01_extraccion/lotes.py",
    "01_extraccion/noche.py",
    "01_extraccion/bucle.py",
    "01_extraccion/codigo/catalogo.py",
    "01_extraccion/codigo/etiquetador.py",
    "01_extraccion/codigo/fie.py",
    "01_extraccion/codigo/mano_corpus.py",
    "01_extraccion/codigo/pista.py",
    "01_extraccion/codigo/posicion.py",
    "01_extraccion/codigo/cono_lib.py",
    "01_extraccion/codigo/luces.py",
    "01_extraccion/codigo/extrae_pareja.py",
    "02_validacion/herramienta/video_revision.py",
    "02_validacion/herramienta/grafica_pista.py",
]
"""LA RUTA CRITICA: lo que el pipeline importa de verdad para extraer. `rutas.py` NO esta en la
lista, y es a proposito: es el unico fichero donde una ruta absoluta es correcta, porque su
trabajo es precisamente concentrarlas."""

PATRON = re.compile(r"""["'][A-Za-z]:[\\/]|["']/(?:home|Users|mnt|content)/""")

PERMITIDO = re.compile(r"D:/FENCINGLY_PY|FENCINGLY_VENV|_VENV_TMP|# ruta-ok")
"""Excepciones declaradas. `rutas.interprete` cita el venv duradero en su propio docstring, y una
linea marcada `# ruta-ok` es una que alguien miro y decidio dejar."""


def escanea(ficheros):
    malas = []
    for rel in ficheros:
        p = RAIZ / rel
        if not p.is_file():
            malas.append((rel, 0, "NO EXISTE"))
            continue
        for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if PATRON.search(ln) and not PERMITIDO.search(ln):
                malas.append((rel, i, ln.strip()[:96]))
    return malas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true")
    a = ap.parse_args()
    print("GUARDIA DE RUTAS")
    print("=" * 96)
    malas = escanea(CRITICOS)
    print("ruta critica: %d ficheros, %d rutas absolutas" % (len(CRITICOS), len(malas)))
    for rel, i, ln in malas:
        print("   %-46s :%-5d %s" % (rel, i, ln))
    if a.todo:
        otros = sorted(
            str(p.relative_to(RAIZ)).replace("\\", "/")
            for p in RAIZ.rglob("*.py")
            if "99_historico" not in p.parts and "05_aparcado" not in p.parts
            and str(p.relative_to(RAIZ)).replace("\\", "/") not in CRITICOS)
        resto = escanea(otros)
        print()
        print("resto del arbol (NO se exige): %d ficheros, %d rutas absolutas en %d de ellos"
              % (len(otros), len(resto), len({r for r, _i, _l in resto})))
        print("   Se dejan a proposito: son los scripts de experimento que producen las cifras")
        print("   publicadas, y el pipeline no los importa.")
    print()
    if malas:
        print("SUCIA. En Colab estas rutas no existen, y lo peor es que NO dan error:")
        print("un directorio que no esta devuelve vacio y el lote dice 'NO VIDEO' 500 veces.")
        return 1
    print("LIMPIA: la ruta critica no depende de esta maquina.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
