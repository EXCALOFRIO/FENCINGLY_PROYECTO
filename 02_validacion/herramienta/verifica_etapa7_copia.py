"""Recalcula la etapa 7 de una muestra EN UNA COPIA y la compara con lo que hay en disco.

Para cada clip corre `grafica_pista.procesa` dos veces sobre el MISMO resultado de `pista.analiza`
(se calcula una vez y se reutiliza: la homografia es el 90 % del coste y no depende de lo que se
mide aqui), con `grafica_pista.SALIDA` apuntando a `--salida`, nunca a `02_validacion/pista`:

  base       FENCINGLY_PUERTA_EVENTO=0, FENCINGLY_MANO_ACCION=0   (tiene que dar lo de disco)
  puerta     solo la puerta por evento
  nuevo      las dos a 1

y compara, fila a fila, `serie_eventos_<clip>.csv`:
  base   contra disco  identico en todas las columnas -> el codigo y los datos son los mismos;
  puerta contra base   (c) columnas del etiquetador identicas, y todas las de antes salvo
                       `publicable`; `publicable_clip` == `publicable` de la base; la puerta (a);
  nuevo  contra base   (f) filas que no son `brazo` identicas en todas las columnas de antes.

    python verifica_etapa7_copia.py --muestra 200 [--extra lista.txt] [--procesos 6]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
import traceback

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "01_extraccion"))
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
sys.path.insert(0, str(RAIZ / "02_validacion" / "herramienta"))

COL_ETIQUETADOR = ("lado", "tipo", "cuadro_inicio", "x_inicio_m")
NUEVAS = ("publicable_clip", "fisica_evento", "motivo_no_publicable", "mano_por", "mano_dudosa")


def _lee(p):
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def uno(args):
    clip, salida = args
    import grafica_pista as GP
    import pista as PI
    import posicion as PO
    import rutas as RU
    base_dir = pathlib.Path(salida) / "base"
    puerta_dir = pathlib.Path(salida) / "puerta"
    nuevo_dir = pathlib.Path(salida) / "nuevo"
    for d in (base_dir, puerta_dir, nuevo_dir):
        d.mkdir(parents=True, exist_ok=True)
    orig = PI.analiza
    cache = {}

    def analiza_una_vez(c, *a, **k):
        if c not in cache:
            cache[c] = orig(c, *a, **k)
        return cache[c]
    PI.analiza = analiza_una_vez
    out = {"clip": clip}
    try:
        for nombre, d, pe, ma in (("base", base_dir, False, False), ("puerta", puerta_dir, True, False),
                                  ("nuevo", nuevo_dir, True, True)):
            GP.SALIDA = d
            PO.PUERTA_EVENTO, PO.MANO_ACCION = pe, ma
            f = GP.procesa(clip, con_figura=False)
            out[nombre] = "ok" if f is not None else "nada"
    except Exception as ex:
        out["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:160])
        out["traza"] = traceback.format_exc()[-600:]
    finally:
        PI.analiza = orig
    disco = _lee(RU.PISTA / ("serie_eventos_%s.csv" % clip))
    b = _lee(base_dir / ("serie_eventos_%s.csv" % clip))
    n = _lee(puerta_dir / ("serie_eventos_%s.csv" % clip))
    nv = _lee(nuevo_dir / ("serie_eventos_%s.csv" % clip))
    out["hay_disco"] = disco is not None
    if b is not None and disco is not None:
        out["base_igual_disco"] = (len(b) == len(disco)
                                   and all(x == y for x, y in zip(b, disco)))
    if b is not None and n is not None:
        cols_b = list(b[0].keys()) if b else []
        out["n_filas"] = len(b)
        out["mismas_filas"] = len(b) == len(n)
        out["etiquetador_igual"] = (len(b) == len(n) and all(
            all(x[k] == y[k] for k in COL_ETIQUETADOR) for x, y in zip(b, n)))
        cols_cmp = [k for k in cols_b if k != "publicable"]
        out["puerta_resto_igual"] = (len(b) == len(n) and all(
            all(x[k] == y[k] for k in cols_cmp) for x, y in zip(b, n)))
        nb = [x for x in b if x["tipo"] != "brazo"]
        nn = [y for y in (nv or []) if y["tipo"] != "brazo"]
        out["no_brazo_igual"] = (nv is not None and len(nb) == len(nn) and all(
            all(x[k] == y[k] for k in cols_cmp) for x, y in zip(nb, nn)))
        out["nuevo_etiquetador_igual"] = (nv is not None and len(b) == len(nv) and all(
            all(x[k] == y[k] for k in COL_ETIQUETADOR) for x, y in zip(b, nv)))
        bb = [x for x in b if x["tipo"] == "brazo"]
        bn = [y for y in (nv or []) if y["tipo"] == "brazo"]
        out["brazo_base"], out["brazo_nuevo"] = len(bb), len(bn)
        out["brazo_mano_cambia"] = sum(1 for x in {(r["lado"], r["mano"]) for r in bb}
                                       ^ {(r["lado"], r["mano"]) for r in bn})
        malos = 0
        rebajadas = 0
        for y in n:
            ok, _m = PO.fisica_evento(y, float(_fps(puerta_dir, clip)))
            if y.get("publicable") == "True" and ok is False:
                malos += 1
            if y.get("publicable_clip") == "True" and y.get("publicable") == "False":
                rebajadas += 1
        out["a_publicable_y_mal"] = malos
        out["rebajadas"] = rebajadas
        # en la base, las columnas de siempre; `publicable` en base == publicable_clip en nuevo
        out["publicable_clip_igual"] = (len(b) == len(n) and all(
            x.get("publicable") == y.get("publicable_clip") for x, y in zip(b, n)))
    return out


def _fps(d, clip):
    r = _lee(pathlib.Path(d) / ("posicion_clip_%s.csv" % clip))
    try:
        return float(r[0]["fps"])
    except Exception:
        return 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=200)
    ap.add_argument("--extra", default=None, help="fichero con clips a incluir siempre")
    ap.add_argument("--solo-extra", action="store_true")
    ap.add_argument("--procesos", type=int, default=6)
    ap.add_argument("--semilla", type=int, default=20261009)
    ap.add_argument("--salida", default="D:/FENCINGLY_SCRATCH/agente_etapa7/copia_etapa7")
    a = ap.parse_args()
    import lotes as LO
    import rutas as RU
    ids = LO.conjunto("anotados")
    con = sorted(c for c in ids if (RU.PISTA / ("serie_eventos_%s.csv" % c)).is_file())
    random.Random(a.semilla).shuffle(con)
    sel = [] if a.solo_extra else con[:a.muestra]
    if a.extra:
        sel += [l.strip() for l in open(a.extra, encoding="utf-8") if l.strip() and l.strip() not in sel]
    print("clips: %d" % len(sel), flush=True)
    from multiprocessing import Pool
    res = []
    with Pool(a.procesos) as pool:
        for i, r in enumerate(pool.imap_unordered(uno, [(c, a.salida) for c in sel])):
            res.append(r)
            if (i + 1) % 25 == 0:
                print("   %d / %d" % (i + 1, len(sel)), flush=True)
    keys = ["clip", "hay_disco", "base", "puerta", "nuevo", "error", "n_filas", "base_igual_disco",
            "mismas_filas", "etiquetador_igual", "puerta_resto_igual", "publicable_clip_igual",
            "no_brazo_igual", "nuevo_etiquetador_igual",
            "a_publicable_y_mal", "rebajadas", "brazo_base", "brazo_nuevo", "brazo_mano_cambia"]
    sal = pathlib.Path(a.salida)
    with open(sal / "comparacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(res)
    err = [r for r in res if r.get("error")]
    comp = [r for r in res if "etiquetador_igual" in r]
    con_d = [r for r in comp if r.get("hay_disco") and "base_igual_disco" in r]

    def cuenta(k, sub):
        return "%d de %d" % (sum(1 for r in sub if r.get(k)), len(sub))
    print()
    print("ETAPA 7 RECALCULADA EN COPIA: %d clips, %d con excepcion" % (len(res), len(err)))
    for r in err:
        print("   EXC %-38s %s" % (r["clip"], r["error"]))
        print("       " + r["traza"].strip().splitlines()[-3].strip() if r.get("traza") else "")
    print("base identica a disco          ", cuenta("base_igual_disco", con_d))
    print("PUERTA SOLA contra base:")
    print("  (c) columnas etiquetador igual ", cuenta("etiquetador_igual", comp))
    print("  resto de columnas igual        ", cuenta("puerta_resto_igual", comp))
    print("  publicable_clip == base        ", cuenta("publicable_clip_igual", comp))
    print("  (a) publicable=True y mal (e2e) %d filas" % sum(r.get("a_publicable_y_mal", 0) for r in comp))
    print("      rebajadas por la puerta     %d filas" % sum(r.get("rebajadas", 0) for r in comp))
    print("PUERTA + MANO_ACCION contra base:")
    print("  (f) filas no-brazo iguales     ", cuenta("no_brazo_igual", comp))
    print("  columnas etiquetador igual     ", cuenta("nuevo_etiquetador_igual", comp))
    print("brazo: filas base %d -> nuevo %d; (lado, mano) distintos en %d clips"
          % (sum(r.get("brazo_base", 0) for r in comp), sum(r.get("brazo_nuevo", 0) for r in comp),
             sum(1 for r in comp if r.get("brazo_mano_cambia"))))
    dif = [r["clip"] for r in con_d if not r.get("base_igual_disco")]
    if dif:
        print("base distinta de disco en: %s" % ", ".join(dif[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
