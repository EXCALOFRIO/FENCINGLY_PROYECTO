"""La puerta fisica POR EVENTO de la etapa 7, medida sobre TODO lo ya procesado (criterio en
`04_documentacion/informes/ETAPA7_PUERTAS_2026-10-09.md` §1, escrito antes de correr esto).

Lee `serie_eventos_<clip>.csv` de `02_validacion/pista` SIN TOCARLOS, aplica en memoria la MISMA
funcion que aplica la etapa 7 (`posicion.aplica_puerta_evento`, no una copia) y deja las series
con la puerta en `--salida` para poder mirarlas. Da:

  (a) filas con `publicable=True` que violan la fisica, antes y despues (despues tiene que ser 0);
  (b) que fraccion de las filas que hoy son `publicable=True` rebaja la puerta (criterio <= 5 %);
  y, sin ser puerta, cuantas de las filas de la auditoria del 09/10 (`v > 8` o `|x| > 8`) caza.

    python verifica_puerta_evento.py [--conjunto anotados] [--salida D:/FENCINGLY_SCRATCH/agente_etapa7/series_con_puerta]
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "01_extraccion" / "codigo"))
import catalogo as CA  # noqa: E402
import posicion as PO  # noqa: E402
import rutas as RU  # noqa: E402


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _fps(c):
    p = RU.PISTA / ("posicion_clip_%s.csv" % c)
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            r = next(csv.DictReader(fh), None)
        v = _f((r or {}).get("fps"))
        if v:
            return v, "posicion_clip"
    return 30.0, "defecto_30"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="anotados")
    ap.add_argument("--salida", default="D:/FENCINGLY_SCRATCH/agente_etapa7/series_con_puerta")
    a = ap.parse_args()
    import lotes as LO
    sal = pathlib.Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    man = CA.manifiesto().set_index("clip_id")
    ids = LO.conjunto(a.conjunto)
    n_cl = n_fil = 0
    antes_pub = antes_mal = despues_mal = rebajadas = 0
    aud = aud_cazadas = 0
    por_motivo = collections.Counter()
    por_tipo = collections.Counter()
    por_tipo_pub = collections.Counter()
    por_arma = collections.Counter()
    por_arma_pub = collections.Counter()
    clips_rebajados = set()
    fps_via = collections.Counter()
    sin_cazar = []
    filas_rebajadas = []
    for c in ids:
        p = RU.PISTA / ("serie_eventos_%s.csv" % c)
        if not p.is_file() or p.stat().st_size == 0:
            continue
        with open(p, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames or [])
            filas = list(rd)
        fps, via = _fps(c)
        fps_via[via] += 1
        n_cl += 1
        arma = man.loc[c, "weapon"] if c in man.index else "?"
        out = []
        for r in filas:
            n_fil += 1
            pub = r.get("publicable") == "True"
            sinh = r.get("sin_homografia") == "True"
            g = dict(r)
            g["publicable"] = pub
            ok, _m = PO.fisica_evento(g, fps)
            if pub:
                antes_pub += 1
                por_tipo_pub[r["tipo"]] += 1
                por_arma_pub[arma] += 1
                if ok is False:
                    antes_mal += 1
            v = _f(r.get("velocidad_m_s"))
            xs = [x for x in (_f(r.get("x_inicio_m")), _f(r.get("x_fin_m"))) if x is not None]
            es_aud = pub and not sinh and ((v is not None and v > 8) or any(abs(x) > 8 for x in xs))
            PO.aplica_puerta_evento(g, fps)
            if g["publicable"] is True and PO.fisica_evento(g, fps)[0] is False:
                despues_mal += 1
            if pub and g["publicable"] is False:
                rebajadas += 1
                clips_rebajados.add(c)
                por_motivo[g["motivo_no_publicable"]] += 1
                por_tipo[r["tipo"]] += 1
                por_arma[arma] += 1
                filas_rebajadas.append((c, arma, r["tipo"], r.get("lado"), r.get("cuadro_inicio"),
                                        v, r.get("x_inicio_m"), r.get("x_fin_m"),
                                        r.get("duracion_s"), g["motivo_no_publicable"]))
            if es_aud:
                aud += 1
                if g["publicable"] is False:
                    aud_cazadas += 1
                else:
                    sin_cazar.append((c, r["tipo"], v, xs, r.get("duracion_s")))
            out.append(g)
        cols2 = cols + [k for k in ("publicable_clip", "fisica_evento", "motivo_no_publicable")
                        if k not in cols]
        with open(sal / p.name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols2, extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
    with open(sal.parent / "puerta_evento_filas_rebajadas.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "arma", "tipo", "lado", "cuadro_inicio", "velocidad_m_s",
                    "x_inicio_m", "x_fin_m", "duracion_s", "motivo"])
        w.writerows(filas_rebajadas)

    print("PUERTA FISICA POR EVENTO, sobre lo ya procesado de `%s`" % a.conjunto)
    print("=" * 78)
    print("clips con serie: %d   filas: %d   fps de: %s" % (n_cl, n_fil, dict(fps_via)))
    print("filas publicable=True ANTES: %d" % antes_pub)
    print()
    print("(a) publicable=True que violan la fisica   antes %d   DESPUES %d" % (antes_mal, despues_mal))
    fr = rebajadas / max(antes_pub, 1)
    print("(b) rebajadas por la puerta: %d de %d = %.4f  (criterio <= 0,05)  en %d clips"
          % (rebajadas, antes_pub, fr, len(clips_rebajados)))
    print("    por motivo:", dict(por_motivo.most_common()))
    print("    por tipo (rebajadas / publicables):")
    for t, n in por_tipo_pub.most_common():
        print("      %-8s %6d / %7d = %.4f" % (t, por_tipo[t], n, por_tipo[t] / max(n, 1)))
    print("    por arma (rebajadas / publicables):")
    for t, n in por_arma_pub.most_common():
        print("      %-8s %6d / %7d = %.4f" % (t, por_arma[t], n, por_arma[t] / max(n, 1)))
    print()
    print("auditoria 09/10 (v>8 o |x|>8, publicable, con homografia): %d filas, cazadas %d (%.3f)"
          % (aud, aud_cazadas, aud_cazadas / max(aud, 1)))
    if sin_cazar:
        xm = sorted(max(abs(x) for x in s[3]) if s[3] else 0 for s in sin_cazar)
        print("   sin cazar: %d; su max|x| va de %.2f a %.2f m (mediana %.2f); ninguna con v > 8"
              % (len(sin_cazar), xm[0], xm[-1], xm[len(xm) // 2]))
        for s in sin_cazar[:8]:
            print("     ", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
