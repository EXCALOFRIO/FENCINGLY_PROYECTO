"""ETAPA E, paso 26: POR QUE LA HOJA SE VE EN UNOS CUADROS Y NO EN OTROS, con dos candidatos nuevos.

`e23` M4 tumbo la explicacion que el proyecto daba por buena: comparando la velocidad de la MANO en
los recortes con deteccion contra los sin deteccion sale **0,96x**, o sea ninguna diferencia. Asi que
el desenfoque de movimiento **no queda apoyado** y el mecanismo seguia sin identificar. Aqui se miden
los dos candidatos que quedaban, los dos sobre los recortes nuevos de `e25`, que llevan las dos
columnas calculadas en el momento de la deteccion:

  C1  CONTRASTE LOCAL de la region donde se busca la hoja (recorrido p90-p10 del gris, normalizado).
      Hipotesis: una hoja sobre pista clara o sobre publico oscuro se resuelve mejor que sobre un
      fondo del mismo tono. **Declarado antes: el contraste mediano de los recortes CON deteccion
      tiene que ser al menos 1,15x el de los SIN**, que es el mismo margen que se le exigio a la
      velocidad de la mano en `e23` M4, para que las dos hipotesis se juzguen igual.

  C2  ESCORZO del antebrazo, largo proyectado partido por el alto de la caja. Es geometrico y no
      discutible: un arma perpendicular a la camara proyecta su largo entero y una de frente proyecta
      casi nada, asi que **si el escorzo manda, la tasa de deteccion tiene que subir con el escorzo**.
      Declarado igual: razon >= 1,15x entre el quintil mas extendido y el mas corto.

Y LA TERCERA PREGUNTA, que es del coordinador y no es de mecanismo sino de calidad del vertice:

  C3  ¿SUBE LA TASA DE DETECCION EN LOS TIRADORES DE MANO FIRME? El otro agente adopto en
      `cono_lib.mano_armada` la mano por pie adelantado con firmeza >= 0,85 y publica `dudosa`. Si el
      vertice de un tirador dudoso esta mal puesto, el recorte se centra donde no esta el arma y la
      hoja no se puede ver. **Declarado: tasa en los firmes >= 1,15x la de los dudosos.** Si sale
      asi, refuerza que excluir los dudosos es lo correcto; si no, la marca no ayuda a esto y se dice.

NINGUNA DE LAS TRES NECESITA GPU: las tres columnas las escribio `e25` mientras el recorte estaba en
memoria. Esto es CPU y lectura de JSON.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono_lib as CL
import e23_campo_de_probabilidad as E23

PRUEBAS = CL.PRUEBAS
MARGEN = 1.15
N_QUINT = 5


def carga():
    """Solo los recortes de `e25`, que son los unicos que traen `contraste` y `escorzo`."""
    out = []
    for f in sorted(PRUEBAS.glob("cono_hoja_*.json")):
        clip = f.name[len("cono_hoja_"):-len(".json")]
        for r in json.loads(f.read_text(encoding="utf-8")):
            if "contraste" not in r or r.get("contaminado"):
                continue
            out.append({"clip": clip, "asalto": E23.asalto(clip), "cuadro": r["cuadro"],
                        "lado": r["lado"], "contraste": r["contraste"], "escorzo": r["escorzo"],
                        "dudosa": bool(r.get("dudosa", True)),
                        "firme": bool(r.get("firme_pie", False)),
                        "det": bool(r["det"].get("p99.5", {}).get("linea"))})
    return out


def compara(reg, campo, nom):
    con = np.array([r[campo] for r in reg if r["det"] and np.isfinite(r[campo])])
    sin = np.array([r[campo] for r in reg if not r["det"] and np.isfinite(r[campo])])
    if con.size < 5 or sin.size < 5:
        print("    %-26s n insuficiente (con=%d, sin=%d)" % (nom, con.size, sin.size))
        return None
    raz = float(np.median(con)) / max(float(np.median(sin)), 1e-9)
    print("    %-26s CON n=%3d p50 %.4f  ·  SIN n=%3d p50 %.4f  ·  razon **%.2fx**  ->  %s" % (
        nom, con.size, np.median(con), sin.size, np.median(sin), raz,
        "APOYA" if raz >= MARGEN else ("en contra" if raz <= 1 / MARGEN else "no separa")))
    return {"n_con": int(con.size), "n_sin": int(sin.size), "p50_con": float(np.median(con)),
            "p50_sin": float(np.median(sin)), "razon": raz, "apoya": bool(raz >= MARGEN)}


def por_quintil(reg, campo, nom):
    v = np.array([r[campo] for r in reg if np.isfinite(r[campo])])
    d = np.array([r["det"] for r in reg if np.isfinite(r[campo])])
    if v.size < 40:
        print("    %-26s n insuficiente (%d)" % (nom, v.size))
        return None
    bo = np.percentile(v, np.linspace(0, 100, N_QUINT + 1))
    tasas, ns = [], []
    for j in range(N_QUINT):
        m = (v >= bo[j]) & (v <= bo[j + 1])
        ns.append(int(m.sum()))
        tasas.append(float(d[m].mean()) if m.sum() else float("nan"))
    print("    %-26s tasa por quintil: %s" % (
        nom, "  ".join("%.0f%%(n=%d)" % (100 * t, n) for t, n in zip(tasas, ns))))
    raz = tasas[-1] / max(tasas[0], 1e-9)
    print("    %-26s quintil alto / quintil bajo = **%.2fx**  ->  %s" % (
        "", raz, "APOYA" if raz >= MARGEN else ("en contra" if raz <= 1 / MARGEN else "no separa")))
    return {"bordes": bo.tolist(), "tasas": tasas, "n": ns, "razon_alto_bajo": raz,
            "apoya": bool(raz >= MARGEN)}


def main():
    reg = carga()
    print("=" * 104)
    print("POR QUE SE VE LA HOJA  ·  %d recortes de `e25` con las dos columnas nuevas" % len(reg))
    print("=" * 104)
    if not reg:
        print("  no hay recortes de `e25`: hay que correr `e25_recoge_hojas.py` primero")
        return
    asaltos = sorted(set(r["asalto"] for r in reg))
    print("  asaltos distintos: %d  ·  tasa global **%.1f %%**" % (
        len(asaltos), 100 * np.mean([r["det"] for r in reg])))
    print("  %-32s %6s %6s %8s" % ("asalto", "n", "det", "tasa"))
    for a in asaltos:
        s = [r for r in reg if r["asalto"] == a]
        print("  %-32s %6d %6d %7.1f %%" % (
            a[:32], len(s), sum(r["det"] for r in s), 100 * np.mean([r["det"] for r in s])))

    res = {"n": len(reg), "asaltos": len(asaltos),
           "tasa_global": float(np.mean([r["det"] for r in reg]))}
    print()
    print("  C1 · CONTRASTE LOCAL (margen declarado %.2fx)" % MARGEN)
    res["c1_contraste"] = compara(reg, "contraste", "contraste del gris")
    res["c1_quintil"] = por_quintil(reg, "contraste", "tasa por contraste")
    print()
    print("  C2 · ESCORZO DEL ANTEBRAZO (margen declarado %.2fx)" % MARGEN)
    res["c2_escorzo"] = compara(reg, "escorzo", "escorzo (largo/alto)")
    res["c2_quintil"] = por_quintil(reg, "escorzo", "tasa por escorzo")

    print()
    print("  C3 · ¿SUBE LA TASA EN LOS TIRADORES DE MANO FIRME? (margen declarado %.2fx)" % MARGEN)
    fir = [r for r in reg if r["firme"]]
    dud = [r for r in reg if not r["firme"]]
    if len(fir) >= 10 and len(dud) >= 10:
        tf = float(np.mean([r["det"] for r in fir]))
        td = float(np.mean([r["det"] for r in dud]))
        raz = tf / max(td, 1e-9)
        print("    firmes   n=%3d  tasa **%.1f %%**" % (len(fir), 100 * tf))
        print("    dudosos  n=%3d  tasa **%.1f %%**" % (len(dud), 100 * td))
        print("    razon firme/dudoso = **%.2fx**  ->  %s" % (
            raz, "APOYA excluir los dudosos" if raz >= MARGEN else
            ("en contra" if raz <= 1 / MARGEN else "no separa")))
        res["c3_firmeza"] = {"n_firme": len(fir), "n_dudoso": len(dud), "tasa_firme": tf,
                             "tasa_dudoso": td, "razon": raz, "apoya": bool(raz >= MARGEN)}
    else:
        print("    n insuficiente: firmes %d, dudosos %d." % (len(fir), len(dud)))
        print("    Y ESO YA ES UN RESULTADO: si casi ningun tirador sale firme, la marca `dudosa`")
        print("    no puede usarse como puerta del cono sin quedarse casi sin datos.")
        res["c3_firmeza"] = {"n_firme": len(fir), "n_dudoso": len(dud), "concluyente": False}

    print()
    print("  C4 · LA ESCALA APARENTE (px/m), que es un candidato que aparecio al mirar los clips")
    print("  22.11 ya habia probado px/m y NO explicaba la diferencia 720p contra 1080p (221 px/m")
    print("  contra 183-261). Pero los clips nuevos llegan a **360 px/m**, un tramo que 22.11 no")
    print("  cubria, asi que la pregunta vuelve a estar viva **dentro** de 1080p.")
    filas = []
    for a in asaltos:
        s = [r for r in reg if r["asalto"] == a]
        pxm = []
        for c in sorted(set(r["clip"] for r in s)):
            try:
                with np.load(PRUEBAS / ("cono_poses_%s.npz" % c), allow_pickle=True) as z:
                    B, H = z["B"], int(z["H"])
                    ks = (z["cuadros_muestreados"] if "cuadros_muestreados" in z.files
                          else np.arange(B.shape[0]))
                """LA ESCALA SE MIDE EN LOS CUADROS MUESTREADOS, no en el clip entero. La mediana del
                clip mezcla planos que no se usaron y en `GentleBlue` daba 360 px/m cuando en los
                cuadros que de verdad se miraron son 312."""
                fr = np.array([float(B[k, l, 3] - B[k, l, 1]) for k in ks for l in (0, 1)
                               if np.isfinite(B[k, l]).all()])
                if fr.size:
                    pxm.append(float(np.median(fr)) * H / CL.ALTURA_M)
                    filas.append({"asalto": a, "clip": c, "pxm": pxm[-1],
                                  "alto_p95_frac": float(np.percentile(fr, 95)),
                                  "frac_primer_plano": float(np.mean(fr > 0.75)),
                                  "tasa": float(np.mean([r["det"] for r in s]))})
            except Exception:
                pass
    if len(filas) >= 4:
        px = np.array([f["pxm"] for f in filas])
        ta = np.array([f["tasa"] for f in filas])
        o1, o2 = np.argsort(np.argsort(px)), np.argsort(np.argsort(ta))
        sp = float(np.corrcoef(o1, o2)[0, 1])
        print("  %-32s %8s %10s %12s" % ("clip", "px/m", "tasa", "alto p95/H"))
        for f in sorted(filas, key=lambda x: -x["pxm"]):
            print("  %-32s %8.0f %9.1f %% %11.2f %s" % (
                f["clip"][:32], f["pxm"], 100 * f["tasa"], f["alto_p95_frac"],
                "<- PRIMER PLANO" if f["alto_p95_frac"] > 0.80 else ""))
        print("  spearman entre px/m y tasa sobre %d clips = **%+.2f**  ->  %s" % (
            len(filas), sp, "APOYA que la escala aparente manda" if sp >= 0.5 else
            ("en contra" if sp <= -0.5 else "no concluye con esta n")))
        res["c4_escala"] = {"n_clips": len(filas), "spearman": sp,
                            "clips": filas, "apoya": bool(sp >= 0.5)}
        pp = float(np.mean([f["frac_primer_plano"] for f in filas]))
        print("  REGIMEN DE PLANO EN LOS CUADROS MUESTREADOS: fraccion por encima de 0,75 del alto")
        print("  del cuadro = **%.1f %%**." % (100 * pp))
        if pp < 0.02:
            print("  O SEA QUE NO HAY SESGO DE PRIMER PLANO EN ESTA MUESTRA, y la razon es")
            print("  ESTRUCTURAL y no suerte: las rafagas exigen **los DOS tiradores presentes**, y en")
            print("  un primer plano de uno el otro no cabe en el cuadro. El filtro de plano de la")
            print("  seccion 12 queda satisfecho por construccion.")
            print("  Y ADEMAS ESO DESCONFUNDE C4: como el regimen de plano no varia (todo normal),")
            print("  el rango de px/m que se ve aqui, **%.0f a %.0f**, es variacion de ESCALA DE"
                  % (min(f["pxm"] for f in filas), max(f["pxm"] for f in filas)))
            print("  CAMARA a plano constante, no de encuadre. La atribucion a px/m es limpia.")
        res["c4_escala"]["frac_primer_plano"] = pp
        cerca = [f for f in filas if f["frac_primer_plano"] > 0.10]
        if cerca:
            print("  AVISO DE MUESTRA: %d clip(s) estan en regimen de PRIMER PLANO (alto p95 > 0,80"
                  % len(cerca))
            print("  del cuadro): %s. La seccion 12 mide que ese regimen es un plano distinto y la"
                  % ", ".join(f["clip"][:24] for f in cerca))
            print("  tuberia lo FILTRA, asi que sus hojas no son representativas del corpus sobre el")
            print("  que correra el cono. Van contadas aparte en el desglose por asalto.")
            res["c4_escala"]["primer_plano"] = [f["clip"] for f in cerca]
    print()
    apoyan = [k for k in ("c1_contraste", "c2_escorzo", "c4_escala")
              if res.get(k) and res[k].get("apoya")]
    print("  VEREDICTO DEL MECANISMO: %s" % (
        "apoyan " + " y ".join(apoyan) if apoyan else
        "NINGUNO de los dos candidatos separa. Con el desenfoque ya descartado en `e23` M4, el "
        "mecanismo sigue SIN IDENTIFICAR"))
    (PRUEBAS / "cono_e26_por_que.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("guardado %s" % (PRUEBAS / "cono_e26_por_que.json"))


if __name__ == "__main__":
    main()
