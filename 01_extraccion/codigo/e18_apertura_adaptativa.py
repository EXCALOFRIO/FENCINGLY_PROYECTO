"""ETAPA E, paso 18: ARREGLAR EL DEFECTO QUE MAS PESA. La apertura que crece con el escorzo.

EL DEFECTO, medido en `e17`. El eje del brazo se desestabiliza cuando el brazo esta ESCORZADO: en el
quintil de brazo proyectado mas CORTO el eje salta una mediana de **3,06 grados** entre cuadros
consecutivos, contra **0,93** en el quintil mas largo. Razon **2,22x** (rango 1,45 a 5,53 en 7 clips).
Y la razon es geometrica y no discutible: con menos brazo de palanca, un pixel de error de la pose
gira mas el eje.

QUE ESTA MAL DEL CONO ACTUAL. La apertura es **constante**, la misma con el brazo extendido de lado
que con el brazo apuntando a la camara. Asi que en los cuadros escorzados el cono es demasiado
ESTRECHO para su propio ruido: promete una precision que el eje no tiene, y ahi es donde el solape
aparece y desaparece de un cuadro al siguiente.

EL ARREGLO, declarado antes de medir. La apertura tiene que cubrir DOS errores independientes:
  - el error de predecir la hoja desde el brazo, que `e14` midio en 23,1 grados (n=6, empate entre
    ejes), y que no depende del escorzo
  - el TEMBLOR PROPIO del eje en ese regimen de escorzo, que es lo que `e17` acaba de medir
Al ser de origen independiente se suman **en cuadratura**, que es lo que se hace con dos errores que
no estan correlacionados:

    apertura(k) = raiz( APERT^2 + temblor_p90(largo_proyectado(k))^2 )

`temblor_p90` se mide del propio clip por quintiles del largo proyectado, asi que **no hay ninguna
constante nueva inventada**: sale de los datos del clip.

COMO SE JUZGA SI EL ARREGLO ES MEJOR, y aqui esta la clave de que se pueda juzgar SIN verdad-terreno:

  CRITERIO 1, EL PARPADEO. Un contacto de hojas real dura varios cuadros; un contacto que aparece en
  un cuadro y desaparece en el siguiente es ruido. Asi que se mide la fraccion de pares de cuadros
  consecutivos en los que el booleano «se solapan» CAMBIA. **El arreglo es mejor si el parpadeo
  baja.** Y hay referencia previa: `e3` midio este mismo parpadeo, con el eje del codo y apertura
  constante, en el **3,5-4,9 % de los pares**.

  CRITERIO 2, QUE NO SE SATURE. Ensanchar una apertura siempre baja el parpadeo si se ensancha
  bastante, porque al final se solapa siempre. Asi que el arreglo solo vale si la frecuencia de
  solape **sigue dentro del 5-80 %** declarado en `e15`. Bajar el parpadeo saturando la senal no es
  arreglar, es taparla, y este criterio es el que lo impide.

  CONTROL DECLARADO: se compara tambien con una apertura constante SUBIDA al mismo valor MEDIO que la
  adaptativa. Si la constante equivalente baja el parpadeo igual, entonces lo que ayuda es «mas
  apertura» y no «apertura donde hace falta», y el arreglo no aporta nada propio. Este control es el
  que distingue las dos explicaciones.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cono3d_lib as C3
import cono_lib as CL

PRUEBAS = CL.PRUEBAS
APERT = 23.1
FREC_MIN, FREC_MAX = 0.05, 0.80
N_QUINT = 5


def clips():
    corpus = set(x.strip() for x in
                 (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    return [f.name[len("cono_poses_"):-4] for f in sorted(PRUEBAS.glob("cono_poses_*.npz"))
            if not f.name.endswith("_512x384.npz")
            and f.name[len("cono_poses_"):-4] in corpus]


def prepara(clip):
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        pres = z["presente"]
        mask = z["mask"] if "mask" in z.files else np.ones(T, dtype=bool)
    g = {"W": W, "H": H, "T": T, "lados": {}}
    for lado in (0, 1):
        m, _d = CL.mano_armada(kp, sc, B, W, H, lado)
        if m is None:
            return None
        vert, u, ok, largo = C3.eje_brazo_2d(kp, sc, lado, m, "hombro")
        _pk, pxm = CL.escala_px_m(B, lado, H)
        g["lados"][lado] = {"vert": vert, "u": u, "ok": ok, "largo": largo, "pxm": pxm,
                            "L": CL.HOJA_M * pxm}
    g["dos"] = np.array([bool(mask[k] and pres[k].sum() == 2) for k in range(T)])
    return g


def temblor_por_quintil(g, lado):
    """p90 del salto del eje entre cuadros consecutivos, por quintil del largo proyectado.

    Se mide EN EL PROPIO CLIP, asi que no introduce ninguna constante nueva.
    """
    d = g["lados"][lado]
    a = np.degrees(np.arctan2(d["u"][:, 1], d["u"][:, 0]))
    p = d["ok"][:-1] & d["ok"][1:] & g["dos"][:-1] & g["dos"][1:]
    lg = d["largo"][:-1][p]
    sa = np.abs(CL.dif_ang(a[1:][p], a[:-1][p]))
    m = np.isfinite(lg) & np.isfinite(sa)
    if m.sum() < 25:
        return None
    lg, sa = lg[m], sa[m]
    bordes = np.percentile(lg, np.linspace(0, 100, N_QUINT + 1))
    tabla = []
    for j in range(N_QUINT):
        sel = (lg >= bordes[j]) & (lg <= bordes[j + 1])
        tabla.append(float(np.percentile(sa[sel], 90)) if sel.sum() >= 5 else float("nan"))
    tabla = np.array(tabla)
    if not np.isfinite(tabla).any():
        return None
    tabla[~np.isfinite(tabla)] = float(np.nanmedian(tabla))
    return bordes, tabla


def apert_adaptativa(g, lado, tq):
    """Apertura por cuadro: `raiz(APERT^2 + temblor_p90(quintil)^2)`."""
    d = g["lados"][lado]
    bordes, tabla = tq
    idx = np.clip(np.searchsorted(bordes[1:-1], d["largo"], side="right"), 0, N_QUINT - 1)
    return np.sqrt(APERT ** 2 + tabla[idx] ** 2)


def corre(g, ap0, ap1):
    """Devuelve el booleano de solape y la validez. `ap0`/`ap1` pueden ser escalares o por cuadro."""
    T = g["T"]
    d0, d1 = g["lados"][0], g["lados"][1]
    cru = np.zeros(T, dtype=bool)
    val = np.zeros(T, dtype=bool)
    for k in range(T):
        if not (g["dos"][k] and d0["ok"][k] and d1["ok"][k]):
            continue
        a0 = float(ap0[k]) if np.ndim(ap0) else float(ap0)
        a1 = float(ap1[k]) if np.ndim(ap1) else float(ap1)
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        val[k] = True
        r = CL.cruce(d0["vert"][k], d0["u"][k], d0["L"], a0,
                     d1["vert"][k], d1["u"][k], d1["L"], a1)
        cru[k] = r["n_inter"] > 0
    return cru, val


def parpadeo(cru, val):
    p = val[:-1] & val[1:]
    return (float((cru[1:][p] != cru[:-1][p]).mean()), int(p.sum())) if p.any() else (
        float("nan"), 0)


res = {}
print("=" * 104)
print("APERTURA ADAPTATIVA AL ESCORZO  ·  arreglo del defecto D4  ·  sin GPU")
print("=" * 104)
print("  %-30s | %-22s | %-22s | %-22s" % (
    "clip", "CONSTANTE 23,1 gr", "ADAPTATIVA", "CONSTANTE equivalente"))
print("  %-30s | %7s %7s %6s | %7s %7s %6s | %7s %7s %6s" % (
    "", "solape", "parpad", "ap.med", "solape", "parpad", "ap.med", "solape", "parpad", "ap.med"))
for clip in clips():
    g = prepara(clip)
    if g is None:
        continue
    tq = {lado: temblor_por_quintil(g, lado) for lado in (0, 1)}
    if any(v is None for v in tq.values()):
        print("  %-30s sin datos suficientes para el temblor por quintil" % clip[:30])
        continue
    apA = {lado: apert_adaptativa(g, lado, tq[lado]) for lado in (0, 1)}
    med = float(np.nanmean([np.nanmean(apA[0]), np.nanmean(apA[1])]))

    cC, vC = corre(g, APERT, APERT)
    cA, vA = corre(g, apA[0], apA[1])
    cE, vE = corre(g, med, med)
    fila = []
    for cru, val in ((cC, vC), (cA, vA), (cE, vE)):
        fr = float(cru[val].mean()) if val.any() else float("nan")
        pa, npa = parpadeo(cru, val)
        fila.append((fr, pa, npa))
    print("  %-30s | %6.1f %% %6.1f %% %6.1f | %6.1f %% %6.1f %% %6.1f | %6.1f %% %6.1f %% %6.1f" % (
        clip[:30], 100 * fila[0][0], 100 * fila[0][1], APERT,
        100 * fila[1][0], 100 * fila[1][1], med,
        100 * fila[2][0], 100 * fila[2][1], med))
    res[clip] = {"apert_media_adaptativa": med,
                 "temblor_p90_por_quintil": {str(l): list(tq[l][1]) for l in (0, 1)},
                 "constante": {"solape": fila[0][0], "parpadeo": fila[0][1], "n_pares": fila[0][2]},
                 "adaptativa": {"solape": fila[1][0], "parpadeo": fila[1][1], "n_pares": fila[1][2]},
                 "constante_equivalente": {"solape": fila[2][0], "parpadeo": fila[2][1],
                                           "n_pares": fila[2][2]}}

if res:
    def med(nom, campo):
        v = [r[nom][campo] for r in res.values() if np.isfinite(r[nom][campo])]
        return float(np.median(v)) if v else float("nan")
    print()
    print("  MEDIANA SOBRE %d CLIPS" % len(res))
    print("  %-24s %10s %10s" % ("", "solape", "parpadeo"))
    for nom, et in (("constante", "constante 23,1 gr"), ("adaptativa", "ADAPTATIVA"),
                    ("constante_equivalente", "constante equivalente")):
        print("  %-24s %9.1f %% %9.1f %%" % (et, 100 * med(nom, "solape"),
                                             100 * med(nom, "parpadeo")))
    pc, pa, pe = med("constante", "parpadeo"), med("adaptativa", "parpadeo"), \
        med("constante_equivalente", "parpadeo")
    fa = med("adaptativa", "solape")
    print()
    print("  CRITERIO 1 (baja el parpadeo?): %.1f %% -> %.1f %%  =  %s" % (
        100 * pc, 100 * pa, "SI, baja %.1f puntos" % (100 * (pc - pa)) if pa < pc else
        "NO, sube %.1f puntos" % (100 * (pa - pc))))
    print("  CRITERIO 2 (sigue informando?): solape %.1f %%  ->  %s" % (
        100 * fa, "SI, dentro del 5-80 %" if FREC_MIN <= fa <= FREC_MAX else "NO, saturado"))
    print("  CONTROL (la constante equivalente hace lo mismo?): parpadeo %.1f %% contra %.1f %% de "
          "la adaptativa  ->  %s" % (
              100 * pe, 100 * pa,
              "el arreglo NO aporta nada propio: lo que ayuda es solo mas apertura"
              if abs(pe - pa) < 0.005 else
              "la adaptativa %s a la constante equivalente" % ("BATE" if pa < pe else "PIERDE")))
    print()
    print("  REFERENCIA PREVIA: `e3` midio este mismo parpadeo con el eje del CODO y apertura")
    print("  constante en el 3,5-4,9 % de los pares.")

(PRUEBAS / "cono_e18_apertura_adaptativa.json").write_text(
    json.dumps({"apert_base": APERT, "criterio_frec": [FREC_MIN, FREC_MAX], "clips": res},
               indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("guardado %s" % (PRUEBAS / "cono_e18_apertura_adaptativa.json"))
