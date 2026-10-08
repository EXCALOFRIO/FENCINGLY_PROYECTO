"""EL CONTROL DEL CENTRO DEL CONO, y de donde sale la anchura de +-8,6 gr.

POR QUE EXISTE ESTE FICHERO. El usuario encontro mirando una figura que el cono no estaba centrado
en el florete, y el pie de la misma figura afirmaba que si. Ese defecto vivio hasta que alguien lo
vio; no habia ninguna comprobacion que lo cazara. Esto es esa comprobacion, y ademas es lo que cazo
mi segundo error del mismo dia -- poner de anchura los 4,2 gr del temblor -- antes de publicarlo.

TRES COSAS, Y CADA UNA CON SU CRITERIO DECLARADO:

  1. EL DEFECTO VIEJO, documentado con su cifra. |hoja - eje del ANTEBRAZO| sobre los cuadros con
     hoja vista. Es la distancia a la que caia la hoja del centro que se dibujaba, y la referencia
     del 2026-09-11 es `BabyishIdioticCockerspaniel` c0167 lado 0: **21,4 gr** contra un nucleo
     solido de +-14,4. No es un caso raro: pasa en 26 de 60 cuadros.

  2. EL CENTRADO, que es lo que se arreglo. Con `campo_origen == 0` el centro del campo tiene que
     ser el angulo MEDIDO de la hoja. **Criterio: |centro - direccion de la hoja| < 0,01 gr**, o sea
     igualdad numerica. Si falla, el campo no se ha centrado en la hoja y da igual todo lo demas.

  3. LA ANCHURA, y aqui esta la trampa que me pillo. La cunia tiene el vertice en la **raiz de la
     mano** y la recta detectada **no pasa por ella**. Asi que la anchura no puede ser la
     incertidumbre de la DIRECCION (4,2 gr, el temblor de `e27`): tiene que absorber tambien esa
     separacion. Se mide la desviacion de los dos extremos de la hoja vista respecto del centro,
     **vista desde el vertice**, y sale que el extremo de la PUNTA esta a p90 8,6 gr y el extremo de
     la MANO a p90 35,0 gr, que es el artefacto de mirar un punto casi pegado al vertice
     (`r/L = 0,24`, y 7 px de separacion ahi subtienden 9 gr).
     **Criterio: el cono dibujado contiene LA PUNTA de la hoja en >= 0,90 de los cuadros**, que es
     lo que un p90 significa. Contener el segmento ENTERO no es un criterio valido y por eso no se
     usa: ni los +-33,0 ya publicados lo cumplen (0,88), porque el extremo de la mano no mide la
     direccion de la amenaza.

QUE NO COMPRUEBA: nada sobre si la hoja detectada es de verdad una hoja. Eso lo deciden `e5`/`e13` y
aqui se leen de disco. Este control es sobre el DIBUJO, no sobre la deteccion.

USO:  python verifica_cono_campo.py [CLIP ...]      CPU, sin GPU y sin decodificar video
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

CODIGO = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, str(CODIGO))

import cono3d_lib as C3
import cono_lib as CO
import e20_estado_del_cono as E20
import e21_pinta_cono_arreglado as E21

CLIPS = ["proudDungBeetleBoglenTerrier", "uptightSlothEveningBat", "BabyishIdioticCockerspaniel"]
"""Los tres que se publican. Son los unicos que hay que defender en un repo publico, y son la
muestra de la que sale el p90 de 8,6: o sea que esta medida es DENTRO DE MUESTRA y se declara."""
COBERTURA_MIN = 0.90
"""La fraccion de cuadros en que el cono tiene que contener la punta. Es 0,90 porque la anchura ES el
p90 de esa misma desviacion: el criterio y la constante son la misma afirmacion, y por eso no hay
umbral que ajustar. Si alguien cambia `APERT_HOJA_VISTA_SEMI_GR` sin volver a medir, esto falla."""


def campo_del_clip(clip):
    f = CO.PRUEBAS / ("cono_campo_%s.npz" % clip)
    if not f.is_file():
        return None
    with np.load(f, allow_pickle=True) as z:
        return {"mu": z["campo_mu_gr"], "sigma": z["campo_sigma_gr"], "org": z["campo_origen"],
                "T": int(len(z["campo_origen"])),
                "temblor": float(z["temblor_hoja_gr"]) if "temblor_hoja_gr" in z.files
                else float("nan")}


def un_clip(clip, hojas):
    """Una fila por cuadro y lado con `campo_origen == 0`. Devuelve `(filas, fallos)`."""
    campo = campo_del_clip(clip)
    if campo is None:
        return [], ["%s sin cono_campo_*.npz" % clip]
    d = E20.carga(clip)
    kp, sc, B, W, H = d["kp"], d["sc"], d["B"], d["W"], d["H"]
    if campo["T"] != d["T"]:
        return [], ["%s: T del campo %d != T de las poses %d, el indexado NO esta alineado"
                    % (clip, campo["T"], d["T"])]
    filas, fallos = [], []
    for lado in (0, 1):
        mano, det = CO.mano_armada(kp, sc, B, W, H, lado)
        if mano is None:
            fallos.append("%s lado %d sin mano armada decidible" % (clip, lado))
            continue
        vert, u, ok, _lg = C3.eje_brazo_2d(kp, sc, lado, mano, "codo")
        _pk, pxm = CO.escala_px_m(B, lado, H)
        L = C3.ALCANCE_M * pxm
        ang_eje = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        for k in np.where(campo["org"][:, lado] == 0)[0]:
            ln = hojas.get((clip, int(k), lado))
            if ln is None or not np.isfinite(vert[k]).all():
                continue
            mu, ap = float(campo["mu"][k, lado]), float(campo["sigma"][k, lado])
            if not (np.isfinite(mu) and np.isfinite(ap) and ap > 0):
                continue
            if np.cos(np.radians(mu - ang_eje[k])) < 0:
                """Los 180 gr de un segmento sin sentido propio, resueltos contra el eje de ESTAS
                poses. En los tres clips publicados no hace falta ni una vez; se deja porque el NPZ
                oriento la hoja contra otras poses y con otro clip podria hacer falta."""
                mu += 180.0
            p1 = np.asarray(ln["p1"], "float64")
            p2 = np.asarray(ln["p2"], "float64")
            r1, r2 = np.linalg.norm(p1 - vert[k]), np.linalg.norm(p2 - vert[k])
            p_cerca, p_lejos = (p1, p2) if r1 < r2 else (p2, p1)
            dev = {}
            for nom, p in (("cerca", p_cerca), ("lejos", p_lejos)):
                v = p - vert[k]
                dev[nom] = abs(float(CO.dif_ang(
                    float(np.degrees(np.arctan2(v[1], v[0]))), mu)))
            dd = p2 - p1
            mu_hoja = float(np.degrees(np.arctan2(dd[1], dd[0])))
            if np.cos(np.radians(mu_hoja - mu)) < 0:
                mu_hoja += 180.0
            filas.append({
                "clip": clip, "lado": lado, "cuadro": int(k), "apert_gr": ap,
                "dif_centro_vs_hoja": abs(float(CO.dif_ang(mu, mu_hoja))),
                "dif_eje_gr": abs(float(CO.dif_ang(mu, ang_eje[k]))),
                "dev_punta": dev["lejos"], "dev_mano": dev["cerca"],
                "punta_dentro": dev["lejos"] <= ap,
                "r_sobre_L": float(min(r1, r2)) / max(L, 1e-9),
                "dudosa": bool(det.get("dudosa")),
                "frac_pie": float(det.get("frac_pie", float("nan")))})
    return filas, fallos


def main():
    clips = sys.argv[1:] or CLIPS
    hojas = E21.hojas_cacheadas()
    print("=" * 104)
    print("CONTROL DEL CENTRO Y LA ANCHURA DEL CONO  ·  sin GPU y sin decodificar video")
    print("=" * 104)
    print("  anchura vigente: hoja VISTA +-%.1f gr  ·  SUPUESTA +-%.1f gr  ·  nucleo p50 %.1f  ·  "
          "temblor de la direccion %.1f"
          % (C3.APERT_HOJA_VISTA_SEMI_GR, C3.APERT_ENVOLVENTE_SEMI_GR,
             C3.APERT_NUCLEO_SEMI_GR, C3.TEMBLOR_HOJA_SEMI_GR))
    print("  hojas en cache: %d" % len(hojas))
    print()
    todas, fallos = [], []
    for clip in clips:
        fs, fl = un_clip(clip, hojas)
        todas.extend(fs)
        fallos.extend(fl)
        if not fs:
            print("  %-30s sin cuadros con hoja vista utilizables" % clip[:30])
            continue
        dif = np.array([r["dif_eje_gr"] for r in fs])
        dentro = sum(1 for r in fs if r["punta_dentro"])
        print("  %-30s %3d cuadros  ·  |hoja-antebrazo| p50 %5.2f max %5.2f  ·  %2d de %2d caian "
              "fuera del nucleo viejo  ·  punta dentro %2d de %2d"
              % (clip[:30], len(fs), float(np.median(dif)), float(dif.max()),
                 int((dif > C3.APERT_NUCLEO_SEMI_GR).sum()), len(fs), dentro, len(fs)))
        for lado in (0, 1):
            sub = [r for r in fs if r["lado"] == lado]
            if sub:
                print("      lado %d: %2d cuadros  ·  mano %s" % (
                    lado, len(sub),
                    "DUDOSA, voto del pie %.3f" % sub[0]["frac_pie"] if sub[0]["dudosa"]
                    else "firme"))
    if not todas:
        print("\n  nada que medir.")
        return 1
    print()
    mal_centro = [r for r in todas if r["dif_centro_vs_hoja"] > 0.01]
    if mal_centro:
        fallos.append("el centro del campo NO es la direccion de la hoja en %d de %d cuadros "
                      "(max %.3f gr)" % (len(mal_centro), len(todas),
                                         max(r["dif_centro_vs_hoja"] for r in mal_centro)))
    dif = np.array([r["dif_eje_gr"] for r in todas])
    pu = np.array([r["dev_punta"] for r in todas])
    ma = np.array([r["dev_mano"] for r in todas])
    dentro = int(sum(1 for r in todas if r["punta_dentro"]))
    cob = dentro / len(todas)
    print("  1. EL DEFECTO VIEJO, sobre %d cuadros con hoja vista de %d clips:" % (len(todas),
                                                                                   len(clips)))
    print("     |hoja - eje del antebrazo|   p50 %5.2f   p90 %5.2f   max %5.2f gr"
          % (*np.percentile(dif, [50, 90]), dif.max()))
    print("     caian FUERA del nucleo solido de +-%.1f gr: **%d de %d**. El usuario lo vio en un"
          % (C3.APERT_NUCLEO_SEMI_GR, int((dif > C3.APERT_NUCLEO_SEMI_GR).sum()), len(todas)))
    print("     cuadro y no era un caso raro.")
    print()
    print("  2. EL CENTRADO: |centro del campo - direccion de la hoja| max %.4f gr  ->  %s"
          % (max(r["dif_centro_vs_hoja"] for r in todas),
             "FALLA" if mal_centro else "PASA, el campo esta centrado en la hoja medida"))
    print()
    print("  3. LA ANCHURA, desviacion desde el VERTICE respecto del centro:")
    print("     extremo LEJOS (la punta)   p50 %5.2f   p90 %5.2f   max %5.2f gr   <- de aqui sale "
          "el +-%.1f" % (*np.percentile(pu, [50, 90]), pu.max(), C3.APERT_HOJA_VISTA_SEMI_GR))
    print("     extremo CERCA de la mano   p50 %5.2f   p90 %5.2f   max %5.2f gr   <- artefacto del "
          "vertice, r/L p50 %.2f" % (*np.percentile(ma, [50, 90]), ma.max(),
                                     float(np.median([r["r_sobre_L"] for r in todas]))))
    print("     el cono dibujado contiene LA PUNTA en **%d de %d (%.3f)**, criterio >= %.2f  ->  %s"
          % (dentro, len(todas), cob, COBERTURA_MIN,
             "PASA" if cob >= COBERTURA_MIN else "FALLA"))
    for ap in (C3.TEMBLOR_HOJA_SEMI_GR, C3.APERT_HOJA_VISTA_SEMI_GR, C3.APERT_NUCLEO_SEMI_GR,
               C3.APERT_ENVOLVENTE_SEMI_GR):
        print("        con +-%5.1f gr habria contenido la punta en %2d de %2d (%.3f)"
              % (ap, int((pu <= ap).sum()), len(pu), float((pu <= ap).mean())))
    if cob < COBERTURA_MIN:
        fallos.append("el cono contiene la punta en %.3f de los cuadros, por debajo del %.2f "
                      "declarado" % (cob, COBERTURA_MIN))
    print()
    if fallos:
        print("  FALLA el control, %d motivos:" % len(fallos))
        for f in fallos:
            print("     %s" % f)
        return 1
    print("  PASA los tres controles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
