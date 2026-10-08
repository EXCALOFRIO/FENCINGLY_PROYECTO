"""Aplica a los NPZ de poses YA CALCULADOS los dos filtros que se anadieron despues, sin GPU.

## POR QUE SE PUEDE HACER SIN VOLVER A PASAR EL MODELO

La tabla de checks encontro dos cosas en los NPZ existentes:

  P4  **10.140 keypoints entregados caian fuera de la caja de su propia persona**. El filtro que lo
      evita existia en `capas_limpias.limpia_keypoints`, que es la ruta de las capas densas, pero NO
      en `video_revision.poses_del_clip`, que es la ruta que produce y guarda las poses entregadas.
  P1  Seis NPZ escritos antes de que existiera el filtro de poses vacias tienen poses MEDIDAS con
      entre 0 y 8 puntos utiles del cuerpo dentro.

Los dos filtros son **geometricos y locales**: uno compara cada keypoint con la caja de su persona y
el otro cuenta puntos utiles. Los dos datos que necesitan, `KP`/`SC` y `B`, ya estan en el NPZ. Asi
que no hace falta volver a correr Sapiens 2, que son horas de GPU.

## LA PIEZA QUE LO HACE POSIBLE ES `origen`

En el NPZ, `origen` vale 0 donde la pose es MEDIDA y 1 o 2 donde se invento. O sea que se sabe
exactamente que poses vinieron del modelo y cuales son interpolacion, y se puede:

  1. quedarse solo con las MEDIDAS, tirando lo inventado
  2. aplicarles el filtro de keypoints fuera de caja
  3. aplicarles el filtro de poses vacias, que ahora si puede tirar alguna mas
  4. **volver a inventar** las que faltan, desde anclas ya limpias

El resultado es identico a lo que produciria la tuberia nueva. Y es la razon por la que en este
proyecto `origen` va guardado junto a los puntos y no se debe perder: sin esa marca este arreglo
seria imposible y habria que recalcular todo.

## LO QUE ESTE ARREGLO NO PUEDE HACER

No cambia las CAJAS. Si un NPZ se escribio con una pareja mal elegida (por ejemplo el arbitro, como
en `PowerfulAllLarva`), sigue teniendo al arbitro: eso se arregla en `pareja_v2` y exige volver a
pasar la pose. Este script arregla los puntos dentro de las cajas que hay, nada mas.

Los originales se guardan con sufijo `.previo` la primera vez, para poder comparar y para no perder
nada.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import video_revision as VR

PRUEBAS = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\pruebas")


def repara(f):
    z = np.load(f, allow_pickle=True)
    if "origen" not in z.files:
        return None
    KP = np.array(z["KP"], dtype="float32")
    SC = np.array(z["SC"], dtype="float32")
    B, pres, mask = np.array(z["B"]), np.array(z["pres"]), np.array(z["mask"])
    origen = np.array(z["origen"])
    W, H, T = int(z["W"]), int(z["H"]), int(z["T"])

    """Se borra lo INVENTADO para no filtrarlo como si fuera medido: un punto interpolado que cae
    fuera de la caja no es espurio, es la consecuencia de que su ancla lo estuviera, y lo que hay que
    arreglar es el ancla."""
    inv = origen > 0
    KP[inv] = np.nan
    SC[inv] = 0.0

    fuera = 0
    for k in range(T):
        for lado in (0, 1):
            if origen[k, lado] != 0:
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            x, y = float(b[0] * W), float(b[1] * H)
            w, h = float(b[2] * W) - x, float(b[3] * H) - y
            antes = int((SC[k, lado, :VR.CARA_DESDE] >= VR.CONF_PINTA).sum())
            SC[k, lado] = VR._fuera_de_caja(KP[k, lado], SC[k, lado], [x, y, w, h])
            fuera += antes - int((SC[k, lado, :VR.CARA_DESDE] >= VR.CONF_PINTA).sum())

    """`pres` se reconstruye desde `origen`: lo medido es lo que valia como ancla, y sobre eso vuelven
    a correr los dos filtros en el orden de la tuberia."""
    pres = (origen == 0)
    nvac = VR.tira_poses_vacias(KP, SC, pres, mask, T)
    """Y se vuelve a recortar los extremos DESPUES de tirar las poses vacias, porque el recorte de
    `extrae_pareja` corre sobre las cajas y no sabe todavia si de esa caja va a salir una pose. Sin
    esto los bordes se quedan con poses SOSTENIDAS, que es lo que el recorte venia a evitar."""
    mask, nrec = VR.recorta_por_poses(pres, mask)
    val = VR.valida_inventadas(KP, SC, pres, mask, T)
    KP, SC, nuevo, hueco = VR.inventa_poses(KP, SC, pres, mask, T)

    prev = f.with_suffix(".npz.previo")
    if not prev.exists():
        shutil.copy2(f, prev)
    datos = {k: z[k] for k in z.files}
    datos.update({"KP": KP, "SC": SC, "origen": nuevo, "pres": pres, "mask": mask,
                  "hueco": hueco,
                  "filtro_vacias": np.array(nvac, dtype="int32"),
                  "puntos_min": np.array(VR.PUNTOS_MIN, dtype="int32"),
                  "filtro_fuera_de_caja": np.array(fuera, dtype="int32"),
                  "recortados_por_pose": np.array(nrec, dtype="int32")})
    np.savez_compressed(f, **datos)
    ent = int(mask.sum())
    hh = hueco[(nuevo > 0)]
    return {"clip": f.name[6:-4], "fuera": fuera, "vacias": nvac, "recorte": nrec,
            "inventadas_antes": int((origen > 0).sum()), "inventadas_ahora": int((nuevo > 0).sum()),
            "sostenidas": int((nuevo[mask] == 2).sum()),
            "hueco_max": int(hh.max()) if hh.size else 0,
            "sobre_24": int((hh > 24).sum()),
            "sin_pose": int((nuevo[mask] < 0).sum()), "entregados": ent,
            "err_rel": (val["rel_p50"] if val else float("nan"))}


def main():
    fich = sorted(PRUEBAS.glob("poses_*.npz"))
    if not fich:
        raise SystemExit("no hay NPZ de poses")
    print("%-38s %7s %7s %8s %9s %6s %8s" % (
        "clip", "fuera", "vacias", "recorte", "inv ahora", "sost", "sin pose"))
    print("-" * 90)
    filas = []
    for f in fich:
        r = repara(f)
        if r is None:
            print("%-38s se salta: no lleva `origen`" % f.name[:38])
            continue
        filas.append(r)
        print("%-38s %7d %7d %8d %9d %6d %8d" % (
            r["clip"][:38], r["fuera"], r["vacias"], r["recorte"],
            r["inventadas_ahora"], r["sostenidas"], r["sin_pose"]))
    if not filas:
        return
    print()
    print("=" * 84)
    print("TOTAL sobre %d NPZ y %d cuadros entregados" % (
        len(filas), sum(f["entregados"] for f in filas)))
    print("=" * 84)
    print("  keypoints espurios quitados (fuera de su propia caja): %d" % sum(
        f["fuera"] for f in filas))
    print("  poses medidas tiradas por quedarse vacias: %d" % sum(f["vacias"] for f in filas))
    print("  cuadros recortados de los extremos por no tener las dos poses: %d" % sum(
        f["recorte"] for f in filas))
    print("  poses inventadas: %d antes, %d ahora" % (
        sum(f["inventadas_antes"] for f in filas), sum(f["inventadas_ahora"] for f in filas)))
    print("  de ellas SOSTENIDAS, que no se pueden validar: %d" % sum(
        f["sostenidas"] for f in filas))
    print("  cuadros entregados que se quedan SIN pose: %d" % sum(f["sin_pose"] for f in filas))
    er = [f["err_rel"] for f in filas if np.isfinite(f["err_rel"])]
    if er:
        print("  error de inventar, escondiendo poses conocidas: mediana %.1f %% del torso" % (
            100 * float(np.median(er))))
    print()
    print("  HUECOS FUERA DE LA ZONA MEDIDA, que es lo que hay que poder pesar despues:")
    print("    poses inventadas con hueco de mas de 24 cuadros: %d de %d (%.1f %%)" % (
        sum(f["sobre_24"] for f in filas), sum(f["inventadas_ahora"] for f in filas),
        100.0 * sum(f["sobre_24"] for f in filas)
        / max(sum(f["inventadas_ahora"] for f in filas), 1)))
    tot_lados = 2 * sum(f["entregados"] for f in filas)
    print("    sobre TODAS las poses entregadas eso es %d de %d, un %.1f %%" % (
        sum(f["sobre_24"] for f in filas), tot_lados,
        100.0 * sum(f["sobre_24"] for f in filas) / max(tot_lados, 1)))
    print("    hueco mas largo del conjunto: %d cuadros" % max(f["hueco_max"] for f in filas))
    print("    el error de interpolar solo esta medido hasta 24 cuadros (25,7 % del torso); por")
    print("    encima de eso no hay medida y el NPZ lleva `hueco` para poder pesarlo o excluirlo")
    print()
    print("  originales guardados como `poses_*.npz.previo`")


if __name__ == "__main__":
    main()
