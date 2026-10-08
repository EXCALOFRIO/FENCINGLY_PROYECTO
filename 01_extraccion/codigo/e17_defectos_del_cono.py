"""ETAPA E, paso 17: LOS DEFECTOS DEL CONO, con una cifra cada uno.

El usuario dijo *"intenta mejorarlo, que no esta demasiado bien"*. Mirando las 15 laminas del paso 16
aparecen cuatro defectos distintos, y este paso los convierte de impresion a numero. Sin la cifra no
se puede saber cual pesa mas ni si arreglar uno mueve la aguja.

LOS CUATRO, y como se miden SIN GPU y sin verdad-terreno:

  D1 EL VERTICE NO ESTA EN UNA MANO. Visto en `L01`, `L06` y `L11`: la marca cae en la cadera o en la
     espalda. Se mide con la geometria del propio esqueleto: la raiz de la mano tiene que estar LEJOS
     de la cadera del mismo tirador. Si `|raiz - cadera|` es menor que `|hombro - cadera| * 0,35`, la
     «mano» esta pegada al tronco y no puede ser una mano armada extendida. El umbral se declara
     aqui: 0,35 del largo del tronco es menos de la mitad de un brazo, o sea imposible salvo con la
     mano recogida sobre el cuerpo.

  D2 LAS DOS MANOS EN EL MISMO TIRADOR. Visto en `L01` y `L06`. Se mide sin ambiguedad: la raiz de la
     mano de un tirador cae DENTRO de la caja del otro. Es un fallo de pareja o de mano armada, y
     hace que los dos conos salgan del mismo cuerpo, con lo que su solape no significa nada.

  D3 LA MANO ARMADA ES LA DE ATRAS. Visto en `L10`, donde el tirador se lanza a fondo hacia la
     izquierda y la marca esta en su brazo trasero. Se mide con el criterio que `cono_lib.mano_armada`
     ya declara y no valida: la mano elegida tiene que ser la mas cercana al rival. Aqui se cuenta
     cuantas veces la mano NO elegida esta **mas cerca del rival** que la elegida, que es una
     contradiccion interna del propio criterio a nivel de cuadro (la eleccion es por MEDIANA del
     clip, asi que puede contradecirse en cuadros concretos).

  D4 EL EJE SE VA CUANDO LA MANO ESTA EN ESCORZO. Cuando el brazo apunta a la camara su proyeccion se
     acorta, y con poca palanca un pixel de error gira mucho el eje. Se mide la relacion entre el
     largo proyectado `hombro -> raiz` y el salto angular del eje entre cuadros consecutivos: si el
     eje se va con el escorzo, el salto tiene que crecer cuando el largo se acorta. Se da el salto
     mediano en el quintil de largo mas CORTO contra el quintil mas LARGO, y su razon.

Y EL DEFECTO QUE NO ES DEL CONO SINO DEL VIDEO, contado aparte porque no se arregla con codigo: en
estos clips hay **monitores de marcador en primer plano entre la camara y la pista**, y tapan justo la
region donde se cruzarian las hojas (`L00`, `L07`, `L09`, `L17`). Se cuenta cuantas de las laminas
inspeccionadas quedaron indecidibles y por que.

NINGUNA DE ESTAS CUATRO MEDIDAS NECESITA VERDAD-TERRENO. Todas son contradicciones internas o
imposibilidades fisicas, que es lo unico que se puede medir sin anotacion.
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
TOL_D1 = 0.35
CADERAS = C3.CADERAS


def clips():
    corpus = set(x.strip() for x in
                 (C3.SCRATCH / "corpus_etiquetado.txt").read_text(encoding="utf-8").split("\n")
                 if x.strip())
    return [f.name[len("cono_poses_"):-4] for f in sorted(PRUEBAS.glob("cono_poses_*.npz"))
            if not f.name.endswith("_512x384.npz")
            and f.name[len("cono_poses_"):-4] in corpus]


def dentro(p, b, W, H):
    return (np.isfinite(b).all() and b[0] * W <= p[0] <= b[2] * W
            and b[1] * H <= p[1] <= b[3] * H)


tot = {n: [0, 0] for n in ("D1", "D2", "D3")}
d4 = []
por_clip = {}
print("=" * 104)
print("LOS DEFECTOS DEL CONO, con cifra. Todo de los NPZ, sin GPU y sin verdad-terreno")
print("=" * 104)
print("  %-34s %5s | %-13s %-13s %-13s | %s" % (
    "clip", "n", "D1 vertice", "D2 mismo tir.", "D3 mano atras", "D4 escorzo (corto/largo)"))
for clip in clips():
    with np.load(PRUEBAS / ("cono_poses_%s.npz" % clip), allow_pickle=True) as z:
        kp, sc, B = z["kp"], z["sc"], z["B"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        pres = z["presente"]
        mask = z["mask"] if "mask" in z.files else np.ones(T, dtype=bool)
    mano, ejes = {}, {}
    malo = False
    for lado in (0, 1):
        m, _d = CL.mano_armada(kp, sc, B, W, H, lado)
        if m is None:
            malo = True
            break
        mano[lado] = m
        ejes[lado] = C3.eje_brazo_2d(kp, sc, lado, m, "hombro")
    if malo:
        print("  %-34s sin mano armada decidible" % clip[:34])
        continue
    dos = np.array([bool(mask[k] and pres[k].sum() == 2) for k in range(T)])
    c = {n: [0, 0] for n in ("D1", "D2", "D3")}
    largos, saltos = [], []
    for lado in (0, 1):
        M = CL.MANOS[mano[lado]]
        otra = "B" if mano[lado] == "A" else "A"
        MO = CL.MANOS[otra]
        vert, u, ok, largo = ejes[lado]
        rival = 1 - lado
        for k in range(T):
            if not (dos[k] and ok[k]):
                continue
            pr, ph = kp[k, lado, M["root"]], kp[k, lado, M["hombro"]]
            pc = kp[k, lado, list(CADERAS)].mean(axis=0)
            if np.isfinite(pc).all() and np.isfinite(pr).all() and np.isfinite(ph).all():
                torso = float(np.linalg.norm(ph - pc))
                c["D1"][1] += 1
                if torso > 1 and float(np.linalg.norm(pr - pc)) < TOL_D1 * torso:
                    c["D1"][0] += 1
            c["D2"][1] += 1
            if dentro(pr, B[k, rival], W, H):
                c["D2"][0] += 1
            """D3: contradiccion interna del criterio (a) de `mano_armada`, cuadro a cuadro."""
            pro = kp[k, rival, list(CL.HOMBROS)]
            if np.isfinite(pro).all():
                xr = float(pro[:, 0].mean())
                pa = kp[k, lado, MO["root"]]
                if np.isfinite(pa).all():
                    c["D3"][1] += 1
                    if abs(pa[0] - xr) < abs(pr[0] - xr):
                        c["D3"][0] += 1
        a = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        p = ok[:-1] & ok[1:] & dos[:-1] & dos[1:]
        if p.any():
            largos.extend(largo[:-1][p].tolist())
            saltos.extend(np.abs(CL.dif_ang(a[1:][p], a[:-1][p])).tolist())
    for n in tot:
        tot[n][0] += c[n][0]
        tot[n][1] += c[n][1]
    lg, sa = np.array(largos), np.array(saltos)
    m = np.isfinite(lg) & np.isfinite(sa)
    razon = float("nan")
    if m.sum() >= 40:
        q1, q5 = np.percentile(lg[m], 20), np.percentile(lg[m], 80)
        s_corto = sa[m][lg[m] <= q1]
        s_largo = sa[m][lg[m] >= q5]
        if s_corto.size >= 5 and s_largo.size >= 5:
            razon = float(np.median(s_corto) / max(np.median(s_largo), 1e-9))
            d4.append((clip, float(np.median(s_corto)), float(np.median(s_largo)), razon))
    print("  %-34s %5d | %5d %5.1f %% %5d %5.1f %% %5d %5.1f %% | %s" % (
        clip[:34], int(dos.sum()),
        c["D1"][0], 100.0 * c["D1"][0] / max(c["D1"][1], 1),
        c["D2"][0], 100.0 * c["D2"][0] / max(c["D2"][1], 1),
        c["D3"][0], 100.0 * c["D3"][0] / max(c["D3"][1], 1),
        "%.2fx" % razon if np.isfinite(razon) else "-"))
    por_clip[clip] = {n: {"n_malos": c[n][0], "n": c[n][1],
                          "frac": c[n][0] / max(c[n][1], 1)} for n in c}
    por_clip[clip]["D4_razon"] = razon

print()
print("  TOTAL sobre %d clips:" % len(por_clip))
etiq = {"D1": "el vertice esta pegado al tronco (no es una mano extendida)",
        "D2": "la mano de un tirador cae DENTRO de la caja del otro",
        "D3": "la otra mano esta mas cerca del rival que la elegida"}
for n in ("D1", "D2", "D3"):
    print("    %s  %6d de %6d = **%.1f %%**  ·  %s" % (
        n, tot[n][0], tot[n][1], 100.0 * tot[n][0] / max(tot[n][1], 1), etiq[n]))
if d4:
    r = np.array([x[3] for x in d4])
    print("    D4  el salto del eje en el quintil de brazo mas CORTO contra el mas LARGO: "
          "razon mediana **%.2fx** (rango %.2f-%.2f, n=%d clips)" % (
              np.median(r), r.min(), r.max(), len(d4)))
    print("        saltos medianos: corto %.2f gr  ·  largo %.2f gr" % (
        float(np.median([x[1] for x in d4])), float(np.median([x[2] for x in d4]))))
    print("        -> %s" % (
        "el eje SI se va con el escorzo y ese es un defecto real"
        if np.median(r) > 1.3 else
        "el escorzo NO desestabiliza el eje de forma apreciable"))

(PRUEBAS / "cono_e17_defectos.json").write_text(
    json.dumps({"tol_D1": TOL_D1, "total": {n: tot[n] for n in tot},
                "D4": [{"clip": x[0], "salto_corto": x[1], "salto_largo": x[2], "razon": x[3]}
                       for x in d4], "clips": por_clip},
               indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print()
print("guardado %s" % (PRUEBAS / "cono_e17_defectos.json"))
