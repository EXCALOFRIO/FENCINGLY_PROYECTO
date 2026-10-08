"""Checks 4 y 5 (lectura SOLO de poses_<clip>.npz, nada se escribe en el proyecto):
  4. tasa de intercambio izquierda/derecha ANATOMICO del tobillo (indices Sapiens crudos 13/14)
     cuadro a cuadro, en frames MEDIDOS (origen==0).
  5. confianza media de los keypoints por resolucion del video (720p/1080p vs 480p/360p).
"""
import os, sys, collections
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU

TRABAJO = RU.TRABAJO
OUT = r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\calidad"

poses_files = sorted(f for f in os.listdir(TRABAJO) if f.startswith("poses_") and f.endswith(".npz"))
print("clips a procesar:", len(poses_files))

THR = 0.3
IDX_A, IDX_B = 13, 14  # tobillo izq/der ANATOMICO crudos de Sapiens (ver zona_tocado._MAPA)

swap_rows = []     # clip, lado, n_transiciones_validas, n_swaps
conf_rows = []      # clip, W, H, res_bucket, media_sc_medida, frac_bajo_umbral
errores = 0

def bucket(h):
    if h >= 1000:
        return "1080p"
    if h >= 700:
        return "720p"
    if h >= 450:
        return "480p"
    if h >= 300:
        return "360p"
    return "otro(%d)" % h

for i, fn in enumerate(poses_files):
    cid = fn[len("poses_"):-4]
    p = TRABAJO / fn
    try:
        z = np.load(p, allow_pickle=True)
        if "KP" not in z.files:
            continue
        KP, SC, origen, pres = z["KP"], z["SC"], z["origen"], z["pres"]
        B = z["B"]
        W, H = int(z["W"]), int(z["H"])
    except Exception:
        errores += 1
        continue

    T = KP.shape[0]
    rb = bucket(H)

    # --- check 5: confianza por resolucion, solo en frames/lado con pose MEDIDA (pres) ---
    for lado in (0, 1):
        m = pres[:, lado]
        if not m.any():
            continue
        sc = SC[m, lado, :]  # (n_frames_medidos, 308)
        conf_rows.append((cid, W, H, rb, lado, float(sc.mean()), float((sc < THR).mean()), int(m.sum())))

    # --- check 4: swap de tobillo anatomico, frame a frame, solo con AMBOS medidos y conf>THR ---
    for lado in (0, 1):
        bh = np.maximum(B[:, lado, 3] - B[:, lado, 1], 1e-6) * H  # alto de caja en pixeles
        okA = (SC[:, lado, IDX_A] > THR) & pres[:, lado] & (origen[:, lado] == 0)
        okB = (SC[:, lado, IDX_B] > THR) & pres[:, lado] & (origen[:, lado] == 0)
        ok = okA & okB
        n_trans = 0
        n_swap = 0
        prevA = prevB = prev_bh = None
        for t in range(T):
            if not ok[t]:
                prevA = prevB = prev_bh = None
                continue
            A, Bp = KP[t, lado, IDX_A, :], KP[t, lado, IDX_B, :]
            if prevA is not None:
                same = np.linalg.norm(A - prevA) + np.linalg.norm(Bp - prevB)
                swap = np.linalg.norm(A - prevB) + np.linalg.norm(Bp - prevA)
                movimiento = same / max(prev_bh, bh[t])
                n_trans += 1
                if swap < same * 0.5 and movimiento > 0.08:
                    n_swap += 1
            prevA, prevB, prev_bh = A, Bp, bh[t]
        if n_trans >= 10:
            swap_rows.append((cid, lado, n_trans, n_swap, n_swap / n_trans))

    if (i + 1) % 1000 == 0:
        print("  ...", i + 1, "clips procesados")

print("errores de lectura:", errores)

conf_df = pd.DataFrame(conf_rows, columns=["clip", "W", "H", "res", "lado", "sc_media", "frac_bajo_umbral", "n_frames"])
conf_df.to_csv(os.path.join(OUT, "conf_por_resolucion.csv"), index=False)
print("\n=== 5. confianza de keypoints por resolucion (solo poses MEDIDAS) ===")
print(conf_df.groupby("res").agg(
    clips_lado=("clip", "count"), sc_media=("sc_media", "mean"),
    frac_bajo_umbral_media=("frac_bajo_umbral", "mean"),
    sc_media_p10=("sc_media", lambda s: s.quantile(0.10)),
).sort_values("sc_media"))

swap_df = pd.DataFrame(swap_rows, columns=["clip", "lado", "n_trans", "n_swap", "tasa"])
swap_df.to_csv(os.path.join(OUT, "swap_tobillo.csv"), index=False)
print("\n=== 4. tasa de swap izq/der ANATOMICO del tobillo, frame a frame (solo medidas) ===")
print("lado-clips evaluados (>=10 transiciones validas):", len(swap_df))
tot_trans = swap_df["n_trans"].sum()
tot_swap = swap_df["n_swap"].sum()
print("transiciones validas totales: %d, swaps detectados: %d, tasa global: %.5f"
      % (tot_trans, tot_swap, tot_swap / tot_trans if tot_trans else float("nan")))
print("\ndistribucion de la tasa por lado-clip:")
print(swap_df["tasa"].describe())
print("\npeores 20 lado-clip (mayor tasa, con >=30 transiciones):")
peor = swap_df[swap_df["n_trans"] >= 30].sort_values("tasa", ascending=False).head(20)
print(peor.to_string(index=False))

# cruce con resolucion para ver si el swap es mas comun en baja resolucion
merge = swap_df.merge(conf_df[["clip", "lado", "res"]], on=["clip", "lado"], how="left")
print("\ntasa de swap por resolucion:")
print(merge.groupby("res").agg(n=("tasa", "count"), tasa_media=("tasa", "mean"),
                                 trans=("n_trans", "sum"), swaps=("n_swap", "sum")))

print("\nFIN chk_poses.py")
