import sys, glob, pathlib, pickle, numpy as np
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_saltos")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, lotes as LO, regla as R
ids = set(LO.conjunto("anotados"))
filas = []
for f in sorted(glob.glob(str(RU.TRABAJO / "revision_*.npz"))):
    c = pathlib.Path(f).stem[9:]
    if c not in ids: continue
    with np.load(f, allow_pickle=True) as z:
        B, pres, mask = z["B"].copy(), z["pres"].copy(), z["mask"].copy()
    n, q = R.saltos_pie(B, pres, mask)
    m2 = R.recorta(pres, mask)
    filas.append((c, n, int(mask.sum()), int(m2.sum()), q))
pickle.dump(filas, open(r"D:\FENCINGLY_SCRATCH\agente_saltos\poblacion.pkl", "wb"))
ch = [x for x in filas if x[1] > 0]
n1 = [x for x in ch if x[1] == 1]
print("clips", len(filas), "cambian", len(ch), "(%.3f)" % (len(ch) / len(filas)), " de ellos solo 1 cuadro:", len(n1), " >=5 cuadros:", sum(x[1] >= 5 for x in ch))
print("cuadros quitados", sum(x[1] for x in ch), " cuadros de mascara perdidos", sum(x[2] - x[3] for x in ch), "de", sum(x[2] for x in filas))
