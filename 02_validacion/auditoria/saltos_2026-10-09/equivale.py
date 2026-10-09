import sys, os, numpy as np, pickle
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_saltos")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, regla as R, extrae_pareja as EP
clips = open("S/lista.txt").read().split() + open("V/lista.txt").read().split() + open("K/lista.txt").read().split()
dif = 0
for c in clips:
    with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
        B, pres, mask = z["B"], z["pres"], z["mask"]
    B1, p1 = B.copy(), pres.copy(); n1, _ = R.saltos_pie(B1, p1, mask)
    B2, p2 = B.copy(), pres.copy(); n2 = EP._saltos_pie(B2, p2, mask)
    if n1 != n2 or not np.array_equal(p1, p2) or not np.array_equal(np.isnan(B1), np.isnan(B2)):
        dif += 1; print("DIFIERE", c, n1, n2)
print("clips", len(clips), "difieren", dif, "SALTO_PIE por defecto:", EP.SALTO_PIE)
# humo: procesa desde det_ sin yolo, con la puerta encendida
EP.SALTO_PIE = True
for c in ["CleanGraveFlyinglemur", "PassionateEasygoingCrocodile"]:
    cajas, cuadros, W, H, T = pickle.load(open(RU.TRABAJO / ("det_%s.pkl" % c), "rb"))
    r = EP.procesa(cajas, cuadros, W, H, T)
    print(c, "saltos_pie", r["saltos_pie"], "mask", int(r["mask"].sum()))
EP.SALTO_PIE = False
for c in ["CleanGraveFlyinglemur"]:
    cajas, cuadros, W, H, T = pickle.load(open(RU.TRABAJO / ("det_%s.pkl" % c), "rb"))
    r = EP.procesa(cajas, cuadros, W, H, T)
    print(c, "apagada: saltos_pie", r["saltos_pie"], "mask", int(r["mask"].sum()))
