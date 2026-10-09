import pickle, sys, numpy as np, cv2
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo"); sys.path.insert(0, ".")
import fit, pista_tripode as PT
c = sys.argv[1]
r = pickle.load(open(r"D:\FENCINGLY_SCRATCH\agente_tripode\cache\%s.pkl" % c, "rb"))
w, h, tq, s = r["W"], r["H"], r["tq"], r["s"]
ev = fit.evidencia_de(r)
res = PT.ajusta(ev, r["pies"][tq], r["alt"][tq], w, h)
P, tipo, linea, P2 = PT.puntos_modelo()
ah, aw = r["ash"]
filas = []
for t in [ev.ts[0], ev.ts[len(ev.ts)//2], tq]:
    e = np.unpackbits(r["aristas"][t])[:ah*aw].reshape(ah, aw)
    img = np.dstack([e*110]*3).astype(np.uint8)
    M = PT.cadena(r["G"], tq, [t])[t]
    Hs = []
    if r["prod"].get("ok"):
        Mp = PT.cadena(r["G"], tq, [r["prod"]["t"]])[r["prod"]["t"]]
        Htq = r["prod"]["H"] @ Mp          # imagen_tq -> pista
        Hs.append((M @ np.linalg.inv(Htq), (0, 255, 0)))
    Hp, d, C = PT.camara(res["p"], w, h)
    Hs.append((M @ Hp, (0, 0, 255)))
    for Hm, col in Hs:
        q = Hm @ P; u, v = q[0]/q[2]*s, q[1]/q[2]*s
        for x_, y_, ti in zip(u, v, tipo):
            if 0 <= x_ < aw and 0 <= y_ < ah:
                cv2.circle(img, (int(x_), int(y_)), 1 if ti < 2 else 2, col, -1)
    cv2.putText(img, "t=%d" % t, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    filas.append(img)
cv2.imwrite(r"D:\FENCINGLY_SCRATCH\agente_tripode\dib_%s.png" % c, np.vstack(filas))
print(c, "prod", r["prod"].get("x"), "trip", res["x"], res["rayas"], "marg", res["margen"])
