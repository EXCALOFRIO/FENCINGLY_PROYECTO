import pickle, sys, numpy as np, cv2
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, ".")
import fit, pista_tripode as PT
from scipy.optimize import least_squares

def tripode_de_H(Hip, w, h, inits=()):
    """Ajusta los 7 parametros para reproducir la homografia imagen->pista Hip."""
    Hp_obj = np.linalg.inv(Hip)
    P, tipo, linea = PT.puntos_modelo()
    q = Hp_obj @ P; uo, vo = q[0]/q[2], q[1]/q[2]
    m = (uo > -0.2*w) & (uo < 1.2*w) & (vo > -0.2*h) & (vo < 1.2*h)
    def res(p):
        Hp, d, C = PT.camara(p, w, h)
        q = Hp @ P[:, m]; u, v = q[0]/q[2], q[1]/q[2]
        return np.r_[u - uo[m], v - vo[m]]
    best = None
    for x0 in inits:
                try:
                    r = least_squares(res, x0, method="lm", max_nfev=2000)
                except Exception:
                    continue
                c = float(np.mean(r.fun**2))
                if best is None or c < best[0]:
                    best = (c, r.x)
    return best

c = sys.argv[1]
with open(r"D:\FENCINGLY_SCRATCH\agente_tripode\cache\%s.pkl" % c, "rb") as fh:
    r = pickle.load(fh)
w, h = r["W"], r["H"]; tq = r["tq"]
ev = fit.evidencia_de(r)
P, tipo, linea = PT.puntos_modelo()
prod = r["prod"]
M = PT.cadena(r["G"], tq, [prod["t"]])
Htq = prod["H"] @ M[prod["t"]]
inits = []
for D in (8, 12, 18, 25, 35):
    for Z in (2, 4, 7):
        q0 = PT.inicial(prod["x"], r["pies"][tq], r["alt"][tq], w, h, D=D, Zc=Z)
        if q0 is not None: inits.append(q0)
b = tripode_de_H(Htq, w, h, inits)
print("prod t", prod["t"], "tq", tq, "x", round(prod["x"], 2), "ajuste tripode a H prod: rms px", round(np.sqrt(b[0]), 2), "p", np.round(b[1], 3), "f", round(np.exp(b[1][6])))
# llevar la H de prod (del cuadro prod t) a tq: H_tq = H_t @ M(tq->t)
M = PT.cadena(r["G"], tq, [prod["t"]])
print("cadena a prod t", prod["t"] in M)
pp = b[1]
o_prod = PT.objetivo(pp, ev, P, tipo, r["pies"][tq], r["alt"][tq], w, h)
rec, n = PT.puntua(pp, ev, P, tipo, w, h)
pen, xs = PT.prior(pp, r["pies"][tq], r["alt"][tq], w, h)
print("objetivo de la camara de prod:", round(o_prod/len(ev.ts), 2), "rec/cuadro", round(rec/len(ev.ts), 2), "pen", round(pen, 2), "xs", xs)
res = PT.ajusta(ev, r["pies"][tq], r["alt"][tq], w, h)
print("optimo encontrado: obj", round(res["obj"], 2), "rec", round(res["rec"], 2), "x", round(res["x"], 2), "p", np.round(res["p"], 3), "f", round(np.exp(res["p"][6])))
pen2, xs2 = PT.prior(res["p"], r["pies"][tq], r["alt"][tq], w, h)
print("   pen", round(pen2, 2), "xs", xs2)
# dibujo: aristas del cuadro tq + modelo prod (verde) + tripode (rojo)
ah, aw = r["ash"]
e = np.unpackbits(r["aristas"][tq])[:ah*aw].reshape(ah, aw)
img = np.dstack([e*120]*3).astype(np.uint8)
for p_, col in ((pp, (0,255,0)), (res["p"], (0,0,255))):
    Hp, d, C = PT.camara(p_, w, h)
    q = Hp @ P; u, v = q[0]/q[2]*r["s"], q[1]/q[2]*r["s"]
    for x_, y_ in zip(u, v):
        if 0 <= x_ < aw and 0 <= y_ < ah:
            cv2.circle(img, (int(x_), int(y_)), 1, col, -1)
for s_ in (0, 1):
    f_ = r["pies"][tq][s_]
    if np.isfinite(f_).all():
        cv2.circle(img, (int(f_[0]*r["s"]), int(f_[1]*r["s"])), 4, (255, 255, 0), -1)
cv2.imwrite(r"D:\FENCINGLY_SCRATCH\agente_tripode\diag_%s.png" % c, img)
