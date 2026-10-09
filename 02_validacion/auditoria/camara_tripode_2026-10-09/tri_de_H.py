import numpy as np, sys
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista_tripode as PT
from scipy.optimize import least_squares

def tripode_de_H(Hip, w, h, pies, alt, x_ref):
    """Camara de tripode que mejor reproduce, DENTRO de la imagen, la homografia imagen->pista Hip."""
    Hp_obj = np.linalg.inv(Hip)
    P, tipo, linea = PT.puntos_modelo(paso=0.25)
    q = Hp_obj @ P
    ok = q[2] > 0
    uo, vo = q[0] / q[2], q[1] / q[2]
    m = ok & (uo > 0) & (uo < w) & (vo > 0) & (vo < h)
    if m.sum() < 8:
        return None
    def res(p):
        Hp, d, C = PT.camara(p, w, h)
        q = Hp @ P[:, m]
        z = np.where(np.abs(q[2]) > 1e-9, q[2], 1e-9)
        return np.r_[q[0] / z - uo[m], q[1] / z - vo[m]]
    best = None
    for D in (8, 12, 18, 25, 35, 50):
        for Z in (1.5, 3, 5, 8, 12):
            x0 = PT.inicial(x_ref, pies, alt, w, h, D=D, Zc=Z)
            if x0 is None:
                continue
            for xc in (0.0, x_ref):
                x0b = x0.copy(); x0b[0] = xc
                try:
                    r = least_squares(res, x0b, loss="soft_l1", f_scale=5.0, max_nfev=3000)
                except Exception:
                    continue
                c = float(np.sqrt(np.mean(r.fun ** 2)))
                if best is None or c < best[0]:
                    best = (c, r.x, int(m.sum()))
    return best
