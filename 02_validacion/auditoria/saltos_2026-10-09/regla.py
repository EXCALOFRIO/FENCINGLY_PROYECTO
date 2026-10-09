"""La regla candidata, como funcion pura sobre (B, pres, mask). Misma que ira a extrae_pareja._saltos_pie."""
import numpy as np
U, G = 0.75, 8          # salto del pie en altos de caja, hueco maximo
F_ALTO, F_PIE, F_ALTO_SOLO = 1.5, 0.5, 2.0  # incoherente: alto x1,5 o pie a mas de 0,5 altos
VEC = 5                 # vecinos a cada lado para el cuadro suelto

def _incoherente(h, y2, h_ref, y2_ref):
    r = max(h, h_ref) / max(min(h, h_ref), 1e-6)
    return r > F_ALTO_SOLO or (r > F_ALTO and abs(y2 - y2_ref) > F_PIE * h_ref)

def saltos_pie(B, pres, mask):
    """Quita (pres False, B NaN) lo que tras un salto del pie no es coherente con el tramo principal
    del lado, y los cuadros sueltos incoherentes con sus vecinos. Devuelve (n_cuadros, detalle)."""
    quit = []
    for l in (0, 1):
        ks = np.nonzero(mask & pres[:, l] & np.isfinite(B[:, l]).all(1))[0]
        if len(ks) < 3:
            continue
        b = B[ks, l]; h = b[:, 3] - b[:, 1]; y2 = b[:, 3]
        dp = np.abs(np.diff(y2)) / np.maximum(h[:-1], 1e-6)
        dk = np.diff(ks)
        cortes = [i + 1 for i in range(len(dp)) if dk[i] <= G and dp[i] > U]
        if not cortes:
            continue
        segs = np.split(np.arange(len(ks)), cortes)
        p = max(segs, key=len)
        hp, yp = float(np.median(h[p])), float(np.median(y2[p]))
        fuera = set()
        for s in segs:
            if s is p:
                continue
            if _incoherente(float(np.median(h[s])), float(np.median(y2[s])), hp, yp):
                fuera.update(int(i) for i in s)
        # cuadros sueltos: solo los que tocan un corte (a un lado u otro del salto)
        bordes = set()
        for c in cortes:
            bordes.update((c - 1, c))
        for i in bordes:
            if i in fuera:
                continue
            vec = [j for j in range(max(0, i - VEC), min(len(ks), i + VEC + 1)) if j != i and j not in fuera]
            if len(vec) < 3:
                continue
            if _incoherente(float(h[i]), float(y2[i]), float(np.median(h[vec])), float(np.median(y2[vec]))):
                fuera.add(i)
        for i in sorted(fuera):
            pres[ks[i], l] = False
            B[ks[i], l] = np.nan
            quit.append((l, int(ks[i])))
    return len(quit), quit

def recorta(pres, mask):
    ambos = (pres.sum(axis=1) == 2) & mask
    if not ambos.any():
        return np.zeros_like(mask)
    a, b = int(np.argmax(ambos)), int(len(ambos) - 1 - np.argmax(ambos[::-1]))
    m = mask.copy(); m[:a] = False; m[b + 1:] = False
    return m
