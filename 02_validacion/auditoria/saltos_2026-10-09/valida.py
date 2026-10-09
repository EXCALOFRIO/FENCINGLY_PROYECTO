import sys, pickle, random, numpy as np, cv2, pathlib
sys.path.insert(0, r"D:\FENCINGLY_SCRATCH\agente_saltos")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, regla as R
D = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_saltos")
filas = pickle.load(open(D / "poblacion.pkl", "rb"))
dev = set(open(D / "S" / "lista.txt", encoding="utf-8").read().split())
cambian = sorted(c for c, n, m0, m1, q in filas if n > 0 and c not in dev)
V = random.Random(20261011).sample(cambian, 30)
sal = D / "V"; sal.mkdir(exist_ok=True)
(sal / "lista.txt").write_text("\n".join(V) + "\n", encoding="utf-8")
print("cambian fuera del desarrollo", len(cambian), "muestra", len(V))
info = {c: (n, m0, m1, q) for c, n, m0, m1, q in filas}
for c in V:
    n, m0, m1, q = info[c]
    with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
        B, pres, mask = z["B"].copy(), z["pres"].copy(), z["mask"].copy()
    B2, p2 = B.copy(), pres.copy()
    R.saltos_pie(B2, p2, mask); m2 = R.recorta(p2, mask)
    ks = []
    lados = sorted({l for l, k in q})
    for l in lados:
        kk = sorted(k for ll, k in q if ll == l)
        sel = [kk[0], kk[len(kk) // 2], kk[-1]]
        pres_l = np.nonzero(mask & pres[:, l])[0]
        antes = [k for k in pres_l if k < kk[0] and k not in kk]
        despues = [k for k in pres_l if k > kk[-1] and k not in kk]
        if antes: sel = [antes[-1]] + sel
        if despues: sel = sel + [despues[0]]
        ks += [(l, k) for k in sel]
    perd = np.nonzero(mask & ~m2)[0]
    perd_ok = [k for k in perd if not any(k == kq for _, kq in q)]
    if len(perd_ok): ks.append((-1, int(perd_ok[len(perd_ok) // 2])))
    need = sorted({k for _, k in ks})
    cap = cv2.VideoCapture(RU.video(c)); ims = {}; k = 0
    while k <= max(need):
        ok, im = cap.read()
        if not ok: break
        if k in need: ims[k] = im
        k += 1
    pan = []
    qset = set(q)
    for l, k in ks:
        if k not in ims: continue
        im = ims[k].copy(); H, W = im.shape[:2]
        for lado, col in ((0, (0, 220, 0)), (1, (0, 0, 255))):
            b = B[k, lado]
            if np.isfinite(b).all():
                x1, y1, x2, y2 = (b * [W, H, W, H]).astype(int)
                quitado = (lado, k) in qset
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255) if quitado else col, 9 if quitado else (5 if lado == l else 2))
        im = cv2.resize(im, (480, int(H * 480 / W)))
        et = ("c%d %s" % (k, "LR"[l])) if l >= 0 else "c%d MASCARA PERDIDA" % k
        if l >= 0 and (l, k) in qset: et += " QUITADO"
        cv2.putText(im, et, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(im, et, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        pan.append(im)
    fil = [np.hstack(pan[i:i + 3] + [np.zeros_like(pan[0])] * (3 - len(pan[i:i + 3]))) for i in range(0, len(pan), 3)]
    cv2.imwrite(str(sal / ("%s.jpg" % c)), np.vstack(fil), [cv2.IMWRITE_JPEG_QUALITY, 72])
    rang = {}
    for l, k in q: rang.setdefault(l, []).append(k)
    print("%-40s quita %3d  mask %d->%d  %s" % (c, n, m0, m1, "  ".join("%s:%d-%d(n%d)" % ("LR"[l], min(v), max(v), len(v)) for l, v in rang.items())), flush=True)
