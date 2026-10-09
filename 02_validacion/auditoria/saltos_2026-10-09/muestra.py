import sys, pickle, random, pathlib, numpy as np, cv2
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU
D = pathlib.Path(r"D:\FENCINGLY_SCRATCH\agente_saltos")
filas = pickle.load(open(D / "saltos.pkl", "rb"))
info = {c: r for c, r in filas}
S_pob = sorted(c for c, r in filas if r and r["dpie"] > 0.75)
K_pob = sorted(c for c, r in filas if r and r["dpie"] < 0.30)
modo = sys.argv[1]
if modo == "S":
    clips = random.Random(20261009).sample(S_pob, 30)
elif modo == "K":
    clips = random.Random(20261010).sample(K_pob, 10)
else:
    clips = sys.argv[2:]
sal = D / modo
sal.mkdir(exist_ok=True)
(sal / "lista.txt").write_text("\n".join(clips) + "\n", encoding="utf-8")
print(modo, "poblacion", len(S_pob) if modo == "S" else len(K_pob), "muestra", len(clips))

def lee(ruta, ks):
    cap = cv2.VideoCapture(ruta); out = {}; k = 0; obj = set(ks)
    while obj and k <= max(ks):
        ok, im = cap.read()
        if not ok: break
        if k in obj: out[k] = im; obj.discard(k)
        k += 1
    cap.release(); return out

for c in clips:
    r = info[c]
    with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
        B, pres, tipo, T = z["B"], z["pres"], z["tipo"], int(z["T"])
    l, k0, k1 = r["lado"], r["k0"], r["k1"]
    ks = sorted({max(0, k0 - 6), k0, k1, min(T - 1, k1 + 6)})
    ims = lee(RU.video(c), ks)
    H, W = next(iter(ims.values())).shape[:2]
    pan = []
    for k in ks:
        if k not in ims: continue
        im = ims[k].copy()
        for lado, col in ((0, (0, 220, 0)), (1, (0, 0, 255))):
            b = B[k, lado]
            if np.isfinite(b).all():
                x1, y1, x2, y2 = (b * [W, H, W, H]).astype(int)
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 8 if lado == l else 2)
        im = cv2.resize(im, (640, int(H * 640 / W)))
        cv2.putText(im, "c%d %s t%d" % (k, "L" if l == 0 else "R", tipo[k, l]), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        pan.append(im)
    while len(pan) < 4: pan.append(np.zeros_like(pan[0]))
    rej = np.vstack([np.hstack(pan[:2]), np.hstack(pan[2:4])])
    # zoom k0 y k1
    zs = []
    b0, b1 = B[k0, l] * [W, H, W, H], B[k1, l] * [W, H, W, H]
    x1 = min(b0[0], b1[0]); y1 = min(b0[1], b1[1]); x2 = max(b0[2], b1[2]); y2 = max(b0[3], b1[3])
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - 0.4 * w)), int(min(W, x2 + 0.4 * w)); y1, y2 = int(max(0, y1 - 0.2 * h)), int(min(H, y2 + 0.2 * h))
    for k, b in ((k0, b0), (k1, b1)):
        im = ims[k].copy(); bb = b.astype(int)
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 3)
        cr = im[y1:y2, x1:x2]
        s = 640 / max(cr.shape[1], 1)
        cr = cv2.resize(cr, (640, max(1, int(cr.shape[0] * s))))
        if cr.shape[0] > 640: cr = cv2.resize(cr, (int(640 * 640 / cr.shape[0]), 640)); cr = cv2.copyMakeBorder(cr, 0, 0, 0, 640 - cr.shape[1], cv2.BORDER_CONSTANT)
        zs.append(cr)
    hz = max(z.shape[0] for z in zs)
    zs = [cv2.copyMakeBorder(z, 0, hz - z.shape[0], 0, 0, cv2.BORDER_CONSTANT) for z in zs]
    out = np.vstack([rej, np.hstack(zs)])
    cv2.imwrite(str(sal / ("%s.jpg" % c)), out, [cv2.IMWRITE_JPEG_QUALITY, 75])
    print("%-40s lado %d  c%d->c%d hueco %d dpie %.2f dcentro %.2f razon_alto %.2f tipos %d,%d" % (c, l, k0, k1, r["hueco"], r["dpie"], r["dcentro"], r["razon_alto"], r["tipo0"], r["tipo1"]), flush=True)
