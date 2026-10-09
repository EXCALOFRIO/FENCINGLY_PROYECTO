"""tira.py CLIP lado k1 k2 ... -> una tira con esos cuadros, la caja del lado en grueso (de revision_)."""
import sys, numpy as np, cv2
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU
c, l = sys.argv[1], int(sys.argv[2]); ks = [int(x) for x in sys.argv[3:]]
with np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True) as z:
    B = z["B"]
cap = cv2.VideoCapture(RU.video(c)); ims = {}; k = 0
while k <= max(ks):
    ok, im = cap.read()
    if not ok: break
    if k in ks: ims[k] = im
    k += 1
pan = []
for k in ks:
    im = ims[k].copy(); H, W = im.shape[:2]
    for lado, col in ((0, (0, 220, 0)), (1, (0, 0, 255))):
        b = B[k, lado]
        if np.isfinite(b).all():
            x1, y1, x2, y2 = (b * [W, H, W, H]).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 8 if lado == l else 2)
    im = cv2.resize(im, (480, int(H * 480 / W)))
    cv2.putText(im, "c%d" % k, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    pan.append(im)
fil = [np.hstack(pan[i:i + 3] + [np.zeros_like(pan[0])] * (3 - len(pan[i:i + 3]))) for i in range(0, len(pan), 3)]
cv2.imwrite(r"D:\FENCINGLY_SCRATCH\agente_saltos\tira_%s.jpg" % c, np.vstack(fil), [cv2.IMWRITE_JPEG_QUALITY, 75])
