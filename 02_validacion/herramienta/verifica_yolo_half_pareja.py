"""(2026-10-08) Ver extrae_pareja.YOLO_HALF.   python verifica_yolo_half_pareja.py 20 7
Puerta 2 (declarada antes de medir): la PAREJA elegida por EP.procesa con detecciones fp16-lote8
coincide con la de fp32: >=99,5 % de los cuadros con caja en los dos tienen IoU>=0,95 por lado,
y la presencia (pres) coincide en >=99,5 % de los cuadros."""
import sys, random
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU, catalogo as CA, extrae_pareja as EP
import cv2, numpy as np
from ultralytics import YOLO
y = YOLO(RU.pesos_yolo()); y.to("cuda")
ids = list(CA.manifiesto().query("weapon=='foil'").clip_id.astype(str))
random.seed(int(sys.argv[2])); random.shuffle(ids)
clips = [c for c in ids if RU.video(c)][:int(sys.argv[1])]
def det(fr, half, lote):
    cajas, cuadros = [], []
    for i in range(0, len(fr), lote):
        rs = y.predict(fr[i:i+lote], imgsz=EP.IMGSZ_YOLO, conf=EP.CONF_YOLO, verbose=False, device=0, half=half)
        for k, r in enumerate(rs):
            if r.boxes is not None:
                for b in r.boxes.xyxy.cpu().numpy(): cajas.append(b); cuadros.append(i+k)
    return np.asarray(cajas, "float32").reshape(-1, 4), np.asarray(cuadros, "int32")
def iou(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    i = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    ar = lambda z: (z[..., 2]-z[..., 0])*(z[..., 3]-z[..., 1])
    return i/(ar(a)+ar(b)-i+1e-12)
tot = okb = nb = okp = npres = 0; peores = []
for c in clips:
    cap = cv2.VideoCapture(RU.video(c)); W = int(cap.get(3)); H = int(cap.get(4)); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        fr.append(im)
    gris = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype("float32") for f in fr]
    R = {}
    for k, (h, l) in {"a": (False, 1), "b": (True, 8)}.items():
        cj, cu = det(fr, h, l)
        R[k] = EP.procesa(cj, cu, W, H, len(fr), yolo=y, frames=fr, gris=gris)
    A, B = R["a"]["B"], R["b"]["B"]
    v = np.isfinite(A).all(-1) & np.isfinite(B).all(-1)
    u = iou(A, B)[v] >= 0.95
    pa, pb = np.asarray(R["a"]["pres"]), np.asarray(R["b"]["pres"])
    okb += u.sum(); nb += u.size; okp += (pa == pb).sum(); npres += pa.size
    peores.append((u.mean() if u.size else 1.0, c, R["a"]["motivo"], R["b"]["motivo"]))
    print("%-40s cajas IoU>=.95 %.4f  pres igual %.4f  motivo %s / %s" % (c[:40], u.mean() if u.size else 1, (pa == pb).mean(), R["a"]["motivo"], R["b"]["motivo"]), flush=True)
print("TOTAL clips %d  cajas pareja IoU>=0.95 %.4f (n=%d)  pres igual %.4f" % (len(clips), okb/nb, nb, okp/npres))
