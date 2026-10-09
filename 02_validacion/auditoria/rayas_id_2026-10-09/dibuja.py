import sys, json, pickle, numpy as np, cv2, collections
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import pista_lineas as PL, zona_tocado as ZT, rutas as RU
F = json.load(open("fiabilidad_rayas.json"))
por = collections.defaultdict(list)
for f in F: por[f["clip"]].append(f)
sel = [c for c, L in por.items() if any(f["linea"] == 0 and f["dist"] < 0.5 for f in L) and len(L) >= 3][:4]
for c in sel:
    r = pickle.load(open("D:/FENCINGLY_SCRATCH/agente_tripode/cache/%s.pkl" % c, "rb"))
    t, Hh = r["prod"]["t"], r["prod"]["H"]
    cap = cv2.VideoCapture(RU.video(c))
    for k in range(t): cap.grab()
    ok, img = cap.read(); cap.release()
    Hi = np.linalg.inv(Hh)
    for x in ZT.CFG_PISTA["lines_m"]:
        p = PL.to_px(Hh, np.float32([[x, -0.75], [x, 0.75]]))
        cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (0, 255, 255), 1)
        cv2.putText(img, "%+d" % x, tuple((p[1] + [0, 25]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for f in por[c]:
        cv2.putText(img, "%s z%.0f" % ("D" if f["dashed"] else "-", f["zch"]), tuple(PL.to_px(Hh, np.float32([[f["x"], 0.9]]))[0].astype(int) + [0, 50]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    h = img.shape[0]
    cv2.imwrite("ojo_%s.jpg" % c, cv2.resize(img[int(0.35 * h):, :], None, fx=0.6, fy=0.6))
    print(c, [(f["linea"], f["dashed"], round(f["zch"])) for f in por[c]])
