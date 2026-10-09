"""Tira de fotogramas de la banda del rotulo en los cuadros pedidos: para comprobar A OJO un encendido."""
import sys, pathlib, cv2, numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU
def tira(clip, cuadros, salida, y0=0.74, ancho=900):
    cap = cv2.VideoCapture(RU.video(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted({max(0, min(n - 1, c)) for c in cuadros})
    filas, k = {}, 0
    while want and k <= want[-1]:
        ok, img = cap.read()
        if not ok: break
        if k in want:
            H, W = img.shape[:2]
            b = img[int(y0 * H):]
            b = cv2.resize(b, (ancho, int(b.shape[0] * ancho / W)))
            cv2.putText(b, "cuadro %d" % k, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            filas[k] = b
        k += 1
    cap.release()
    img = np.vstack([filas[c] for c in sorted(filas)])
    cv2.imwrite(str(salida), img)
if __name__ == "__main__":
    clip = sys.argv[1]; cuadros = [int(x) for x in sys.argv[2].split(",")]
    tira(clip, cuadros, sys.argv[3])
