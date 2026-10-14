"""Pregunta 3 (14/10): cuanto cuesta decodificar (read) frente a grab, por resolucion, y cuanto cuesta
`extrae_pareja.procesa` sin cuadros (la primera llamada de `pista.pareja` en la etapa 7). Solo mide
funciones en memoria; no escribe ninguna salida."""
import json, pathlib, pickle, sys, time
import cv2
import numpy as np
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import rutas as RU  # noqa: E402
import extrae_pareja as EP  # noqa: E402
idx = RU.indice_videos()
res = {}
for c in ("PortlyVariableGrosbeak", "comfortableRedSpittingCobraBison"):
    v = idx.get(c) or str(RU.DESCARGAS / ("%s.mp4" % c))
    out = {}
    for modo in ("read", "grab", "grab2_read1"):
        cap = cv2.VideoCapture(v); t = time.perf_counter(); n = 0
        while True:
            if modo == "read":
                ok, _ = cap.read()
            elif modo == "grab":
                ok = cap.grab()
            else:
                ok = cap.grab()
                if ok and n % 2 == 0:
                    ok, _ = cap.retrieve()
            if not ok:
                break
            n += 1
        out[modo + "_s"] = round(time.perf_counter() - t, 2); out["n"] = n
        out["W"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
    pk = RU.TRABAJO / ("det_%s.pkl" % c)
    if pk.is_file():
        cajas, cuadros, W, H, T = pickle.loads(pk.read_bytes())
        t = time.perf_counter(); EP.procesa(cajas, cuadros, W, H, T); out["procesa_sin_cuadros_s"] = round(time.perf_counter() - t, 2)
    res[c] = out
    print(c, out, flush=True)
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
