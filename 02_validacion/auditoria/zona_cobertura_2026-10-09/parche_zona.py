"""Aplica a `zona_tocado.py` el respaldo N (franja del rotulo mas abajo) tras FENCINGLY_ZONA_RESPALDO.

    python parche_zona.py DEFECTO BANDA          p. ej.  python parche_zona.py 0 0.84
Conserva los finales de linea CRLF del fichero. Falla si algun trozo no se encuentra una sola vez.
"""
import pathlib
import sys

DEFECTO, BANDA = sys.argv[1], sys.argv[2]
p = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo\zona_tocado.py")
raw = p.read_bytes()
crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")

rep = []
rep.append(('''import csv
import pathlib
import sys
import time
''', '''import csv
import os
import pathlib
import sys
import time
'''))
rep.append(('''BANDA_ROTULO = 0.78
"""Fraccion de la altura a partir de la cual esta el rotulo del marcador: se excluye de las rayas."""
''', '''BANDA_ROTULO = 0.78
"""Fraccion de la altura a partir de la cual esta el rotulo del marcador: se excluye de las rayas."""
RESPALDO = os.environ.get("FENCINGLY_ZONA_RESPALDO", "%s") == "1"
BANDA_RESPALDO = %s
"""EL RESPALDO DEL ROTULO (`ZONA_COBERTURA_2026-10-09.md`, candidata N). Medido el 2026-10-09: de los
clips que no contestan por "sin homografia confiable", casi la mitad no ve NUNCA dos rayas
transversales en un mismo cuadro, y a ojo la causa es que `observe` no encuentra el borde CERCANO de
la pista: queda justo debajo de `BANDA_ROTULO` o tapado por el rotulo FIE, que empieza hacia 0,84 de
la altura. Sin los dos bordes no se buscan rayas, aunque se vean. El respaldo SOLO actua donde la
regla de siempre no contesta: rehace los rasgos con la franja en `BANDA_RESPALDO` y aplica la misma
regla (calibracion hasta el tocado + 3, confianza >= 0,35, +-0,5 s). Un clip que ya contestaba sale
identico por construccion. La salida lo dice en `via_h` (`estricta` / `rotulo_<b>`). La cifra y la
decision (que es la que pone el defecto de arriba) estan en el informe."""
''' % (DEFECTO, BANDA)))
rep.append(('''COLUMNAS = ["clip", "motivo", "cuadro_tocado", "via_luz", "cuadro_usado", "x_izq_m", "x_der_m",
            "x_tocado_m", "zona", "conf_h", "metodo_h", "frac_cuadros_con_h", "s"]''',
'''COLUMNAS = ["clip", "motivo", "cuadro_tocado", "via_luz", "cuadro_usado", "x_izq_m", "x_der_m",
            "x_tocado_m", "zona", "conf_h", "metodo_h", "frac_cuadros_con_h", "s", "via_h"]'''))
rep.append(('''    k26 = a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    cap = cv2.VideoCapture(video)
    pf = PL.PisteFeatures(CFG_VIDEO, (W, H))
    for t in range(min(T, tf + 3)):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[t]).all(1)
        pf.process(img, Bpx[t][okb].astype(np.float32), BANDA_ROTULO * H, k26[t][okb])
    cap.release()
    Tn = len(pf.feats)
''', '''    k26 = a_halpe(KP, SC)
    Bpx = B * np.array([W, H, W, H])
    pf = _rasgos(video, W, H, T, tf, Bpx, k26, BANDA_ROTULO)
    Tn = len(pf.feats)
'''))
rep.append(('''    cal = PL.calibrate(pf.feats, pies, alt, CFG_PISTA, (W, H))
    Hs, conf = cal["H"], cal["conf"]
    r["frac_cuadros_con_h"] = round(float((conf >= CFG_PISTA["min_conf"]).mean()), 3) if Tn else 0.0
    tq = min(tf, Tn - 1)
    w = int(VENTANA_S * fps)
    cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
            if np.isfinite(Hs[t]).all() and conf[t] >= CFG_PISTA["min_conf"]]
    if not cand:
''', '''    cal = PL.calibrate(pf.feats, pies, alt, CFG_PISTA, (W, H))
    Hs, conf = cal["H"], cal["conf"]
    r["frac_cuadros_con_h"] = round(float((conf >= CFG_PISTA["min_conf"]).mean()), 3) if Tn else 0.0
    tq = min(tf, Tn - 1)
    w = int(VENTANA_S * fps)
    cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
            if np.isfinite(Hs[t]).all() and conf[t] >= CFG_PISTA["min_conf"]]
    r["via_h"] = "estricta" if cand else None
    if not cand and RESPALDO:
        pf2 = _rasgos(video, W, H, T, tf, Bpx, k26, BANDA_RESPALDO)
        if len(pf2.feats) == Tn:
            cal2 = PL.calibrate(pf2.feats, pies, alt, CFG_PISTA, (W, H))
            cand = [t for t in range(max(0, tq - w), min(Tn, tq + 2))
                    if np.isfinite(cal2["H"][t]).all() and cal2["conf"][t] >= CFG_PISTA["min_conf"]]
            if cand:
                cal = cal2
                Hs, conf = cal["H"], cal["conf"]
                r["via_h"] = "rotulo_%.2f" % BANDA_RESPALDO
    if not cand:
'''))
rep.append(('''def escribe(r):''', '''def _rasgos(video, W, H, T, tf, Bpx, k26, banda):
    """Los rasgos de `pista_lineas` hasta el tocado + 3, con la franja del rotulo en `banda`."""
    cap = cv2.VideoCapture(video)
    pf = PL.PisteFeatures(CFG_VIDEO, (W, H))
    for t in range(min(T, tf + 3)):
        ok, img = cap.read()
        if not ok:
            break
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        okb = np.isfinite(Bpx[t]).all(1)
        pf.process(img, Bpx[t][okb].astype(np.float32), banda * H, k26[t][okb])
    cap.release()
    return pf


def escribe(r):'''))
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
out = s.replace("\n", "\r\n") if crlf else s
p.write_bytes(out.encode("utf-8"))
print("parcheado, CRLF" if crlf else "parcheado, LF")
