"""Pregunta 3 (14/10): en que se va el tiempo de la etapa 9 (`luces.lee_clip`, que no escribe nada) en dos
clips, con cProfile: cuantas veces se abre y se decodifica el video y cuanto cuesta."""
import cProfile, io, json, os, pathlib, pstats, sys, time
os.environ.setdefault("FENCINGLY_LUCES_V2", "1")
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import cv2  # noqa: E402
import rutas as RU, luces as LU, catalogo as CA  # noqa: E402
df = CA.manifiesto().set_index("clip_id")
idx = RU.indice_videos()
res = {}
abre = {"n": 0}
_VC = cv2.VideoCapture
class Cuenta(_VC):
    def __init__(self, *a, **k):
        abre["n"] += 1
        super().__init__(*a, **k)
cv2.VideoCapture = Cuenta
for c in ("PortlyVariableGrosbeak", "comfortableRedSpittingCobraBison"):
    v = idx.get(c) or str(RU.DESCARGAS / ("%s.mp4" % c))
    abre["n"] = 0
    pr = cProfile.Profile(); t = time.perf_counter()
    pr.enable(); filas, via, fps = LU.lee_clip(v, RU.roi_sede(df.loc[c, "torneo"])); pr.disable()
    tot = time.perf_counter() - t
    st = pstats.Stats(pr)
    d = {"s": round(tot, 2), "via": via, "aperturas_VideoCapture": abre["n"]}
    for (fn, ln, nom), (cc, nc, tt, ct, _) in st.stats.items():
        if nom in ("read", "grab", "retrieve") or "resize" in nom or "cvtColor" in nom:
            d["%s_%s" % (nom, os.path.basename(fn))] = {"llamadas": nc, "s": round(tt, 2)}
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
    d["top"] = [l.strip()[:150] for l in s.getvalue().splitlines() if l.strip() and l.strip()[0].isdigit()][:8]
    res[c] = d
    print(c, json.dumps(d, indent=1, ensure_ascii=False), flush=True)
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
