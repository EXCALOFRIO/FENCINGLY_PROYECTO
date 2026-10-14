"""Pregunta 3 (14/10): en que se va el tiempo de la etapa 11: `zona_tocado.analiza` (no escribe) y
`angulo_lejano.rasgos` (la fusion), con cProfile, en dos clips cuyas entradas estan en el scratch
`D:/FENCINGLY_SCRATCH/memoria2/trabajo` (FENCINGLY_TRABAJO apunta alli; nada va a las salidas de verdad)."""
import cProfile, io, json, os, pathlib, pstats, sys, time
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo"))
import cv2  # noqa: E402
import rutas as RU, catalogo as CA, zona_tocado as ZT, angulo_lejano as AL  # noqa: E402
assert "memoria2" in str(RU.TRABAJO), RU.TRABAJO
df = CA.manifiesto().set_index("clip_id")
res = {}
for c in ("wanderingKauai‘Ō‘ōOleanderHawkMoth", "comfortableRedSpittingCobraBison"):
    v = RU.video(c)
    out = {}
    for nom, f in (("analiza", lambda: ZT.analiza(c, v, df.loc[c, "torneo"])),):
        pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); r = f(); pr.disable()
        out[nom] = {"s": round(time.perf_counter() - t, 2), "motivo": r.get("motivo"), "via_h": r.get("via_h")}
        st = pstats.Stats(pr)
        for (fn, ln, n), (cc, nc, tt, ct, _) in st.stats.items():
            if n in ("read", "grab"):
                out[nom]["%s_llamadas" % n] = nc; out[nom]["%s_s" % n] = round(tt, 2)
            if n in ("_rasgos", "calibrate", "process") and "zona_tocado" in fn or n == "calibrate":
                out[nom]["%s_acum_s" % n] = round(ct, 2); out[nom]["%s_llamadas" % n] = nc
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
        out[nom]["top_acum"] = [l.strip()[:140] for l in s.getvalue().splitlines() if l.strip() and l.strip()[0].isdigit()][:14]
    zr = {k: (None if r.get(k) is None else str(r.get(k))) for k in ("motivo", "cuadro_usado", "cuadro_tocado")}
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); AL.rasgos(c, zr, v); pr.disable()
    st = pstats.Stats(pr)
    out["angulo"] = {"s": round(time.perf_counter() - t, 2)}
    for (fn, ln, n), (cc, nc, tt, ct, _) in st.stats.items():
        if n in ("read", "grab"):
            out["angulo"]["%s_llamadas" % n] = nc; out["angulo"]["%s_s" % n] = round(tt, 2)
    res[c] = out
    print(c[:30], json.dumps(out, indent=1, ensure_ascii=False), flush=True)
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
