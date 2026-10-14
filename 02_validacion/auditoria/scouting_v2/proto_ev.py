"""PROTOTIPO (paso 5 del scouting v2): la logica de eventos de la etapa 7 (`posicion.eventos`) alimentada
con las posiciones de la homografia POR CUADRO de la etapa 11 (`pista_h_`) en vez de la homografia unica
de la 7 + estabilizacion. Mide, en clips donde la 7 SI tiene homografia, cuanto se parecen los eventos de
las dos versiones, y en clips donde la 7 NO la tiene, cuantos eventos en metros salen y si son fisicos.
Solo lee: TRABAJO del scratch (entradas enlazadas), no escribe en ningun sitio de verdad."""
import os, sys, json, pathlib, random, csv, time
os.environ["FENCINGLY_TRABAJO"] = r"C:\Users\Alejandro\FENCINGLY_TMP\scout_h\trabajo"
os.environ["FENCINGLY_PISTA"] = r"C:\Users\Alejandro\FENCINGLY_TMP\scout_h\pista"
RAIZ = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO")
sys.path.insert(0, str(RAIZ / "01_extraccion" / "codigo")); sys.path.insert(0, str(RAIZ / "02_validacion" / "herramienta"))
import numpy as np  # noqa: E402
import rutas as RU, pista as PI, posicion as PO, pista_lineas as PL  # noqa: E402
REAL_P = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\pista")
T_ = RU.TRABAJO


def apoyos_v2(res, poses, ph):
    """Como `posicion.apoyos`, con la H de cada cuadro (`pista_h_`): sin `Mk`, porque cada H ya lleva los
    pixeles de SU cuadro a metros."""
    r = res["_r"]
    B, pres, mask, W, H, T = r["B"], r["pres"], r["mask"], r["W"], r["H"], r["T"]
    z = np.load(ph, allow_pickle=True)
    Hs, conf, mc = z["H"], z["conf"], float(z["min_conf"])
    C = np.full((T, 2, 2), np.nan); P1 = np.full((T, 2, 2), np.nan); P2 = np.full((T, 2, 2), np.nan)
    org = np.full((T, 2), -1, "int8")
    KP, SC = (poses["KP"], poses["SC"]) if poses is not None and poses["KP"].shape[0] >= T else (None, None)
    for k in np.nonzero(mask)[0]:
        if k >= len(Hs) or not (np.isfinite(Hs[k]).all() and conf[k] >= mc):
            continue
        for lado in (0, 1):
            if not pres[k, lado]:
                continue
            pies = PO._pie_pose(KP, SC, k, lado) if KP is not None else [None, None]
            if pies[0] is not None and pies[1] is not None:
                q = [PL.to_m(Hs[k], np.asarray(p, np.float32)[None])[0].astype(float) for p in pies]
                P1[k, lado], P2[k, lado] = q[0], q[1]
                C[k, lado] = 0.5 * (q[0] + q[1]); org[k, lado] = 0
                continue
            b = B[k, lado]
            if not np.isfinite(b).all():
                continue
            p = np.array([[0.5 * (b[0] + b[2]) * W, b[3] * H]], np.float32)
            C[k, lado] = PL.to_m(Hs[k], p)[0]; org[k, lado] = 1
    x0, x1 = np.nanmedian(C[:, 0, 0]), np.nanmedian(C[:, 1, 0])
    if np.isfinite(x0) and np.isfinite(x1) and x0 > x1:
        for M in (C, P1, P2):
            M[:, :, 0] *= -1.0
    dpies = np.abs(P1[:, :, 0] - P2[:, :, 0])
    ok = mask & np.isfinite(C[:, 0, 0]) & np.isfinite(C[:, 1, 0])
    return {"C": C, "P1": P1, "P2": P2, "origen": org, "dpies": dpies, "dpies_2d": np.linalg.norm(P1 - P2, axis=2),
            "dist": np.abs(C[:, 0, 0] - C[:, 1, 0]), "dist_2d": np.linalg.norm(C[:, 0] - C[:, 1], axis=1),
            "cruces": int((C[ok, 0, 0] > C[ok, 1, 0]).sum()), "n_con_los_dos": int(ok.sum()), "convenio_eje": "v2"}


def eventos_de(clip, version):
    res = PI.analiza(clip)
    poses = PO.carga_poses(clip)
    fps = PO._fps(PI.video_clip(clip))
    if version == "7":
        if not res.get("ok"):
            return None, res
        ap = PO.apoyos(res, poses)
    else:
        if "_r" not in res or "_Mk" not in res:
            return None, res
        ap = apoyos_v2(res, poses, T_ / ("pista_h_%s.npz" % clip))
    ev = PO.eventos(ap, res["_r"], fps, poses, res=res)
    return ev, res


MAG = {}


def compara(e7, e2):
    out = {}
    for t in ("marcha", "romper", "fondo", "alto", "brazo"):
        a = [(e["lado"], int(e["inicio"]), e.get("magnitud_m")) for e in e7 if e["tipo"] == t]
        b = [(e["lado"], int(e["inicio"]), e.get("magnitud_m")) for e in e2 if e["tipo"] == t]
        ok_a = sum(1 for l, k, _m in a if any(l == l2 and abs(k - k2) <= 3 for l2, k2, _ in b))
        ok_b = sum(1 for l, k, _m in b if any(l == l2 and abs(k - k2) <= 3 for l2, k2, _ in a))
        for l, k, m in a:
            par = [m2 for l2, k2, m2 in b if l == l2 and abs(k - k2) <= 3]
            if par and m is not None and par[0] is not None and np.isfinite(m) and np.isfinite(par[0]):
                MAG.setdefault(t, []).append((float(m), float(par[0])))
        out[t] = (len(a), len(b), ok_a, ok_b)
    return out


if __name__ == "__main__":
    orden = open(r"D:\FENCINGLY_SCRATCH\scout\clips_h.txt", encoding="utf-8").read().split()
    hechos = [c for c in orden if (T_ / ("pista_h_%s.npz" % c)).is_file()]
    con7, sin7 = [], []
    for c in hechos:
        f = REAL_P / ("posicion_clip_%s.csv" % c)
        if not f.is_file():
            continue
        sh = list(csv.DictReader(open(f, encoding="utf-8")))[0]["sin_homografia"] == "True"
        (sin7 if sh else con7).append(c)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("clips con pista_h_: %d (la 7 con H %d, sin H %d); se prueban %d y %d" % (len(hechos), len(con7), len(sin7), N, N), flush=True)
    claves = None
    print(json.dumps(PI.analiza.__doc__[:80] if PI.analiza.__doc__ else ""), flush=True)
    tot = {}
    res_con = []
    t0 = time.time()
    for c in con7[:N]:
        try:
            e7, res = eventos_de(c, "7"); e2, _ = eventos_de(c, "v2")
        except Exception as ex:
            print(c, "FALLA", type(ex).__name__, str(ex)[:100], flush=True); continue
        if e7 is None or e2 is None:
            print(c, "sin eventos en una version", e7 is None, e2 is None, flush=True); continue
        if claves is None:
            claves = sorted(e7[0].keys()) if e7 else None
            print("claves de un evento:", claves, flush=True)
        cm = compara(e7, e2)
        for t, v in cm.items():
            tot[t] = tuple(a + b for a, b in zip(tot.get(t, (0, 0, 0, 0)), v))
        res_con.append(c)
    print("CON homografia en la 7 (%d clips, %.0f s):" % (len(res_con), time.time() - t0), flush=True)
    for t, (na, nb, oa, ob) in tot.items():
        print("   %-7s eventos 7 %4d, v2 %4d | de los de la 7 con pareja en v2 +-3 cuadros %s | de v2 en la 7 %s" % (
            t, na, nb, "%.0f %%" % (100 * oa / na) if na else "-", "%.0f %%" % (100 * ob / nb) if nb else "-"), flush=True)
    for t, L in MAG.items():
        L = np.array(L)
        print("   magnitud %-7s n=%4d  7 p50 %.2f  v2 p50 %.2f  |dif| p50 %.2f  corr %.2f" % (
            t, len(L), np.median(np.abs(L[:, 0])), np.median(np.abs(L[:, 1])), np.median(np.abs(L[:, 0] - L[:, 1])),
            np.corrcoef(L[:, 0], L[:, 1])[0, 1] if len(L) > 2 else float("nan")), flush=True)
    nuevos = {}; n_ok = 0; fis = []
    for c in sin7[:N]:
        try:
            e2, res = eventos_de(c, "v2")
        except Exception as ex:
            print(c, "FALLA", type(ex).__name__, str(ex)[:100], flush=True); continue
        if e2 is None:
            print(c, "sin _r/_Mk:", sorted(k for k in res if k.startswith("_")), res.get("motivo"), flush=True); continue
        n_ok += 1
        for e in e2:
            nuevos[e["tipo"]] = nuevos.get(e["tipo"], 0) + 1
    print("SIN homografia en la 7 (%d probados, %d con eventos v2): %s" % (min(N, len(sin7)), n_ok, nuevos), flush=True)
