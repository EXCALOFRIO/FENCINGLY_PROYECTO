"""Pinta el cuadro del tocado con lo que ve `pista_lineas`: segmentos (verde), bordes lejano/cercano
(azul/rojo), rayas transversales (amarillo), pies (magenta). Para caracterizar a ojo.

    python mira.py SALIDA clip1 clip2 ...
"""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import catalogo as CA  # noqa: E402
import pista_lineas as PL  # noqa: E402
import rutas as RU  # noqa: E402
import zona_tocado as ZT  # noqa: E402


def linea(img, l, col):
    a, b, c = l
    w = img.shape[1]
    if abs(b) < 1e-9:
        return
    y0, y1 = -(a * 0 + c) / b, -(a * w + c) / b
    cv2.line(img, (0, int(y0)), (w, int(y1)), col, 2)


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    df = CA.manifiesto().set_index("clip_id")
    for clip in sys.argv[2:]:
        video = RU.video(clip)
        z = np.load(RU.TRABAJO / ("poses_%s.npz" % clip), allow_pickle=True)
        KP, SC, B, origen = z["KP"], z["SC"], z["B"], z["origen"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        tf, _ = ZT.cuadro_tocado(clip, video, df.loc[clip, "torneo"])
        k26 = ZT.a_halpe(KP, SC)
        Bpx = B * np.array([W, H, W, H])
        cap = cv2.VideoCapture(video)
        pf = PL.PisteFeatures(ZT.CFG_VIDEO, (W, H))
        img_t = None
        for t in range(min(T, tf + 1)):
            ok, img = cap.read()
            if not ok:
                break
            okb = np.isfinite(Bpx[t]).all(1)
            pf.process(img, Bpx[t][okb].astype(np.float32), ZT.BANDA_ROTULO * H, k26[t][okb])
            img_t = img
        cap.release()
        t = len(pf.feats) - 1
        f = pf.feats[t]
        pies = np.full((2, 2), np.nan, np.float32)
        alt = np.full(2, np.nan, np.float32)
        for s in (0, 1):
            if origen[t, s] in (0, 1) and np.isfinite(Bpx[t, s]).all():
                pies[s] = ZT.apoyo(k26[t, s], Bpx[t, s])
                alt[s] = ZT.largo_cuerpo(k26[t, s])
        obs = PL.observe(f, pies, alt, (W, H))
        vis = img_t.copy()
        for s in f.segs:
            cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 255, 0), 1)
        if obs.far is not None:
            linea(vis, obs.far, (255, 0, 0))
        if obs.near is not None:
            linea(vis, obs.near, (0, 0, 255))
        for a, b in (obs.trans or []):
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 3)
        for p in pies:
            if np.isfinite(p).all():
                cv2.circle(vis, (int(p[0]), int(p[1])), 8, (255, 0, 255), -1)
        cv2.putText(vis, "%s c%d trans=%d" % (clip[:30], t, len(obs.trans or [])), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        s = 960 / W
        cv2.imwrite(str(out / ("%s.jpg" % clip)), cv2.resize(vis, (960, int(H * s))))
        print(clip, "tf", tf, "trans", len(obs.trans or []), "far", obs.far is not None,
              "near", obs.near is not None, flush=True)


if __name__ == "__main__":
    main()
