"""Lee touch_lights del detector viejo (fencingly_v2, 'fie') de los NPZ de localrun del respaldo."""
import glob, json, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

def lee(p):
    try:
        with np.load(p, allow_pickle=True) as z:
            m = json.loads(str(z["meta_json"]))
    except Exception as ex:
        return None
    tl = m.get("touch_lights") or {}
    return dict(clip=m.get("clip_id"), ruta=p, version=m.get("pipeline_version"),
                detector=m.get("light_detector"), fuente=m.get("light_channels_source"),
                fps=m.get("fps"), f_touch=m.get("f_touch"), t_touch_s=m.get("t_touch_s"),
                n_valid=m.get("n_valid_lights"), red_L=tl.get("red_L"), green_R=tl.get("green_R"),
                white_L=tl.get("white_L"), white_R=tl.get("white_R"),
                flags="|".join(m.get("light_flags") or []), W=(m.get("size") or [None, None])[0],
                H=(m.get("size") or [None, None])[1], audio=m.get("has_audio"))

if __name__ == "__main__":
    fs = glob.glob(r"D:/FENCINGLY_respaldo_2026-09-09/localrun/**/*.npz", recursive=True)
    with ProcessPoolExecutor(6) as ex:
        filas = [r for r in ex.map(lee, fs, chunksize=200) if r]
    d = pd.DataFrame(filas)
    d.to_csv("viejo_meta.csv", index=False)
    print(len(d), d["clip"].nunique(), d["detector"].value_counts().to_dict())
    print(d["version"].value_counts().head(10).to_dict())
