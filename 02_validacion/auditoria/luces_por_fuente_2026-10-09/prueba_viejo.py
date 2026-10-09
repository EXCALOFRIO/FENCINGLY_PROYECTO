import sys, time, json
sys.path.insert(0, r"D:\FENCINGLY_respaldo_2026-09-09\scratchpad\fz\codigo\FENCINGLY\webapp")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
from fencingly_v2 import lights_generic as LG
import rutas as RU
for c in sys.argv[1:]:
    v = RU.video(c)
    t0 = time.perf_counter()
    r = LG.analyze_any(v, clock=True)
    dt = time.perf_counter() - t0
    print(c, round(dt, 2), r.get("source"), r.get("f_touch"), r.get("fps"), r.get("touch_lights"), r.get("n_valid_lights"), r.get("quality_flags"))
    print("   keys:", sorted(k for k in r if not k.startswith("_")))
