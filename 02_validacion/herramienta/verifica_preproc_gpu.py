"""Puerta del preproceso en GPU: mismas poses (p50 < 0,02 px, max < 0,5 px en puntos con conf>0,3
en las dos), velocidad intercalada."""
import sys, time, numpy as np
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion\herramienta")
import rutas as RU, video_revision as VR, torch
from transformers import Sapiens2ForPoseEstimation, Sapiens2ImageProcessor
snap = RU.hf_modelo("sapiens2-pose-0.4b")
proc = Sapiens2ImageProcessor.from_pretrained(str(snap)); proc.size = VR.RES_POSE
mod = Sapiens2ForPoseEstimation.from_pretrained(str(snap), dtype=torch.float16).to("cuda").eval()
T_ = {False: 0.0, True: 0.0}; D = []; S = []
for c in sys.argv[1:]:
    z = np.load(RU.TRABAJO / ("revision_%s.npz" % c), allow_pickle=True)
    args = (str(z["video"]), z["B"], z["pres"], z["mask"], int(z["W"]), int(z["H"]), int(z["T"]))
    R = {}
    for g in (False, True, False, True):
        VR.PREPROC_GPU = g; torch.cuda.synchronize(); t = time.perf_counter()
        KP, SC, _ = VR.poses_del_clip(proc, mod, torch, *args, guarda_frames=False)
        torch.cuda.synchronize(); T_[g] += time.perf_counter() - t; R.setdefault(g, (KP, SC))
    (a, sa), (b, sb) = R[False], R[True]
    ok = (sa > 0.3) & (sb > 0.3) & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    d = np.linalg.norm(a - b, axis=-1)[ok]; D.append(d); S.append(np.abs(sa - sb)[ok])
    print("%-34s puntos %7d  p50 %.4f  p99 %.3f  max %.3f px   conf |dif| max %.4f" % (c[:34], ok.sum(), np.median(d), np.percentile(d, 99), d.max(), S[-1].max()), flush=True)
d = np.concatenate(D)
print("TOTAL p50 %.4f px  max %.3f px  | tiempo CPU %.1f s  GPU %.1f s  -> %.2fx" % (np.median(d), d.max(), T_[False], T_[True], T_[False] / T_[True]))
