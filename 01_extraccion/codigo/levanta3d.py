"""El 3D de los dos tiradores LEVANTADO desde las poses 2D ya guardadas (MotionAGFormer-B, Apache-2.0).

POR QUE EXISTE (2026-10-09, `04_documentacion/informes/3D_LEVANTADO_2026-10-09.md`). Los rasgos del
etiquetador son 2D: un brazo que apunta a la camara parece corto y uno extendido de lado, largo. Un
levantador 2D->3D da los angulos de verdad (codo, rodilla, inclinacion) sin volver a mirar el video
y sin GPU apreciable. Este modulo solo LEVANTA y saca rasgos; la medida esta en
`02_validacion/herramienta/verifica_3d.py`.

LO QUE NO ES: no es 3D metrico en la pista (la salida es relativa a la pelvis y en una escala
normalizada), y el levantador esta entrenado en Human3.6M, no en esgrima: AthletePose3D midio que
los levantadores de H36M dan errores grandes en movimiento atletico rapido. Por eso los rasgos se
validan por lo que aportan, no por su MPJPE.

EL ESPEJO. Con el volteo en la prueba `(f(x) + flip(f(flip(x))))/2` la salida es exactamente
equivariante al espejo, y los rasgos se toman en el marco de ataque de cada tirador (`s = +1` el de
la izquierda, `-1` el de la derecha), asi que espejar el clip es intercambiar las dos ranuras.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

import numpy as np

MAG_DIR = pathlib.Path(os.environ.get("FENCINGLY_MAG_DIR",
                                      "D:/FENCINGLY_SCRATCH/agente_3d/terceros/MotionAGFormer"))
MAG_PESOS = pathlib.Path(os.environ.get(
    "FENCINGLY_MAG_PESOS",
    "D:/FENCINGLY_SCRATCH/agente_3d/terceros/pesos/motionagformer-b-h36m.pth.tr"))
N_VENT = 243

GOLIATH_A_COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42, 21, 9, 10, 11, 12, 13, 14]
"""COCO-17 desde los 308 puntos, con el mapa MEDIDO de `cono_lib`: la muneca ES la raiz de la mano;
la mano 21 cuelga del hombro 6 (derecho en COCO) y la 42 del 5 (izquierdo)."""
IZQ_H36M = [4, 5, 6, 11, 12, 13]
DER_H36M = [1, 2, 3, 14, 15, 16]

RASGOS_3D = ["brazo_ang", "brazo_ext3d", "brazo_adel3d", "brazo_elev", "v_ext3d", "tronco_incl",
             "fondo_prof", "rodilla_del_ang", "rodilla_tras_ang", "perfil", "cadera_alt",
             "mano_prof"]


def _sin_timm():
    """`timm` no esta en el venv y el modelo solo usa `DropPath`, que en inferencia (y con
    `drop_path=0`) es la identidad. Se pone un modulo minimo en memoria en vez de instalar nada."""
    try:
        import timm.models.layers  # noqa: F401
        return
    except ImportError:
        pass
    import torch.nn as nn

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.p = drop_prob

        def forward(self, x):
            return x

    t, tm, tml = (types.ModuleType("timm"), types.ModuleType("timm.models"),
                  types.ModuleType("timm.models.layers"))
    tml.DropPath = DropPath
    t.models, tm.layers = tm, tml
    sys.modules.update({"timm": t, "timm.models": tm, "timm.models.layers": tml})


def carga_modelo(device="cuda"):
    import torch
    import torch.nn as nn
    _sin_timm()
    if str(MAG_DIR) not in sys.path:
        sys.path.insert(0, str(MAG_DIR))
    from model.MotionAGFormer import MotionAGFormer
    mod = MotionAGFormer(n_layers=16, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3, mlp_ratio=4,
                         act_layer=nn.GELU, attn_drop=0.0, drop=0.0, drop_path=0.0,
                         use_layer_scale=True, layer_scale_init_value=1e-5,
                         use_adaptive_fusion=True, num_heads=8, qkv_bias=False, qkv_scale=None,
                         hierarchical=False, num_joints=17, use_temporal_similarity=True,
                         temporal_connection_len=1, use_tcn=False, graph_only=False,
                         neighbour_num=2, n_frames=N_VENT)
    from numpy._core.multiarray import scalar
    seguros = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
               (type(np.dtype("float64")), "numpy.dtypes.Float64DType")]
    """`weights_only=True` con lista blanca minima: el checkpoint trae un escalar de numpy
    (`min_mpjpe`) y nada mas que no sea tensor. Nunca `weights_only=False` sobre un fichero bajado."""
    with torch.serialization.safe_globals(seguros):
        ck = torch.load(str(MAG_PESOS), map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in ck["model"].items()}
    mod.load_state_dict(sd, strict=True)
    return mod.to(device).eval()


def a_h36m(u, sc):
    """u (T, 308, 2) YA NORMALIZADO a pantalla, sc (T, 308) -> (T, 17, 3) H36M, confianza en el canal 3.

    La conversion es la del demo de MHFormer/MotionAGFormer (`h36m_coco_format`), hecha sobre las
    coordenadas NORMALIZADAS y con medias de PARES (la espina es la media de la media de hombros y la
    de caderas, no la de cuatro puntos seguidos): asi, si `u` es exactamente el espejo (`x -> -x` y
    los pares cambiados), la salida es exactamente el espejo, bit a bit. Ver el cambio 2 del informe."""
    c = np.concatenate([u[:, GOLIATH_A_COCO], sc[:, GOLIATH_A_COCO, None]], axis=2)
    h = np.zeros((len(c), 17, 3), dtype="float64")
    ojos_y = c[:, 1:3, 1].sum(1) - c[:, 0, 1]
    cabeza = np.stack([c[:, 1:5, 0].mean(1), ojos_y, c[:, 1:5, 2].mean(1)], axis=1)
    hombros = 0.5 * (c[:, 5] + c[:, 6])
    caderas = 0.5 * (c[:, 11] + c[:, 12])
    torax = hombros + (c[:, 0] - hombros) / 3.0
    espina = 0.5 * (hombros + caderas)
    h[:, 10], h[:, 8], h[:, 0], h[:, 7] = cabeza, torax, caderas, espina
    for hi, ci in zip([9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3],
                      [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
        h[:, hi] = c[:, ci]
    h[:, 9] -= (h[:, 9] - hombros) / 4.0
    h[:, 7, 0] += 2 * (h[:, 7, 0] - 0.5 * (h[:, 0, 0] + h[:, 8, 0]))
    h[:, 8, 1] -= (c[:, 1:3, 1].mean(1) - c[:, 0, 1]) * 2.0 / 3.0
    for j in (0, 7, 8, 9, 10):
        h[:, j, 2] = np.clip(h[:, j, 2], 0, 1)
    return h


def flip(x):
    y = x.copy()
    y[..., 0] *= -1
    y[..., IZQ_H36M + DER_H36M, :] = y[..., DER_H36M + IZQ_H36M, :]
    return y


def _interpola(y, idx, T):
    """y (n, len(idx), 17, 3) en los cuadros `idx` (no decrecientes, con repeticiones) -> (n, T, 17, 3)
    por interpolacion lineal. Lineal, asi que el espejo exacto se conserva."""
    u, primero = np.unique(idx, return_index=True)
    y = y[:, primero]
    t = np.arange(T)
    j = np.clip(np.searchsorted(u, t, side="right") - 1, 0, len(u) - 1)
    k = np.clip(j + 1, 0, len(u) - 1)
    den = np.where(u[k] > u[j], u[k] - u[j], 1)
    w = np.clip((t - u[j]) / den, 0, 1)[None, :, None, None]
    return y[:, j] * (1 - w) + y[:, k] * w


def levanta(mod, X):
    """X (n, T, 17, 3) normalizado -> (n, T, 17, 3), con el volteo en la prueba.

    UNA VENTANA DE 243 muestreada sobre el clip entero si `T <= 2*243` (cambio 3 del protocolo: la
    mediana de los anotados es 300 cuadros, y dos ventanas costaban el doble para dar lo mismo una vez
    remuestreados los rasgos a 64 pasos), y vuelta a todos los cuadros por interpolacion lineal. Mas
    largos, ventanas de 243 consecutivas, la ultima pegada al final."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = next(mod.parameters()).device
    n, T = X.shape[:2]

    def f(x):
        with torch.no_grad():
            return mod(torch.from_numpy(x[None].astype("float32")).to(dev))[0].double().cpu().numpy()

    def tta(x):
        """UNA SECUENCIA POR LLAMADA y sin TF32 (cambio 1): en lote, la misma secuencia daba otro
        ultimo bit segun su posicion, y el levantador (vecinos temporales por similitud, una
        seleccion discontinua) lo amplificaba."""
        return np.stack([0.5 * (f(x[i]) + flip(f(flip(x[i])))) for i in range(n)])

    if T <= 2 * N_VENT:
        idx = np.round(np.linspace(0, T - 1, N_VENT)).astype(int)
        out = _interpola(tta(X[:, idx]), idx, T)
    else:
        out = np.full((n, T, 17, 3), np.nan, dtype="float64")
        ini = list(range(0, T - N_VENT + 1, N_VENT))
        if ini[-1] + N_VENT < T:
            ini.append(T - N_VENT)
        for a in ini:
            idx = np.arange(a, a + N_VENT)
            y = tta(X[:, idx])
            nuevo = np.isnan(out[0, idx, 0, 0])
            out[:, idx[nuevo]] = y[:, nuevo]
    out[:, :, 0, :] = 0.0
    return out


def levanta_npz(mod, p):
    """QUE DEVUELVE: dict con P3 (T, 2, 17, 3) relativo a la pelvis, y lo del npz que hace falta."""
    with np.load(p, allow_pickle=True) as z:
        KP, SC, mask, pres = z["KP"].astype("float64"), z["SC"].astype("float64"), z["mask"], z["pres"]
        W, H, T = int(z["W"]), int(z["H"]), int(z["T"])
        fps = float(z["fps"]) if "fps" in z.files else 25.0
    X = []
    for lado in (0, 1):
        k, s = KP[:, lado].copy(), SC[:, lado].copy()
        bueno = np.isfinite(k).all(axis=(1, 2)) & (s.max(axis=1) > 0)
        if not bueno.any():
            k[:] = W / 2.0
            s[:] = 0.0
        else:
            """Los cuadros sin pose (fuera de `mask` o sin persona) se rellenan con la pose valida mas
            cercana: el levantador pide la secuencia entera, y esos cuadros no entran en los rasgos."""
            ib = np.nonzero(bueno)[0]
            cerca = ib[np.clip(np.searchsorted(ib, np.arange(T)), 0, len(ib) - 1)]
            izq = ib[np.clip(np.searchsorted(ib, np.arange(T)) - 1, 0, len(ib) - 1)]
            cerca = np.where(np.abs(izq - np.arange(T)) < np.abs(cerca - np.arange(T)), izq, cerca)
            k, s = k[cerca], s[cerca]
        u = np.empty_like(k)
        u[..., 0] = (2.0 * k[..., 0] - W) / W
        u[..., 1] = (2.0 * k[..., 1] - H) / W
        """La normalizacion del demo (`x/W*2 - 1`, `y/W*2 - H/W`), escrita como `(2x - W)/W` para
        que el espejo `x -> W - x` de exactamente `-u`: la resta y la division son simetricas en
        IEEE, y `x/W*2 - 1` no lo es."""
        X.append(a_h36m(u, s))
    P3 = levanta(mod, np.stack(X)).transpose(1, 0, 2, 3)
    return {"P3": P3, "mask": mask, "pres": pres, "fps": fps, "T": T}


def _ang(a, b, c):
    u, v = a - b, c - b
    cu = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1)
                                                   * np.linalg.norm(v, axis=1), 1e-9)
    return np.degrees(np.arccos(np.clip(cu, -1, 1))) / 180.0


def _suave(a, k=5):
    if len(a) < k:
        return a
    return np.convolve(np.pad(a, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def talla(P):
    """Escala 3D de un tirador en el clip: mediana de pierna media + pelvis-torax + torax-cabeza."""
    pierna = 0.5 * (np.linalg.norm(P[:, 1] - P[:, 2], axis=1) + np.linalg.norm(P[:, 2] - P[:, 3], axis=1)
                    + np.linalg.norm(P[:, 4] - P[:, 5], axis=1) + np.linalg.norm(P[:, 5] - P[:, 6], axis=1))
    tronco = np.linalg.norm(P[:, 0] - P[:, 8], axis=1) + np.linalg.norm(P[:, 8] - P[:, 10], axis=1)
    return float(np.median(pierna + tronco))


def rasgos_tirador(P, s, fps):
    """P (T, 17, 3) de UN tirador; s = +1 izquierda, -1 derecha. -> (T, 12) en el orden RASGOS_3D."""
    L = max(talla(P), 1e-6)
    Q = P / L
    brazos = [(11, 12, 13), (14, 15, 16)]
    adel = np.stack([s * (Q[:, m, 0] - Q[:, h, 0]) for h, _c, m in brazos])
    i = np.argmax(adel, axis=0)
    t = np.arange(len(Q))
    h_, c_, m_ = (np.array([b[j] for b in brazos])[i] for j in range(3))
    H_, C_, M_ = Q[t, h_], Q[t, c_], Q[t, m_]
    largo = np.linalg.norm(H_ - C_, axis=1) + np.linalg.norm(C_ - M_, axis=1)
    hm = np.linalg.norm(M_ - H_, axis=1)
    ext = hm / np.maximum(largo, 1e-9)
    tob = np.stack([s * Q[:, 6, 0], s * Q[:, 3, 0]])
    j_del = np.argmax(tob, axis=0)
    piernas = [(4, 5, 6), (1, 2, 3)]
    pd = [np.array([piernas[j][k] for j in j_del]) for k in range(3)]
    pt = [np.array([piernas[1 - j][k] for j in j_del]) for k in range(3)]
    tronco = Q[:, 8] - Q[:, 0]
    hombros = Q[:, 14] - Q[:, 11]
    col = {
        "brazo_ang": _ang(H_, C_, M_),
        "brazo_ext3d": ext,
        "brazo_adel3d": adel[i, t],
        "brazo_elev": -(M_[:, 1] - H_[:, 1]) / np.maximum(hm, 1e-9),
        "v_ext3d": np.gradient(_suave(ext)) * fps,
        "tronco_incl": s * tronco[:, 0] / np.maximum(np.linalg.norm(tronco, axis=1), 1e-9),
        "fondo_prof": np.linalg.norm(Q[:, 6] - Q[:, 3], axis=1),
        "rodilla_del_ang": _ang(Q[t, pd[0]], Q[t, pd[1]], Q[t, pd[2]]),
        "rodilla_tras_ang": _ang(Q[t, pt[0]], Q[t, pt[1]], Q[t, pt[2]]),
        "perfil": np.abs(hombros[:, 0]) / np.maximum(np.linalg.norm(hombros, axis=1), 1e-9),
        "cadera_alt": 0.5 * (Q[:, 3, 1] + Q[:, 6, 1]) - Q[:, 0, 1],
        "mano_prof": M_[:, 2] - H_[:, 2],
    }
    return np.stack([col[n] for n in RASGOS_3D], axis=1)


def rasgos_clip3d(r, largo=64):
    """r = salida de `levanta_npz` -> (2, largo, 12), remuestreado sobre los cuadros utiles como el
    etiquetador (`_remuestrea` sobre `nonzero(mask)`). None si hay menos de 8 cuadros utiles."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import etiquetador as ET
    ks = np.nonzero(r["mask"])[0]
    if len(ks) < 8:
        return None
    out = []
    for lado, s in ((0, 1.0), (1, -1.0)):
        F = rasgos_tirador(r["P3"][:, lado], s, r["fps"])
        out.append(np.stack([ET._remuestrea(F[ks, j]) for j in range(F.shape[1])], axis=1))
    return np.nan_to_num(np.stack(out).astype("float32"))


HUESOS = {"brazo_d": (14, 15), "antebrazo_d": (15, 16), "brazo_i": (11, 12), "antebrazo_i": (12, 13),
          "muslo_d": (1, 2), "pierna_d": (2, 3), "muslo_i": (4, 5), "pierna_i": (5, 6)}


def plausibilidad(r):
    """CV de cada hueso dentro del clip (cuadros utiles) y asimetria |I-D|/media, por tirador."""
    ks = np.nonzero(r["mask"])[0]
    res = []
    for lado in (0, 1):
        P = r["P3"][ks, lado]
        lg = {n: np.linalg.norm(P[:, a] - P[:, b], axis=1) for n, (a, b) in HUESOS.items()}
        cv = {n: float(np.std(v) / max(np.mean(v), 1e-9)) for n, v in lg.items()}
        asim = [abs(np.mean(lg[a]) - np.mean(lg[b])) / max(0.5 * (np.mean(lg[a]) + np.mean(lg[b])), 1e-9)
                for a, b in (("brazo_d", "brazo_i"), ("antebrazo_d", "antebrazo_i"),
                             ("muslo_d", "muslo_i"), ("pierna_d", "pierna_i"))]
        res.append({"cv_mediana": float(np.median(list(cv.values()))),
                    "cv_max": float(max(cv.values())), "asim_media": float(np.mean(asim))})
    return res
