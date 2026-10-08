"""Senal 1: AUDIO. Zumbador (replica f10_audio_zumbador.py, muestra mas grande) + transitorios
de banda ancha pre-tocado (candidato a contacto de hoja).

Lee SOLO del proyecto (manifiesto, lights_<clip>.csv, poses_<clip>.npz, el mp4). Escribe SOLO en
el scratchpad. No usa GPU, no toca nada en el proyecto.

Referencia de tiempo del tocado: t0 = min(t_encendido_s) de lights_<clip>.csv (la primera luz que
se enciende). Esto NO es la 'verdad_luces_ojo' a ojo que usaba f10 (esa solo cubria 21 clips de una
muestra manual): aqui se usa la medida de la propia etapa 9 (lights_), que documentadamente acierta
la ROI al 90,5% por sede y tiene su propio error -- se declara asi, no se esconde.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave

import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion\codigo")
import rutas as RU  # noqa: E402

BASE = pathlib.Path(r"C:\Users\Alejandro\AppData\Local\Temp\claude\C--Users-Alejandro-Desktop-FENCINGLY-AI"
                     r"\2c75119e-09b9-43eb-a841-faa6869534b2\scratchpad\nuevas_senales")
OUT = BASE / "out"
TMP_WAV = BASE / "tmp_wav"
TMP_WAV.mkdir(parents=True, exist_ok=True)

BANDA_ZUMBADOR = (2500.0, 4600.0)
EXCLUYE_HZ = 200.0
VENTANA_ZUMB = (-0.15, 0.45)
CONTROL_ZUMB = (-3.15, -2.55)
TNR_MIN = 8.0
MARGEN_MIN = 6.0

VENTANA_PRE_S = 2.0       # ventana pre-tocado para transitorios
CONTROL_PRE_S = (3.0, 5.0)  # ventana de control: [t0-5, t0-3], misma duracion (2s)
FREC_MIN_TRANS = 1200.0   # banda ancha: por encima de esto para separar de voz/graves de fondo


def ffprobe_audio(ruta: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
           "stream=codec_name,sample_rate,channels", "-of", "json", ruta]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        d = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return dict(tiene_audio=False, codec="", sr=0)
    st = d.get("streams") or []
    if not st:
        return dict(tiene_audio=False, codec="", sr=0)
    return dict(tiene_audio=True, codec=st[0].get("codec_name", ""),
                sr=int(st[0].get("sample_rate", 0) or 0))


def saca_wav(ruta: str, destino: pathlib.Path, sr: int = 48000) -> bool:
    if destino.is_file():
        return True
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", ruta, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "wav", str(destino)]
    subprocess.run(cmd, capture_output=True)
    return destino.is_file() and destino.stat().st_size > 1000


def lee_wav(ruta: pathlib.Path):
    with wave.open(str(ruta), "rb") as w:
        sr = w.getframerate()
        x = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
             / 32768.0)
    return x, sr


def tnr_maximo(x: np.ndarray, sr: int, t0: float, t1: float, n_fft: int = 4096):
    i0, i1 = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
    if i1 - i0 < n_fft:
        return 0.0, 0.0
    seg = x[i0:i1]
    hop = n_fft // 2
    win = np.hanning(n_fft)
    ventanas = [np.abs(np.fft.rfft(seg[k:k + n_fft] * win))
                for k in range(0, len(seg) - n_fft + 1, hop)]
    if not ventanas:
        return 0.0, 0.0
    S = np.mean(ventanas, axis=0)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    db = 20.0 * np.log10(S + 1e-9)
    m = (f >= BANDA_ZUMBADOR[0]) & (f <= BANDA_ZUMBADOR[1])
    idx = np.where(m)[0]
    mejor, fmejor = -np.inf, 0.0
    for i in idx:
        vecinos = (np.abs(f - f[i]) > EXCLUYE_HZ) & (f > 1000) & (f < 8000)
        prom = db[i] - float(np.median(db[vecinos]))
        if prom > mejor:
            mejor, fmejor = prom, float(f[i])
    return float(mejor), fmejor


def flujo_espectral(x: np.ndarray, sr: int, t_ini: float, t_fin: float,
                     n_fft: int = 1024, hop: int = 256, f_min: float = FREC_MIN_TRANS):
    """Novedad espectral banda-ancha (>= f_min) en [t_ini, t_fin]. Devuelve (tiempos, novedad)."""
    i0, i1 = max(0, int(t_ini * sr)), min(len(x), int(t_fin * sr))
    if i1 - i0 < n_fft + hop:
        return np.array([]), np.array([])
    seg = x[i0:i1]
    win = np.hanning(n_fft)
    frames = [np.abs(np.fft.rfft(seg[k:k + n_fft] * win))
              for k in range(0, len(seg) - n_fft + 1, hop)]
    if len(frames) < 3:
        return np.array([]), np.array([])
    S = np.array(frames)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = f >= f_min
    Sb = S[:, m]
    diff = np.diff(Sb, axis=0)
    flujo = np.sum(np.maximum(diff, 0.0), axis=1)
    tiempos = t_ini + (np.arange(len(flujo)) + 1) * hop / sr
    return tiempos, flujo


def cuenta_picos(tiempos: np.ndarray, flujo: np.ndarray, dist_min_s: float = 0.05):
    """Picos sobre una linea base adaptativa (mediana + 3*MAD), separados >= dist_min_s."""
    if len(flujo) < 5:
        return 0, []
    from scipy.signal import find_peaks
    med = np.median(flujo)
    mad = np.median(np.abs(flujo - med)) + 1e-9
    umbral = med + 3.0 * 1.4826 * mad
    dist = max(1, int(dist_min_s / max(1e-6, (tiempos[1] - tiempos[0]))))
    pk, _ = find_peaks(flujo, height=umbral, distance=dist)
    return len(pk), tiempos[pk].tolist()


def t0_de_lights(cid: str) -> tuple:
    p = RU.TRABAJO / f"lights_{cid}.csv"
    if not p.is_file():
        return None, "sin_lights_csv"
    d = pd.read_csv(p)
    if d.empty or "t_encendido_s" not in d.columns:
        return None, "lights_csv_vacio"
    t0 = float(d["t_encendido_s"].min())
    if not np.isfinite(t0) or t0 <= 0:
        return None, "t0_no_valido"
    return t0, "ok"


def procesa_uno(cid: str, torneo: str) -> dict:
    t_ini_proc = time.perf_counter()
    fila = dict(clip=cid, torneo=torneo)
    video = RU.video(cid)
    if video is None:
        fila["estado"] = "sin_video"
        return fila
    info = ffprobe_audio(video)
    fila.update(info)
    if not info["tiene_audio"]:
        fila["estado"] = "sin_audio"
        return fila
    t0, motivo_t0 = t0_de_lights(cid)
    fila["t0_s"] = t0
    fila["motivo_t0"] = motivo_t0
    wav = TMP_WAV / f"{cid}.wav"
    if not saca_wav(video, wav):
        fila["estado"] = "ffmpeg_fallo"
        return fila
    x, sr = lee_wav(wav)
    dur = len(x) / sr
    fila["dur_audio_s"] = round(dur, 2)
    try:
        os.remove(wav)  # no acumular wav en el scratchpad
    except OSError:
        pass

    if t0 is not None and t0 + VENTANA_ZUMB[1] <= dur:
        tnr, fpico = tnr_maximo(x, sr, t0 + VENTANA_ZUMB[0], t0 + VENTANA_ZUMB[1])
        tnr_c, fpico_c = tnr_maximo(x, sr, max(0.0, t0 + CONTROL_ZUMB[0]),
                                     max(0.6, t0 + CONTROL_ZUMB[1]))
        fila.update(tnr_db=round(tnr, 1), f_pico_hz=round(fpico, 0),
                    tnr_control_db=round(tnr_c, 1), margen_db=round(tnr - tnr_c, 1))
        fila["zumbador_pasa"] = bool(tnr >= TNR_MIN and (tnr - tnr_c) >= MARGEN_MIN)
    else:
        fila["zumbador_pasa"] = None

    # transitorios banda ancha: pre-tocado vs control (misma duracion, antes en el tiempo)
    if t0 is not None:
        ti, tf = max(0.0, t0 - VENTANA_PRE_S), t0
        tc0, tc1 = max(0.0, t0 - CONTROL_PRE_S[1]), max(0.0, t0 - CONTROL_PRE_S[0])
        tt, fl = flujo_espectral(x, sr, ti, tf)
        tc, flc = flujo_espectral(x, sr, tc0, tc1)
        n_pre, t_picos_pre = cuenta_picos(tt, fl)
        n_ctrl, _ = cuenta_picos(tc, flc)
        fila["n_transitorios_pre2s"] = n_pre
        fila["n_transitorios_control2s"] = n_ctrl
        fila["gap_ultimo_pico_a_tocado_s"] = (
            round(t0 - t_picos_pre[-1], 3) if t_picos_pre else None)
    else:
        fila["n_transitorios_pre2s"] = None
        fila["n_transitorios_control2s"] = None

    fila["estado"] = "ok"
    fila["t_proc_s"] = round(time.perf_counter() - t_ini_proc, 2)
    return fila


def main():
    muestra = pd.read_csv(OUT / os.environ.get("MUESTRA_CSV", "muestra_clips.csv"))
    filas = []
    t_ini = time.perf_counter()
    for i, r in muestra.iterrows():
        fila = procesa_uno(r["clip_id"], r["torneo"])
        fila["initiated"] = r["initiated"]
        filas.append(fila)
        if (i + 1) % 20 == 0:
            print(f"... {i + 1}/{len(muestra)}  ({time.perf_counter() - t_ini:.0f}s)")
    d = pd.DataFrame(filas)
    d.to_csv(OUT / "audio_resultado.csv", index=False)
    print("\nguardado:", OUT / "audio_resultado.csv")
    print("tiempo total:", round(time.perf_counter() - t_ini, 1), "s  /",
          round((time.perf_counter() - t_ini) / max(1, len(muestra)), 2), "s por clip")

    print("\n--- estado ---")
    print(d["estado"].value_counts())
    con_audio = d[d.estado == "ok"]
    print(f"\ncon pista de audio: {(d.tiene_audio == True).sum()}/{len(d)} "
          f"= {(d.tiene_audio == True).mean():.3f}")
    print(f"procesados ok (audio + t0 valido para zumbador): "
          f"{con_audio.zumbador_pasa.notna().sum()}")
    zz = con_audio[con_audio.zumbador_pasa.notna()]
    if len(zz):
        print(f"zumbador 'pasa' (TNR>={TNR_MIN} y margen>={MARGEN_MIN} dB): "
              f"{zz.zumbador_pasa.sum()}/{len(zz)} = {zz.zumbador_pasa.mean():.3f}")
        print(f"TNR ventana tocado p50={zz.tnr_db.median():.1f} dB, "
              f"TNR control p50={zz.tnr_control_db.median():.1f} dB")


if __name__ == "__main__":
    main()
