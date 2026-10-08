"""El ZUMBADOR del aparato en el audio: la via que seria inmune al diseno del marcador, medida.

LA IDEA, QUE NO ES MIA Y ES BUENA. El aparato senializador PITA cuando enciende una luz, y ese
pitido es del aparato FISICO, no del rotulo de television. Si el pitido se detecta, se tiene un
instante de tocado independiente del diseno del marcador, que es justo lo que cambia en cada torneo.
Lo que el audio NO puede dar es DE QUIEN es la luz: el pitido es uno y no tiene lado ni color. El
diseno que saldria de esto es una combinacion -- el audio da el INSTANTE, el rotulo da el COLOR --,
mejor que cualquiera de los dos por separado.

LO QUE HAY QUE MEDIR ANTES DE CREERSELO, y en este orden:

  1. CUANTOS CLIPS TIENEN AUDIO. Si la pista de audio no esta, no hay nada que detectar. Se mide con
     `ffprobe` sobre los 21 mp4, no con el campo `has_audio` del NPZ viejo, que es lo que dijo otro
     proceso y no lo que hay en el fichero.
  2. SI EL PITIDO ESTA, cuando hay audio. Se busca un TONO estable en 2,5-4,6 kHz en una ventana
     alrededor del encendido anotado a ojo, y se compara con una ventana de CONTROL tres segundos
     antes. El control es imprescindible: si el control tambien da un tono de 8 dB, la deteccion no
     dice nada.
  3. SI LA FRECUENCIA ES LA MISMA EN TODOS LOS TORNEOS. Esa es la pregunta de generalizacion para el
     audio, y es la que decide si sirve una banda fija o hace falta buscarla por clip.

CRITERIO DECLARADO ANTES DE EJECUTAR. El audio pasa a ser la via PRINCIPAL solo si:

  (a) al menos el 90 % de los clips tienen pista de audio, Y
  (b) en al menos el 90 % de esos, la prominencia tonal en la ventana del encendido es >= 8 dB
      Y supera a la de la ventana de control en >= 6 dB.

Si (a) falla, el audio no puede ser la via principal por mucho que (b) salga perfecto, y se dice asi.

COMO SE MIDE LA PROMINENCIA. Espectro medio de la ventana; para cada bin de 2,5-4,6 kHz, el nivel
del bin menos la MEDIANA de los bines vecinos excluyendo +-200 Hz; se toma el maximo. Eso es una
relacion tono-ruido (TNR) y no un simple nivel, asi que no la sube el griterio del pabellon, que es
banda ancha. Los 8 dB no son un umbral inventado: es la prominencia que el propio proyecto ya tenia
medida para el aparato (+9 dB sobre el fondo) en `fencingly_v2/audio.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave

import numpy as np
import pandas as pd

import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")
TMP = os.path.join(F4.SCRATCH, "wav")
BANDA = (2500.0, 4600.0)
EXCLUYE_HZ = 200.0
VENTANA = (-0.15, 0.45)
CONTROL = (-3.15, -2.55)
TNR_MIN = 8.0
MARGEN_MIN = 6.0


def ffprobe_audio(ruta: str) -> dict:
    """Devuelve si el mp4 trae pista de audio y con que codec y frecuencia. Mide el fichero."""
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


def saca_wav(ruta: str, destino: str, sr: int = 48000) -> bool:
    if os.path.exists(destino):
        return True
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", ruta, "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "wav", destino]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(destino) and os.path.getsize(destino) > 1000


def lee_wav(ruta: str) -> tuple[np.ndarray, int]:
    with wave.open(ruta, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def tnr_maximo(x: np.ndarray, sr: int, t0: float, t1: float, n_fft: int = 4096) -> tuple[float, float]:
    """(prominencia maxima en dB, frecuencia del pico en Hz) del tono dominante en BANDA."""
    i0, i1 = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
    if i1 - i0 < n_fft:
        return 0.0, 0.0
    seg = x[i0:i1]
    hop = n_fft // 2
    ventanas = []
    win = np.hanning(n_fft)
    for k in range(0, len(seg) - n_fft + 1, hop):
        ventanas.append(np.abs(np.fft.rfft(seg[k:k + n_fft] * win)))
    if not ventanas:
        return 0.0, 0.0
    S = np.mean(ventanas, axis=0)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    db = 20.0 * np.log10(S + 1e-9)
    m = (f >= BANDA[0]) & (f <= BANDA[1])
    idx = np.where(m)[0]
    mejor, fmejor = -np.inf, 0.0
    for i in idx:
        vecinos = (np.abs(f - f[i]) > EXCLUYE_HZ) & (f > 1000) & (f < 8000)
        prom = db[i] - float(np.median(db[vecinos]))
        if prom > mejor:
            mejor, fmejor = prom, float(f[i])
    return float(mejor), fmejor


def main() -> int:
    os.makedirs(TMP, exist_ok=True)
    ojo = pd.read_csv(os.path.join(SALIDA, "verdad_luces_ojo.csv"))
    refs = {r["sede"]: r for r in F4.carga_referencias()}
    filas = []
    for _, r in ojo.iterrows():
        sede = r["sede"]
        ref = refs.get(sede)
        if ref is None or not ref["ruta"]:
            continue
        info = ffprobe_audio(ref["ruta"])
        fila = dict(sede=sede, clip=r["clip"], **info)
        if info["tiene_audio"]:
            wav = os.path.join(TMP, f"{r['clip']}.wav")
            if saca_wav(ref["ruta"], wav):
                x, sr = lee_wav(wav)
                import cv2
                cap = cv2.VideoCapture(ref["ruta"])
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                cap.release()
                t_luz = r["cuadro_ojo_principal"] / fps
                tnr, fpico = tnr_maximo(x, sr, t_luz + VENTANA[0], t_luz + VENTANA[1])
                tnr_c, fpico_c = tnr_maximo(x, sr, max(0.0, t_luz + CONTROL[0]),
                                            max(0.6, t_luz + CONTROL[1]))
                fila.update(t_luz_s=round(t_luz, 3), tnr_db=round(tnr, 1),
                            f_pico_hz=round(fpico, 0), tnr_control_db=round(tnr_c, 1),
                            f_pico_control_hz=round(fpico_c, 0),
                            margen_db=round(tnr - tnr_c, 1))
                fila["pasa"] = tnr >= TNR_MIN and (tnr - tnr_c) >= MARGEN_MIN
            else:
                fila["pasa"] = False
        else:
            fila["pasa"] = False
        filas.append(fila)

    d = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print("--- AUDIO en los 21 clips de la muestra A, medido con ffprobe sobre el mp4 ---")
    print(d.to_string(index=False))
    con = d[d["tiene_audio"]]
    print(f"\n(1) clips CON pista de audio: {len(con)}/{len(d)} = {100.0 * len(con) / len(d):.1f} %"
          f"   (criterio declarado: >= 90 %)")
    print(f"    sedes SIN audio: {', '.join(d.loc[~d['tiene_audio'], 'sede'].tolist())}")
    if len(con):
        print(f"(2) de los que tienen audio, con tono >= {TNR_MIN} dB y margen >= {MARGEN_MIN} dB "
              f"sobre el control: {con['pasa'].sum()}/{len(con)} = {100.0 * con['pasa'].mean():.1f} %"
              f"   (criterio declarado: >= 90 %)")
        print(f"    TNR en la ventana del encendido: p50={con['tnr_db'].median():.1f} dB   "
              f"en el CONTROL: p50={con['tnr_control_db'].median():.1f} dB")
        print(f"(3) frecuencia del pico, Hz: {sorted(con['f_pico_hz'].tolist())}")
        print(f"    recorrido: {con['f_pico_hz'].min():.0f} a {con['f_pico_hz'].max():.0f} Hz")
    pasa_a = len(con) / len(d) >= 0.90
    pasa_b = len(con) > 0 and con["pasa"].mean() >= 0.90
    print(f"\nVEREDICTO: (a) {'PASA' if pasa_a else 'NO PASA'}   (b) {'PASA' if pasa_b else 'NO PASA'}")
    print("El audio puede ser la via PRINCIPAL solo si pasan LAS DOS.")
    d.to_csv(os.path.join(SALIDA, "audio_zumbador.csv"), index=False)
    print(f"csv: {os.path.join(SALIDA, 'audio_zumbador.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
