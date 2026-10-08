"""Fuentes de clips: descarga directa del CDN y manifiestos nuevos, incluidos JSON por URL.

EL DATO QUE FALTABA: los clips estan en

    https://clips.fencingdatabase.com/<torneo>/<clip_id>.mp4

Las paginas HTML del sitio devuelven **403** a un cliente que no sea navegador, y por eso el sitio
parecia inaccesible. El CDN no lo hace. Medido en `hongkongfoil2026`: **1920x1080 a 60 fps**, o sea
que el material nuevo gana CADENCIA (pasos de 16,7 ms) pero **no resolucion** — sigue siendo 1080p
como Turin 2023 y Tokio 2020, asi que la aritmetica del ancho de la hoja no cambia y solo 4K la
moveria.

QUE HACE ESTE MODULO: resolver `<torneo>/<clip_id>` a un mp4 en disco, saltando lo ya descargado, y
aceptar manifiestos nuevos desde CSV, JSON o **una URL de JSON**, filtrando siempre por arma.
"""
from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request

CDN = "https://clips.fencingdatabase.com/%s/%s.mp4"
CABECERAS = {"User-Agent": "Mozilla/5.0"}
"""Sin este User-Agent el CDN puede rechazar la peticion, igual que las paginas HTML."""


def url_de(torneo: str, clip_id: str) -> str:
    return CDN % (torneo, clip_id)


def descarga(torneo: str, clip_id: str, destino_dir, tiempo_max: int = 90) -> dict:
    """QUE HACE: baja un clip si no esta ya. QUE DEVUELVE: ruta, estado y bytes.

    NO relanza la excepcion: devuelve el estado, porque en una corrida de miles un fallo suelto no
    debe tumbar el lote. El llamante decide.
    """
    d = pathlib.Path(destino_dir)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / ("%s.mp4" % clip_id)
    if dst.is_file() and dst.stat().st_size > 10000:
        return {"clip_id": clip_id, "ruta": str(dst), "estado": "ya_estaba",
                "bytes": dst.stat().st_size}
    pet = urllib.request.Request(url_de(torneo, clip_id), headers=CABECERAS)
    try:
        with urllib.request.urlopen(pet, timeout=tiempo_max) as r, open(dst, "wb") as f:
            while True:
                tro = r.read(1 << 20)
                if not tro:
                    break
                f.write(tro)
    except urllib.error.HTTPError as exc:
        return {"clip_id": clip_id, "ruta": None, "estado": "HTTP %s" % exc.code, "bytes": 0}
    except Exception as exc:
        return {"clip_id": clip_id, "ruta": None, "estado": type(exc).__name__, "bytes": 0}
    n = dst.stat().st_size if dst.is_file() else 0
    if n < 10000:
        try:
            dst.unlink()
        except Exception:
            pass
        return {"clip_id": clip_id, "ruta": None, "estado": "vacio", "bytes": n}
    return {"clip_id": clip_id, "ruta": str(dst), "estado": "descargado", "bytes": n}


def descarga_lote(pares, destino_dir, barra=None) -> list:
    """QUE HACE: baja una lista de (torneo, clip_id) con barra de progreso si se le pasa `barra.Barra`."""
    pares = list(pares)
    salida = []
    ctx = barra.Barra(total=len(pares), desc="descarga", unit="clip") if barra else None
    for torneo, cid in pares:
        r = descarga(torneo, cid, destino_dir)
        r["tournament_id"] = torneo
        salida.append(r)
        if ctx:
            ok = sum(1 for s in salida if s["estado"] in ("descargado", "ya_estaba"))
            ctx.set_postfix(ok=ok, fallo=len(salida) - ok)
            ctx.update(1)
    if ctx:
        ctx.close()
    return salida


def lee_manifiesto(origen, arma: str = "foil") -> "list[dict]":
    """QUE HACE: lee un manifiesto de CSV, JSON local o URL de JSON, y filtra por arma.

    EL FILTRO DE ARMA ES EXPLICITO y no se deduce del nombre del fichero. Una fila sin `weapon` se
    DESCARTA, no se asume florete: asumirla metería espada en un corpus de florete sin ningun aviso.
    """
    txt = None
    s = str(origen)
    if s.startswith("http://") or s.startswith("https://"):
        pet = urllib.request.Request(s, headers=CABECERAS)
        with urllib.request.urlopen(pet, timeout=60) as r:
            txt = r.read().decode("utf-8", errors="replace")
        datos = json.loads(txt)
    else:
        p = pathlib.Path(s)
        if p.suffix.lower() == ".csv":
            import pandas as pd
            datos = pd.read_csv(p, low_memory=False).to_dict("records")
        else:
            datos = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    if isinstance(datos, dict):
        for k in ("clips", "data", "results", "items"):
            if isinstance(datos.get(k), list):
                datos = datos[k]
                break
    if not isinstance(datos, list):
        return []

    fuera, sin_arma, otra_arma = [], 0, 0
    for fila in datos:
        if not isinstance(fila, dict):
            continue
        cid = str(fila.get("clip_id") or fila.get("id") or fila.get("gfycat_gfy_id") or "").strip()
        if not cid:
            continue
        w = fila.get("weapon")
        if w is None or str(w).strip() == "":
            sin_arma += 1
            continue
        if str(w).strip().lower() != arma.lower():
            otra_arma += 1
            continue
        fuera.append({"clip_id": cid,
                      "tournament_id": str(fila.get("tournament_id") or fila.get("tournament") or ""),
                      "weapon": str(w).strip().lower()})
    print("  manifiesto %s" % s[:70])
    print("    filas ................ %d" % len(datos))
    print("    aceptadas (%s) ..... %d" % (arma, len(fuera)))
    print("    descartadas sin arma . %d   <- NO se asume florete" % sin_arma)
    print("    descartadas otra arma  %d" % otra_arma)
    return fuera


def mide(ruta) -> dict:
    """QUE HACE: resolucion, cadencia y cuadros de un mp4, mas el ancho que tendria el fuerte.

    El ancho del fuerte sale de la escala medida del corpus (caja del tirador ~0,388 del alto de
    imagen, tirador 1,75 m) y de los ~4 mm de seccion del fuerte. Se incluye porque es la cifra que
    decide si merece la pena intentar detectar la hoja: por debajo de 1 px no hay pixel que segmentar,
    aunque el CONTRASTE pueda dejar una cresta detectable.
    """
    import cv2

    cap = cv2.VideoCapture(str(ruta))
    d = {"ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
         "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
         "cuadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    ok, _ = cap.read()
    cap.release()
    d["decodifica"] = bool(ok)
    ppm = (d["alto"] * 0.388) / 1.75 if d["alto"] else 0.0
    d["px_por_metro"] = round(ppm, 1)
    d["ancho_fuerte_px"] = round(0.004 * ppm, 2)
    d["diam_cazoleta_px"] = round(0.12 * ppm, 1)
    return d
