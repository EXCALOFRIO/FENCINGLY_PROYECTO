"""Orquesta la corrida por lotes: extraer, indexar, subir, y reanudar por donde se quedo.

POR QUE POR LOTES Y NO TODO SEGUIDO: en Colab los ficheros viven en `/content`, que **muere con el
runtime**. Extraer 6.000 clips y subirlos al final significa que un corte en el clip 5.999 tira los
5.999. Con lotes de N, un corte cuesta como mucho **un lote**.

LA AUTORIDAD DEL ESTADO NO ES UN CONTADOR NI UN FICHERO. Es la interseccion de lo que existe:
  local    fila en `index.csv` **y** `.npz` en disco
  remoto   el `.tar` del lote presente en el repositorio
El fichero `estado_corrida.json` que se escribe aqui es un **registro de auditoria**, no la verdad:
si se borra, la corrida se reanuda igual leyendo disco y remoto. Esa distincion no es burocracia,
es que durante el ensayo un contador dijo "completado = 3" habiendo 6 ficheros.

TRES MODULOS, TRES RESPONSABILIDADES: `pasos.py` son los pasos, `respaldo.py` es el transporte a
Hugging Face, y este orquesta. El cuaderno no lleva logica.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import barra as B
import respaldo as R

TAM_LOTE = 200
"""Clips por lote. Compromiso medido, no arbitrario: a 17,2 s/clip son ~57 min de trabajo por lote,
o sea que un corte cuesta como mucho eso. Subir cada 200 tampoco satura el remoto: son ~12 MB de
poses por lote. Bajarlo mucho multiplica las subidas sin ganar nada; subirlo arriesga mas trabajo."""


def _registro(ws: Path) -> Path:
    return Path(ws) / "estado_corrida.json"


def lee_registro(ws: Path) -> dict:
    """QUE HACE: lee el registro de auditoria. Si no esta o esta roto, devuelve vacio SIN levantar.

    Nunca es la autoridad: solo sirve para contar la historia de la corrida al usuario.
    """
    ruta = _registro(ws)
    if not ruta.is_file():
        return {"lotes": [], "clips_ok": 0, "clips_fallo": 0, "segundos": 0.0}
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        print("  registro ilegible; sigo con disco y remoto, que son la autoridad")
        return {"lotes": [], "clips_ok": 0, "clips_fallo": 0, "segundos": 0.0}


def escribe_registro(ws: Path, datos: dict) -> None:
    """QUE HACE: guarda el registro tras cada lote, para que un corte deje historia."""
    try:
        _registro(ws).write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print("  no pude escribir el registro (%s); no es fatal" % type(exc).__name__)


def pendientes(man, ws: Path, pasos, api=None, repo: str = "", sedes_de=None) -> dict:
    """QUE HACE: calcula lo que falta, cruzando el estado LOCAL con el estado REMOTO.

    Un clip cuenta como hecho si su `.npz` esta en disco con fila en el indice, **o** si su sede ya
    esta subida como lote al remoto. Lo segundo es lo que hace que reanudar en un runtime NUEVO
    (donde `/content` esta vacio) no vuelva a extraer lo que ya se subio.
    """
    est = pasos.paso2_estado(man, ws)
    falta = list(est["pendiente"])
    remoto = {"clips": set(), "lotes": set(), "leido": False}
    if api and repo:
        remoto = R.estado_remoto(api, repo)
        if remoto["leido"] and remoto["clips"]:
            antes = len(falta)
            falta = [c for c in falta if c not in remoto["clips"]]
            if antes != len(falta):
                print("  descontados por estar ya en el remoto ... %d" % (antes - len(falta)))
    return {"pendiente": falta, "estado_local": est, "remoto": remoto}


def por_lotes(pendiente, rutas: dict, ws: Path, extractor, pasos, PL, P, V,
              modelo_index: str, man=None, api=None, repo: str = "", sedes_de=None,
              tam_lote: int = TAM_LOTE, subir: bool = True, borrar_tras_subir: bool = False) -> dict:
    """QUE HACE: recorre los pendientes en lotes; por cada lote extrae, indexa y sube.

    QUE IMPRIME: una barra de lotes con ETA, y dentro de cada lote la barra por clip de la
    extraccion. Tras cada lote, una linea con lo que entro, lo que fallo y lo que se subio.

    QUE DEVUELVE: el detalle por lote y el total. `subidos` es que el remoto acepto el fichero,
    NO que su contenido sea correcto: eso lo dice el paso 4 con el lector de verdad.

    `borrar_tras_subir` libera el `.npz` local despues de subirlo. Por omision NO, porque perder
    el local por un fallo de subida silencioso seria peor que llenar el disco.
    """
    pendiente = list(pendiente)
    lotes = [pendiente[i:i + tam_lote] for i in range(0, len(pendiente), tam_lote)]
    reg = lee_registro(ws)
    hechos_remoto = (R.estado_remoto(api, repo)["clips"] if (api and repo and subir) else set())
    sello = time.strftime("%Y%m%dT%H%M%S")

    print("=" * 78)
    print("CORRIDA POR LOTES")
    print("  pendientes ...... %d" % len(pendiente))
    print("  tamano de lote .. %d" % tam_lote)
    print("  lotes ........... %d" % len(lotes))
    print("  subir ........... %s" % ("si, a %s" % repo if (subir and repo) else "NO"))
    print("=" * 78)

    detalle, t0 = [], time.perf_counter()
    with B.Barra(total=len(lotes), desc="lotes", unit="lote") as bl:
        for i, lote in enumerate(lotes, 1):
            t_lote = time.perf_counter()
            sub = {c: rutas[c] for c in lote if c in rutas}
            res = pasos.paso3_extrae(lote, sub, ws, extractor, PL, P, V)
            ok = int((res["estado"] == "ok").sum()) if ("estado" in res and len(res)) else 0
            fallo = len(res) - ok if len(res) else 0

            idx = pasos.paso4_indice(res, ws, modelo_index, manifiesto=man)

            subidos = 0
            if subir and api and repo:
                grupos = {}
                for c in lote:
                    sede = (sedes_de or {}).get(c) or "sin_sede"
                    grupos.setdefault(sede, []).append(c)
                r = R.respalda_npz(api, repo, Path(ws) / "npz", grupos, hechos_remoto,
                                   etiqueta="%s_%04d" % (sello, i), tmp=Path(ws) / "_tar",
                                   mensaje="lote %d/%d de la corrida" % (i, len(lotes)))
                subidos = r["subidos"]
                for d in r["detalle"]:
                    """Solo cuenta como hecho si el .tar Y su manifiesto subieron: sin manifiesto
                    no se puede descontar por clip en un runtime nuevo, asi que marcarlo como
                    hecho perderia esos clips."""
                    if d.get("ok") and d.get("manifiesto_ok"):
                        for c in d.get("clip_ids", []):
                            hechos_remoto.add(c)
                            if borrar_tras_subir:
                                try:
                                    (Path(ws) / "npz" / ("%s.npz" % c)).unlink()
                                except Exception:
                                    pass

            dt = time.perf_counter() - t_lote
            fila = {"lote": i, "clips": len(lote), "ok": ok, "fallo": fallo,
                    "subidos": subidos, "segundos": round(dt, 1)}
            detalle.append(fila)
            reg["lotes"] = (reg.get("lotes") or []) + [fila]
            reg["clips_ok"] = reg.get("clips_ok", 0) + ok
            reg["clips_fallo"] = reg.get("clips_fallo", 0) + fallo
            reg["segundos"] = round(reg.get("segundos", 0.0) + dt, 1)
            escribe_registro(ws, reg)

            bl.set_postfix(ok=reg["clips_ok"], fallo=reg["clips_fallo"],
                           s_clip=("%.1f" % (dt / max(len(lote), 1))))
            bl.update(1)

    total = time.perf_counter() - t0
    tot_ok = sum(d["ok"] for d in detalle)
    tot_fallo = sum(d["fallo"] for d in detalle)
    print()
    print("=" * 78)
    print("RESULTADO DE LA CORRIDA")
    print("  lotes ........... %d" % len(detalle))
    print("  clips ok ........ %d" % tot_ok)
    print("  clips fallidos .. %d%s" % (tot_fallo, "   <- revisar" if tot_fallo else ""))
    print("  lotes subidos ... %d" % sum(d["subidos"] for d in detalle))
    if tot_ok:
        print("  ritmo ........... %.1f s/clip" % (total / tot_ok))
    print("  registro ........ %s" % _registro(ws))
    print("  AVISO: 'subido' es que el remoto lo acepto, no que el contenido sea correcto.")
    print("=" * 78)
    return {"detalle": detalle, "ok": tot_ok, "fallo": tot_fallo, "segundos": round(total, 1),
            "registro": str(_registro(ws))}


def sedes_del_manifiesto(man) -> dict:
    """QUE HACE: {clip_id: sede} desde el manifiesto, que es lo que agrupa los lotes de subida.

    Los clips sin sede caen en `sin_sede` en vez de perderse, porque perder un clip en silencio es
    peor que agruparlo mal.
    """
    if man is None or not len(man):
        return {}
    col = "tournament_id" if "tournament_id" in man.columns else None
    if not col:
        return {}
    fuera = {}
    for cid, sede in zip(man["clip_id"].astype(str), man[col].astype(str)):
        fuera[cid] = sede if sede and sede != "nan" else "sin_sede"
    return fuera


def informe(ws: Path) -> dict:
    """QUE HACE: cuenta la historia de la corrida desde el registro, para mirarla tras un corte."""
    reg = lee_registro(ws)
    lotes = reg.get("lotes") or []
    print("=" * 78)
    print("HISTORIA DE LA CORRIDA  (registro de auditoria, no autoridad)")
    print("  lotes completados ... %d" % len(lotes))
    print("  clips ok ............ %d" % reg.get("clips_ok", 0))
    print("  clips fallidos ...... %d" % reg.get("clips_fallo", 0))
    seg = reg.get("segundos", 0.0)
    print("  tiempo .............. %.0f s (%.1f h)" % (seg, seg / 3600.0))
    if reg.get("clips_ok"):
        print("  ritmo ............... %.1f s/clip" % (seg / reg["clips_ok"]))
    print("  Para saber que FALTA de verdad, usa `pendientes()`: disco y remoto, no este fichero.")
    print("=" * 78)
    return reg
