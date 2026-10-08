"""Respaldo incremental en Hugging Face: resolver el token, subir por lotes y reanudar sin perder nada.

VIENE DE `05_aparcado/FENCINGLY_COLAB/notebooks/respaldo.py` Y SE TRAJO TAL CUAL (2026-09-23).
No se reescribio a proposito: ya tiene pagadas tres disciplinas que costaron errores en su dia --
el token no se imprime NUNCA, se exige que el repositorio sea privado antes de subir (los clips
enseñan atletas identificables), y un lote no se marca hecho si falta su manifiesto, porque un tar
sin manifiesto no se puede descontar clip a clip al reanudar. Reescribirlo habria sido volver a
pagarlas.

POR QUE EXISTE: una corrida de extraccion en Colab se corta. Sin respaldo incremental, cortarse a mitad
tira el trabajo entero. Con respaldo, la siguiente corrida lee lo que ya hay en el remoto y sigue.

EL TOKEN ES UNA CREDENCIAL. Este modulo lo resuelve de tres fuentes, en orden, y NO LO IMPRIME NUNCA:
no en un log, no en un `repr`, no en un mensaje de error. Lo unico que se imprime es la cuenta, el
repositorio y los ficheros. Si alguna vez ves el token en una salida, es un fallo de este modulo.

LO QUE PUBLICO NO ARREGLA, Y CONVIENE SABERLO ANTES: hacer el repositorio publico quita la
autenticacion para LEER, no para ESCRIBIR. Este modulo escribe, asi que necesita token en los dos
casos. Lo publico sirve para que un consumidor descargue sin token, no para que esta subida lo evite.
"""
from __future__ import annotations

import io
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

CLAVE_SECRETO = "HUGGINFACE"
"""Nombre del secreto en Colab y de la clave en el `.env`. El mismo en los dos sitios a proposito."""

_ORIGEN = None
"""De donde salio el token en esta sesion, para poder decirlo sin decir el token."""


def token() -> str | None:
    """QUE HACE: resuelve el token de tres fuentes, en orden, sin imprimirlo.

    ORDEN, y no es arbitrario:
      1. el secreto de Colab (`userdata.get`), que es lo correcto en Colab: no queda en el cuaderno
         ni en el historial de celdas, y sobrevive a reiniciar el runtime;
      2. la variable de entorno `HF_TOKEN`, que es el convenio de `huggingface_hub`;
      3. un `.env` local, solo para ejecutar aqui.

    DEVUELVE: el token, o `None`. Nunca levanta por ausencia: el llamante decide si es fatal.
    """
    global _ORIGEN
    try:
        from google.colab import userdata
        valor = userdata.get(CLAVE_SECRETO)
        if valor and valor.strip():
            _ORIGEN = "secreto de Colab"
            return valor.strip()
    except Exception:
        pass

    for nombre in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", CLAVE_SECRETO):
        valor = os.environ.get(nombre)
        if valor and valor.strip():
            _ORIGEN = "variable de entorno %s" % nombre
            return valor.strip()

    for ruta in (Path.home() / "Desktop" / ".env", Path.cwd() / ".env", Path("/content/.env")):
        if not ruta.is_file():
            continue
        texto = io.open(ruta, encoding="utf-8", errors="replace").read()
        hallado = re.search(r"^%s\s*=\s*[\"']?([^\"'\r\n]+)" % CLAVE_SECRETO, texto, re.M)
        if hallado and hallado.group(1).strip():
            _ORIGEN = "fichero .env local"
            return hallado.group(1).strip()

    return None


def instrucciones() -> str:
    """QUE HACE: devuelve, en texto, como dejar el token puesto para que no se vuelva a pedir."""
    return (
        "Falta el token. En Colab: icono de la LLAVE en la barra izquierda, 'Add new secret',\n"
        "  nombre  %s\n"
        "  valor   el token de escritura de huggingface.co/settings/tokens\n"
        "y activa 'Notebook access'. Se pone UNA vez y no vuelve a pedirse ni al reiniciar el\n"
        "runtime. NO lo escribas en una celda: quedaria guardado en el cuaderno." % CLAVE_SECRETO)


def sesion(exigir: bool = True) -> dict:
    """QUE HACE: abre sesion y comprueba que el token puede ESCRIBIR, que es lo que hace falta.

    QUE COMPRUEBA: que valida, y que su permiso no es de solo lectura. Un token de lectura
    autentica sin fallo y luego revienta al subir el primer lote, o sea horas despues.
    QUE IMPRIME: cuenta, tipo y permiso. NUNCA el token.
    """
    from huggingface_hub import HfApi

    tk = token()
    if not tk:
        print(instrucciones())
        if exigir:
            raise SystemExit("PUERTA: sin token no hay respaldo, y sin respaldo un corte de Colab "
                             "tira la corrida entera. Aborta.")
        return {"ok": False, "motivo": "sin token"}

    api = HfApi(token=tk)
    try:
        yo = api.whoami()
    except Exception as exc:
        raise SystemExit("PUERTA: el token no valida (%s). Revisa el secreto." % type(exc).__name__)

    permiso = ((yo.get("auth") or {}).get("accessToken") or {}).get("role")
    print("  token de ....... %s" % _ORIGEN)
    print("  cuenta ......... %s (%s)" % (yo.get("name"), yo.get("type")))
    print("  permiso ........ %s" % (permiso or "desconocido"))
    if permiso == "read":
        raise SystemExit("PUERTA: el token es de SOLO LECTURA. Autenticaria bien y fallaria al subir "
                         "el primer lote, horas despues. Crea uno de escritura. Aborta.")
    if permiso == "fineGrained":
        print("  aviso .......... permiso 'fineGrained': el rol NO dice si puede escribir en ESTE")
        print("                   repo. Lo unico que lo demuestra es escribir, asi que la primera")
        print("                   subida es la comprobacion de verdad. Si falla por permisos, sera")
        print("                   ahi y no aqui.")
    return {"ok": True, "api": api, "cuenta": yo.get("name"), "permiso": permiso, "origen": _ORIGEN}


def verifica_repo(api, repo: str, privado_esperado: bool) -> dict:
    """QUE HACE: comprueba que la privacidad del repo es la que el llamante cree, antes de subir.

    POR QUE: subir recortes de retransmision o nombres de tiradores a un repo que se creia privado
    es irreversible en la practica, porque lo publico se indexa y se rastrea. Esta funcion levanta
    en vez de confiar en el nombre del repo.
    """
    info = api.repo_info(repo_id=repo, repo_type="dataset", files_metadata=False)
    print("  repo ........... %s" % repo)
    print("  privado ........ %s   (esperado %s)" % (info.private, privado_esperado))
    if bool(info.private) != bool(privado_esperado):
        raise SystemExit(
            "PUERTA: el repo es %s y se esperaba %s. No subo nada hasta que cuadre, porque "
            "publicar es irreversible en la practica." % (
                "PUBLICO" if not info.private else "privado",
                "publico" if not privado_esperado else "privado"))
    return {"privado": bool(info.private), "ficheros": len(info.siblings or [])}


CARPETA_LOTES = "data/index/lotes"
"""Donde viven los manifiestos de lote. Cada uno lista los `clip_id` que lleva su `.tar`."""


def estado_remoto(api, repo: str, prefijo: str = "data/") -> dict:
    """QUE HACE: lee del remoto QUE CLIPS estan ya subidos, leyendo los manifiestos de lote.

    POR QUE POR CLIP Y NO POR SEDE, que es como estaba y era un fallo: una sede se reparte entre
    varios lotes. Si el estado fuera "la sede X ya esta", el segundo lote que tocara la sede X la
    saltaria entera y **esos clips se perderian en silencio**. La prueba de reanudacion no lo veia
    porque las sedes de prueba cabian en un solo lote.

    QUE DEVUELVE: `clips` con los `clip_id` presentes en el remoto, `lotes` con los nombres de
    manifiesto, y `leido` para distinguir "no hay nada" de "no pude leer".

    Misma disciplina que el paso 2: el estado es lo que EXISTE, nunca un contador. Un contador dijo
    "completado = 3" habiendo 6 ficheros, y ese es el fallo que este diseno evita.
    """
    import csv
    import io as _io

    try:
        ficheros = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as exc:
        print("  no pude leer el remoto (%s): trato el estado como VACIO y subo todo"
              % type(exc).__name__)
        return {"clips": set(), "lotes": set(), "ficheros": [], "leido": False}

    presentes = [f for f in ficheros if f.startswith(prefijo)]
    manifiestos = [f for f in ficheros if f.startswith(CARPETA_LOTES) and f.endswith(".csv")]
    tars = [f for f in presentes if f.endswith(".tar")]

    clips, lotes = set(), set()
    for m in manifiestos:
        lotes.add(Path(m).stem)
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=repo, filename=m, repo_type="dataset")
            with _io.open(local, encoding="utf-8", errors="replace") as fh:
                for fila in csv.DictReader(fh):
                    cid = (fila.get("clip_id") or "").strip()
                    if cid:
                        clips.add(cid)
        except Exception as exc:
            print("  manifiesto ilegible %s (%s): sus clips contaran como NO hechos"
                  % (m, type(exc).__name__))

    print("  manifiestos de lote en el remoto ... %d" % len(lotes))
    print("  clips ya subidos ................... %d" % len(clips))
    if tars and not manifiestos:
        print("  AVISO: hay %d .tar sin manifiesto. Son de una version anterior y NO se pueden"
              % len(tars))
        print("         descontar por clip, asi que sus clips se reextraeran. Sube los manifiestos")
        print("         o borra esos .tar para no duplicar.")
    return {"clips": clips, "lotes": lotes, "ficheros": presentes, "tars": tars, "leido": True}


def empaqueta(rutas: list, destino: Path, arcname_de=None) -> dict:
    """QUE HACE: mete los ficheros en un `.tar` sin comprimir.

    POR QUE TAR Y NO FICHEROS SUELTOS: el Hub admite menos de 100.000 ficheros por repositorio y
    menos de 10.000 por carpeta, y el corpus son 34.374 clips con varios ficheros cada uno. Suelto
    no cabe.
    POR QUE SIN COMPRIMIR: los `.npz` ya vienen comprimidos; volver a comprimir gasta CPU y no baja
    el tamano.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    n, total = 0, 0
    with tarfile.open(destino, "w") as tar:
        for ruta in rutas:
            ruta = Path(ruta)
            if not ruta.is_file():
                continue
            tar.add(str(ruta), arcname=(arcname_de(ruta) if arcname_de else ruta.name))
            n += 1
            total += ruta.stat().st_size
    return {"tar": destino, "ficheros": n, "bytes_dentro": total,
            "bytes_tar": destino.stat().st_size if destino.is_file() else 0}


def sube_lote(api, repo: str, local: Path, destino_repo: str, mensaje: str = "") -> dict:
    """QUE HACE: sube un fichero al repo y devuelve lo subido. Reintenta una vez y no oculta el fallo.

    QUE NO HACE: no borra el local. Si la subida fallo, el trabajo sigue en disco.
    """
    local = Path(local)
    ultimo = None
    for intento in (1, 2):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=destino_repo, repo_id=repo,
                            repo_type="dataset", commit_message=(mensaje or "lote de extraccion"))
            print("    subido  %-46s %8.1f MB" % (destino_repo, local.stat().st_size / 1e6))
            return {"ok": True, "destino": destino_repo, "bytes": local.stat().st_size, "intentos": intento}
        except Exception as exc:
            ultimo = exc
            print("    intento %d fallo (%s)" % (intento, type(exc).__name__))
    print("    NO SUBIDO: %s  -> el fichero sigue en %s" % (destino_repo, local))
    return {"ok": False, "destino": destino_repo, "error": type(ultimo).__name__ if ultimo else "?"}


def respalda_npz(api, repo: str, dir_npz, por_sede: dict, hechos: set, etiqueta: str = "",
                 tmp: Path | None = None, mensaje: str = "") -> dict:
    """QUE HACE: empaqueta los `.npz` por sede y los sube, MAS un manifiesto con sus `clip_id`.

    `por_sede`  {sede: [clip_id, ...]}   de que sede es cada clip
    `hechos`    {clip_id, ...}           clips YA subidos, de `estado_remoto`["clips"]
    `etiqueta`  sufijo unico del lote    para que dos lotes de la misma sede no se pisen el `.tar`

    DOS COSAS QUE ANTES ESTABAN MAL Y AHORA NO:
      1. El nombre del `.tar` lleva `etiqueta`, asi que una sede repartida en varios lotes produce
         `poses_<sede>_<etiqueta>.tar` distintos en vez de sobreescribirse.
      2. Se salta POR CLIP, no por sede. Saltar por sede perdia en silencio los clips de la misma
         sede que cayeran en un lote posterior.

    El manifiesto es lo que permite reanudar por clip en un runtime nuevo, y se sube DESPUES del
    `.tar`: si el `.tar` falla, no queda un manifiesto diciendo que estan subidos.
    """
    dir_npz = Path(dir_npz)
    tmp = Path(tmp or tempfile.gettempdir())
    etiqueta = etiqueta or "lote"
    salida, subidos, saltados = [], 0, 0
    for sede in sorted(por_sede):
        quedan = [c for c in por_sede[sede] if c not in hechos]
        saltados += len(por_sede[sede]) - len(quedan)
        rutas = [dir_npz / ("%s.npz" % c) for c in quedan]
        rutas = [r for r in rutas if r.is_file()]
        if not rutas:
            continue
        nombre = "poses_%s_%s" % (sede, etiqueta)
        emp = empaqueta(rutas, tmp / ("%s.tar" % nombre))
        res = sube_lote(api, repo, emp["tar"], "data/poses/%s/%s.tar" % (sede, nombre),
                        mensaje or "poses de %s" % sede)
        dentro = [r.stem for r in rutas]
        if res.get("ok"):
            csv_local = tmp / ("%s.csv" % nombre)
            lineas = ["clip_id,sede,tar"]
            for cid in dentro:
                lineas.append("%s,%s,%s" % (cid, sede, nombre))
            csv_local.write_text("\n".join(lineas) + "\n", encoding="utf-8")
            man = sube_lote(api, repo, csv_local, "%s/%s.csv" % (CARPETA_LOTES, nombre),
                            "manifiesto de %s" % nombre)
            res["manifiesto_ok"] = bool(man.get("ok"))
            if not man.get("ok"):
                print("    AVISO: el .tar subio pero su manifiesto NO. Esos clips se reextraeran.")
            try:
                csv_local.unlink()
            except Exception:
                pass
        try:
            emp["tar"].unlink()
        except Exception:
            pass
        salida.append({"sede": sede, "clips": emp["ficheros"], "clip_ids": dentro,
                       "bytes": emp["bytes_tar"], **res})
        subidos += 1 if res.get("ok") else 0
    print("  paquetes subidos ... %d   clips saltados por estar ya ... %d" % (subidos, saltados))
    return {"detalle": salida, "subidos": subidos, "saltados": saltados}


def respalda_fichero(api, repo: str, local, destino_repo: str, mensaje: str = "") -> dict:
    """QUE HACE: sube un fichero suelto, para el indice y las tablas, que no van en tar."""
    return sube_lote(api, repo, Path(local), destino_repo, mensaje)


def resumen(estado: dict) -> None:
    """QUE HACE: imprime lo que hay que mirar. Sin el token, y sin convertir cobertura en exito."""
    det = estado.get("detalle") or []
    ok = [d for d in det if d.get("ok")]
    mal = [d for d in det if not d.get("ok")]
    print("=" * 78)
    print("RESPALDO")
    print("  lotes subidos ....... %d" % len(ok))
    print("  lotes FALLIDOS ...... %d%s" % (len(mal), "  <- revisar" if mal else ""))
    if ok:
        print("  bytes subidos ....... %.1f MB" % (sum(d.get("bytes", 0) for d in ok) / 1e6))
    for d in mal:
        print("    fallo: %s (%s)" % (d.get("destino"), d.get("error")))
    print("  NOTA: 'subido' es que el remoto lo acepto, no que el contenido sea correcto.")
    print("=" * 78)
