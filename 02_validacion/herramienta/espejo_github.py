"""Deja el repo de GitHub como copia exacta del publico de Hugging Face: la ficha y los videos.

LO QUE PIDIO EL USUARIO: "en https://github.com/EXCALOFRIO/FENCINGLY.AI pones lo mismo que en ese
publico de Hugging Face, que sea una copia tal cual con el mismo readme y los mismos videos subidos
ahi para tenerlo para explicar lo que hacemos". Y sobre lo que habia: "no lo quiero para nada, eso es
antiguo, puedes ponerlo en otra rama o de hecho borrarlo". Y despues: "los videos si, borralos, que
seran un problema de tamanio luego y no aportan nada; lo otro si quieres dejalo en el historial de git
de otras versiones, pero ahora dejalo limpio".

## QUE SE HACE CON LO QUE HABIA, Y POR QUE ASI

El repo tenia **6,5 GB**: unos 40 clips `.mp4` de television subidos sueltos, 2,6 GB de `.zip` con
mas clips, un notebook de Colab, el PDF del TFG, pesos de modelos antiguos y las fuentes viejas.

  1. **Respaldo completo en disco** antes de tocar nada, con su historia de git, en
     `D:\\FENCINGLY_respaldo_github_2026-09-11`. Lo irreversible se hace despues del respaldo, no
     antes.
  2. **Rama `archivo-2026-09-11`** con todo lo que habia MENOS los ficheros de video
     (`.mp4`, `.avi`, `.gif`, `.zip`): un solo commit, 301 MB en vez de 6,5 GB. Asi "lo otro" sigue
     estando en el repo, que es lo que el usuario dejo elegir, y los videos no.
  3. **`main` se reescribe** con un commit nuevo y solo la ficha y `media/`.

Y hay una razon de fondo para que los videos desaparezcan que va mas alla del tamanio: eran
**grabacion de television publicada en un repo publico**. Lo que se sube ahora en su lugar es solo
salida de modelo sobre negro.

## POR QUE LA FICHA NO SE REESCRIBE AQUI

Se copia la que genero `repo_publico`, byte a byte. Si este fichero tuviera su propia version del
texto, las dos paginas empezarian a divergir en la primera correccion y "copia tal cual" dejaria de
ser verdad. La unica diferencia admitida es el bloque `---` de metadatos de Hugging Face, que en
GitHub no significa nada y saldria como una tabla suelta al principio.

USO:  python espejo_github.py            ensena que haria
      python espejo_github.py --hazlo    lo hace
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

VAL = pathlib.Path(r"C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\02_validacion")
GEN = VAL / "publico_generado"
MEDIOS = GEN / "medios"
TRABAJO = pathlib.Path(r"D:\FENCINGLY_github_nuevo")
RESPALDO = pathlib.Path(r"D:\FENCINGLY_respaldo_github_2026-09-11")
REPO = "https://github.com/EXCALOFRIO/FENCINGLY.AI.git"
RAMA = "main"
AUTOR = ["-c", "user.name=EXCALOFRIO", "-c", "user.email=EXCALOFRIO@users.noreply.github.com"]
"""LOS COMMITS VAN A NOMBRE DE `EXCALOFRIO`, que es el duenio del repo, y a proposito.

La primera tanda los firme con `Alejandro <alejandro.c.ramirez@accenture.com>`, o sea con el correo
de trabajo del usuario, **en un repo publico**. El usuario lo corto: *"no lo hagas con esa cuenta,
ademas en github hazlo con EXCALOFRIO"*. Y tiene dos motivos buenos: el repo es de esa cuenta, y un
correo corporativo en el historial de un repo publico no se quita cambiando un commit, porque queda
en los reflogs y en las copias que alguien ya haya clonado.

`EXCALOFRIO@users.noreply.github.com` es la direccion sin respuesta que GitHub asigna a la cuenta:
atribuye el commit a `EXCALOFRIO` de verdad -sale con su foto y su enlace- y no publica ningun correo
real. El `gh auth status` de esta maquina ya esta como `EXCALOFRIO`, asi que el push y el autor
coinciden.

Y NO se aniade ninguna linea de coautoria de Claude: *"tampoco que salga claude porfa"*."""


def git(*a, cwd=TRABAJO, callado=True):
    r = subprocess.run(["git"] + list(a), cwd=str(cwd), capture_output=True, text=True)
    if r.returncode and not callado:
        raise SystemExit("git %s fallo:\n%s\n%s" % (" ".join(a), r.stdout[-2000:],
                                                    r.stderr[-2000:]))
    return r


def borra_arbol(d):
    """QUE HACE: borra un arbol que contiene un `.git`, en Windows.

    POR QUE NO BASTA `shutil.rmtree`. Git deja los objetos de su base de datos en **solo lectura**, y
    en Windows eso hace fallar el borrado con `PermissionError`. Con `ignore_errors=True` el fallo se
    traga en silencio, el directorio sigue ahi a medias y el `mkdir` de despues muere con
    `FileExistsError` senialando el sitio equivocado: parece que el problema es el directorio, cuando
    el problema fue el borrado dos lineas antes. Es el mismo patron que ya me ha costado tiempo hoy,
    un fallo que se esconde y reaparece disfrazado.

    El manejador quita el bit de solo lectura y reintenta la operacion que fallo.
    """
    import os
    import stat

    def reintenta(func, ruta, _exc):
        os.chmod(ruta, stat.S_IWRITE)
        func(ruta)

    shutil.rmtree(d, onexc=reintenta)


def ficha_sin_cabecera(texto):
    """QUE HACE: quita el bloque YAML de Hugging Face y deja el resto intacto.

    En Hugging Face ese bloque configura la ficha (licencia, etiquetas, `viewer: false`). En GitHub no
    se interpreta: se renderiza como una tabla de guiones al principio del README, que es lo primero
    que veria quien entra. El texto de debajo no se toca.
    """
    if not texto.startswith("---"):
        return texto
    corte = texto.find("\n---", 3)
    if corte < 0:
        return texto
    return texto[corte + 4:].lstrip("\n")


def main():
    hazlo = "--hazlo" in sys.argv
    readme = GEN / "README.md"
    if not readme.exists():
        raise SystemExit("falta %s: corre antes `repo_publico.py`" % readme)
    if not MEDIOS.exists():
        raise SystemExit("falta %s: corre antes `repo_publico.py`" % MEDIOS)
    if not (RESPALDO / ".git").exists():
        raise SystemExit("NO hay respaldo en %s. No se reescribe `main` sin respaldo." % RESPALDO)
    """LA GUARDIA DEL RESPALDO NO ES DECORATIVA: este script hace un `push --force` sobre `main` de un
    repo publico. Sin el clon completo en disco, un fallo a medias no se podria deshacer."""

    ms = sorted(p for p in MEDIOS.iterdir() if p.is_file())
    print("respaldo   %s (%s)" % (RESPALDO, "existe" if RESPALDO.exists() else "NO"))
    print("rama nueva %s con la ficha y %d ficheros de media, %.1f MB"
          % (RAMA, len(ms), sum(p.stat().st_size for p in ms) / 1e6))
    for p in ms[:40]:
        print("   media/%-56s %6.2f MB" % (p.name, p.stat().st_size / 1e6))
    if len(ms) > 40:
        print("   ... y %d mas" % (len(ms) - 40))
    if not hazlo:
        print()
        print("ENSAYO. Con `--hazlo` se reescribe `main` de %s" % REPO)
        return

    if TRABAJO.exists():
        borra_arbol(TRABAJO)
    TRABAJO.mkdir(parents=True)
    git("init", "-q", "-b", RAMA, callado=False)
    git("remote", "add", "origin", REPO, callado=False)
    (TRABAJO / "README.md").write_text(ficha_sin_cabecera(
        readme.read_text(encoding="utf-8")), encoding="utf-8")
    destino = TRABAJO / "media"
    destino.mkdir()
    for p in ms:
        shutil.copy2(p, destino / p.name)
    (TRABAJO / ".gitattributes").write_text(
        "*.mp4 -text\n*.gif -text\n*.png -text\n", encoding="utf-8")
    git("add", "-A", callado=False)
    git(*(AUTOR + ["commit", "-q", "-m", MENSAJE]), callado=False)
    r = git("push", "--force", "origin", RAMA)
    print(r.stdout[-1500:])
    print(r.stderr[-1500:])
    if r.returncode:
        raise SystemExit("el push fallo; `main` remoto sigue como estaba")
    print()
    print("main reescrito: solo la ficha y media/")
    print("   https://github.com/EXCALOFRIO/FENCINGLY.AI")
    print("lo anterior, sin los videos, en la rama archivo-2026-09-11")


MENSAJE = """FENCINGLY: solo la ficha y los videos generados

`main` se queda con dos cosas: el README que explica que se extrae de cada clip
con su precision medida, y los videos e imagenes que lo ensenian.

Todo lo que habia antes esta en la rama `archivo-2026-09-11`, SIN los ficheros de
video: eran varios gigabytes y son grabacion de television que no es nuestra para
redistribuir. Los clips de origen son de FencingDatabase.

Cada capa se ensenia en su propio video, a los fps del clip y en H.264, y cada
cuadro lleva impresa la cifra que la mide."""


if __name__ == "__main__":
    main()
