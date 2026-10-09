"""EL DETECTOR DE LUCES VIEJO DEL PROYECTO (fencingly_v2), traido tal cual: tres ficheros COPIADOS
BYTE A BYTE, sin tocar una linea, para que la cifra medida sea la de este codigo y no la de una
version "parecida".

DE DONDE SALE (2026-10-09, `04_documentacion/informes/LUCES_POR_FUENTE_2026-10-09.md`). Es el
detector de la extraccion `localrun` de septiembre (`pipeline_version 2026-09-05.time1`), conservado
en `D:/FENCINGLY_respaldo_2026-09-09/scratchpad/fz/codigo/FENCINGLY/webapp/fencingly_v2/`:

    lights.py           e7941ff79a552ae067bcf913eb5cfa6661cfd3b75638f38d5fe825c222db760d
    lights_generic.py   b4554083e9bcd246a1777688f65d30563409644b1c36816df082590a9d9d80d0
    video.py            991becb0eee75fe0e2f3d182f3238ab7440394e95d6262888be1b433d8beccaa

`lights.analyze` busca las cajas de nombre del rotulo FIE (azules y quietas), lee la FRANJA de debajo
de cada una (donde se pinta la barra de la luz) y exige un ESCALON; si no encuentra rotulo FIE o no
da tocado, `lights_generic.analyze_generic` busca regiones roja/verde puras que se encienden de golpe
en la banda inferior (Tokio 2020, Kazan, Incheon...). La entrada es `lights_generic.analyze_any`.

POR QUE VUELVE. La etapa 9 (`luces.py`) se escribio para no depender del diseno del rotulo, y medida
contra `touch` del manifiesto sobre los mismos videos acierta MENOS que este en florete, espada y
sable; las cifras, con su denominador y su puerta declarada antes, estan en el informe. `luces.py`
sigue entero: es el respaldo cuando este no da tocado.

`video.py` trae tambien una funcion de descarga que aqui no se usa; se deja para no tocar el fichero.
"""
