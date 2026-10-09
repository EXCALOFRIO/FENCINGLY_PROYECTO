# Luces por fuente (2026-10-09)

Datos y scripts de `04_documentacion/informes/LUCES_POR_FUENTE_2026-10-09.md`. Se corrieron desde
`D:/FENCINGLY_SCRATCH/agente_luces/` (las rutas de dentro apuntan alli); las tiras de fotogramas que
se miraron a ojo siguen alli, en `tiras/` (los PNG no van al repo).

| fichero | que es |
|---|---|
| `mide.py` | recalcula por clip la etapa 9 vigente (R0) y el detector viejo (R1) -> `med_anotados.jsonl.gz` (2.016 clips) |
| `mide_r3.py`, `mide_r3b.py` | los dos arreglos del cuadro 0 -> `med_r3.jsonl.gz`, `med_r3b.jsonl.gz` |
| `evalua.py`, `puertas.py` | las reglas R0-R6 y las puertas declaradas; salida en `evalua_2016.txt`, `puertas_2016.txt` |
| `evalua_clips.csv` | una fila por clip con la lectura de cada regla |
| `c0.py`, `c0b.py`, `ojo10.csv` | las puertas de R3 y R3b; los 10 clips mirados a ojo |
| `fuente.py`, `fuente_*por_torneo.csv` | la hipotesis de la fuente |
| `diag_disco.py`, `analiza_diag.py`, `compara_viejo.py`, `viejo_meta.py` | el diagnostico descriptivo de §1 (`viejo_meta.csv`, 8 MB, se queda en el scratch) |
| `equivalencia.py` | que el codigo portado y el refactorizado dan lo medido |
| `tira.py` | las tiras de la banda del rotulo |
| `lista_anotados.txt` | el orden barajado (semilla 20261009); se midieron los 2.016 primeros |
