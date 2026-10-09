# Angulo + homografia (ANGULO_FUSION_2026-10-09)

Scripts del agente J. Leen `rasgos.jsonl` (el `rasgos.jsonl.gz` de `../rayas_id_2026-10-09/`
descomprimido en el directorio de trabajo) y el `modelo.py` de esa carpeta.

- `fusion.py AJUSTE` / `fusion.py PRUEBA "D(2),inf"`: idea 1 (salidas `fusion_ajuste.txt`, `fusion_prueba.txt`).
- `prior.py MITAD N salida.jsonl procs`: idea 2, rehace la etapa 11 con el prior del angulo en una muestra.
- `evalua_prior.py salida.jsonl MITAD [clave]`: su acierto total contra produccion.
- `check_produccion.py`: la logica de produccion (`angulo_lejano` + `zona_fusion`) reproduce la medida.
- `prueba_control.py`: el control `base` de `prior.py` reproduce el `zona_` de disco.
