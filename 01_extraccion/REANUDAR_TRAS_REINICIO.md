# Reanudar tras un reinicio, un corte de luz o un `--para`

No se pierde nada: el estado de cada clip es **lo que hay en disco** (y en el respaldo de HF), no
la memoria de ningún proceso. Relanzar la misma línea sigue exactamente donde se quedó.

Desde `C:\Users\Alejandro\Desktop\FENCINGLY_PROYECTO\01_extraccion`:

    D:\FENCINGLY_PY\venv\Scripts\python.exe noche.py --estado --conjunto foil
    D:\FENCINGLY_PY\venv\Scripts\python.exe noche.py --conjunto foil --descarga --gpu 2 --cpu 2

(`anotados` en lugar de `foil` si lo que corría era el conjunto anotado.)

Comprobar a los 10 minutos que **los números de `--estado` SUBEN**, no solo que los trabajadores
estén "vivos": el 23-24/09 un proceso huérfano siguió vivo sin avanzar.

Si el disco se perdiera: el código está en GitHub (`EXCALOFRIO/FENCINGLY_PROYECTO`, privado) y los
resultados en HF (`EXCALORIO/fencingly-extraccion`, privado, tars + manifiestos). `lotes.hechas`
cuenta lo del remoto como hecho, así que una máquina nueva no repite lo ya subido.

El procedimiento viejo (bucle.py a mano, 25/09) está en
`04_documentacion/informes/REANUDAR_2026-09-25_viejo.md`.
