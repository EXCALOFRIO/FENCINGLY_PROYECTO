"""La prueba que decide el encargo: AJUSTE en unas familias, PRUEBA en las otras, y las dos cifras aparte.

EL CENSO, MIRADO, QUE DEFINE LAS FAMILIAS. Los 21 rotulos se han mirado uno a uno en
`02_validacion/muestras/luces/censo/mosaico_rotulos_*.png`. Salen CINCO familias de marcador, no 21
casos:

  A  FIE clasica        rotulo azul en losange, dos placas de nombre, reloj central con pestana de
                        periodo, banderas circulares en los extremos. La LUZ es una BARRA que rellena
                        el borde inferior de la placa del nombre. 16 sedes.
  B  banner ruso plano  banner azul oscuro plano, caja de reloj separada y centrada debajo. La LUZ es
                        un DISCO o cuadrado redondeado a un lado del reloj. 2 sedes (Kazan, San
                        Petersburgo).
  C  Tokio 2022         placas rectangulares gris oscuro, pestana de ronda. La LUZ es una BARRA FINA
                        en el borde inferior de la placa. 1 sede.
  D  OBS olimpico       banner claro, la LUZ son dos PUNTOS pequenos que flanquean la caja del
                        periodo, y ademas la placa del que puntua se tine de rosa. 1 sede.
  E  Paris 2022         rotulo pequeno y centrado con texto minusculo y "Tableau 64". La LUZ es UN
                        SOLO cuadradito entre los dos tanteos, que se pone verde o rojo. 1 sede.

CONJUNTO DE AJUSTE Y CONJUNTO DE PRUEBA, declarados. Se ajusta SOLO en la familia A y se prueba en
B, C, D y E, o sea **4 familias y 5 sedes fuera del ajuste**, mas de las 2 que pedia el encargo. Los
cinco arreglos que llevo el detector (la condicion de pico, la ultima racha en vez de la mas larga,
el ancla del ultimo evento fuerte, el umbral del anillo y el filtro de contraste del anillo) salieron
todos de sedes de la familia A: `anaheimfoil2018`, `qatarfoil2021`, `turinfoil2023`,
`serbiafoil2022` y `belgradefoil2022`. Las cinco sedes de B-E no se han usado para mover ningun
umbral.

CRITERIO DECLARADO ANTES DE VER LAS CIFRAS. Un clip cuenta como ACERTADO si el detector da
exactamente el mismo conjunto de (canal, lado) que la anotacion a ojo. Y el criterio de
generalizacion, el que decide si el detector sirve para el encargo:

  el acierto en las familias B-E no puede caer mas de 20 puntos por debajo del de la familia A

Veinte puntos y no cero porque con 5 sedes de prueba un solo fallo son 20 puntos, asi que un umbral
mas estrecho no seria medible con esta muestra. Y se publica el acierto POR SEDE, no solo el global,
porque una media alta puede esconder dos sedes a cero.

LA LINEA BASE GRATIS, al lado de todo: el detector VIEJO del proyecto, que produjo los NPZ. No es un
espantajo: es lo que hay en produccion, con dos implementaciones (`fie` y `generic`) y un censo de
marcadores ya escrito en sus docstrings.

LO QUE ESTA MUESTRA NO PUEDE DECIR. Son 21 clips, UNO por sede. Un acierto de 21/21 con un clip por
sede no es un acierto del 100 %: es un acierto del 100 % en 21 clips. La cobertura de verdad esta en
`f3_no_lights_por_sede.py`, que mide sobre 25.049 clips cuantos pierde el detector viejo.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

import f4_censo_visual as F4

SALIDA = os.path.join(F4.PROY, r"02_validacion\muestras\luces")

FAMILIA = {
    "anaheimfoil2018": "A_FIE", "anaheimfoil2019": "A_FIE", "belgradefoil2022": "A_FIE",
    "bonnfoil2019": "A_FIE", "bonnfoil2022": "A_FIE", "eurochamps2022": "A_FIE",
    "eurosfoil2023": "A_FIE", "incheonfoil2022": "A_FIE", "katowicefoil2019": "A_FIE",
    "qatarfoil2021": "A_FIE", "serbiafoil2022": "A_FIE",
    "tauberbischofsheimfoil2022": "A_FIE", "tbilisifoil2023": "A_FIE",
    "turinfoil2020": "A_FIE", "turinfoil2023": "A_FIE", "worldchamps2022": "A_FIE",
    "kazanfoil2020": "B_banner_ruso", "stpetersburgfoil2019": "B_banner_ruso",
    "tokyofoil2022": "C_tokio2022",
    "tokyoolympics2020": "D_obs_olimpico",
    "parisfoil2022": "E_paris2022",
}
AJUSTE = {"A_FIE"}


def conjunto_ojo(s: str) -> set[tuple[str, str]]:
    """'verde:R:284|blanca:L:280' -> {('verde','R'), ('blanca','L')}.

    El lado 'C' de `parisfoil2022` (indicador CENTRADO, un solo cuadradito entre los tanteos) se
    normaliza al lado que le toca por el color segun la convencion del aparato, porque en esa familia
    la posicion no lleva informacion de lado y exigirsela seria puntuar un fallo que no existe.
    """
    fuera = set()
    for trozo in s.split("|"):
        canal, lado, _ = trozo.split(":")
        if lado == "C":
            lado = "L" if canal == "roja" else "R"
        fuera.add((canal, lado))
    return fuera


def conjunto_viejo(fila) -> set[tuple[str, str]]:
    """Las luces del NPZ viejo como conjunto (canal, lado), desde `touch_lights`."""
    fuera = set()
    for col, par in (("t_red_L", ("roja", "L")), ("t_green_R", ("verde", "R")),
                     ("t_white_L", ("blanca", "L")), ("t_white_R", ("blanca", "R"))):
        if pd.notna(fila.get(col)):
            fuera.add(par)
    return fuera


def main() -> int:
    ojo = pd.read_csv(os.path.join(SALIDA, "verdad_luces_ojo.csv")).set_index("sede")
    det = pd.read_csv(os.path.join(SALIDA, "luces_detectadas.csv"))
    det = det[det["del_tocado"]] if "del_tocado" in det else det
    viejo = pd.read_csv(os.path.join(F4.SCRATCH, "luces_npz_viejo.csv")).set_index("clip")

    filas = []
    for sede, r in ojo.iterrows():
        v_ojo = conjunto_ojo(r["luces_ojo"])
        sub = det[det["sede"] == sede]
        v_mio = {(x["luz"], x["lado"]) for _, x in sub.iterrows()}
        v_vjo = conjunto_viejo(viejo.loc[r["clip"]]) if r["clip"] in viejo.index else set()

        colores_ojo = {p for p in v_ojo if p[0] != "blanca"}
        colores_mio = {p for p in v_mio if p[0] != "blanca"}
        colores_vjo = {p for p in v_vjo if p[0] != "blanca"}

        c_mio = sub[sub["luz"] != "blanca"]
        cuadro_mio = int(c_mio["cuadro_encendido"].min()) if len(c_mio) else -1
        fila = dict(
            sede=sede, familia=FAMILIA[sede], conjunto=("ajuste" if FAMILIA[sede] in AJUSTE else "prueba"),
            n_ojo=int(r["n_luces_ojo"]), ojo="+".join(sorted(f"{a[0].upper()}{b}" for a, b in v_ojo)),
            mio="+".join(sorted(f"{a[0].upper()}{b}" for a, b in v_mio)) or "-",
            viejo="+".join(sorted(f"{a[0].upper()}{b}" for a, b in v_vjo)) or "-",
            mio_exacto=v_mio == v_ojo, viejo_exacto=v_vjo == v_ojo,
            mio_color=colores_mio == colores_ojo, viejo_color=colores_vjo == colores_ojo,
            mio_fp=len(v_mio - v_ojo), mio_fn=len(v_ojo - v_mio),
            viejo_fp=len(v_vjo - v_ojo), viejo_fn=len(v_ojo - v_vjo),
            cuadro_ojo=int(r["cuadro_ojo_principal"]), tol=int(r["tolerancia_cuadros"]),
            cuadro_mio=cuadro_mio,
        )
        fila["error_cuadro_mio"] = (abs(cuadro_mio - fila["cuadro_ojo"]) if cuadro_mio >= 0 else None)
        fila["cuadro_ok_mio"] = (fila["error_cuadro_mio"] is not None
                                 and fila["error_cuadro_mio"] <= fila["tol"] + 4)
        filas.append(fila)

    d = pd.DataFrame(filas)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("--- POR SEDE (ojo = verdad a ojo, mio = luces.py, viejo = NPZ de produccion) ---")
    print(d[["sede", "familia", "conjunto", "n_ojo", "ojo", "mio", "viejo", "mio_exacto",
             "viejo_exacto", "cuadro_ojo", "cuadro_mio", "error_cuadro_mio"]].to_string(index=False))

    print("\n--- POR FAMILIA ---")
    g = d.groupby(["conjunto", "familia"])
    tab = pd.DataFrame({
        "sedes": g.size(),
        "mio_exacto": g["mio_exacto"].sum(),
        "mio_color": g["mio_color"].sum(),
        "viejo_exacto": g["viejo_exacto"].sum(),
        "viejo_color": g["viejo_color"].sum(),
        "mio_fp": g["mio_fp"].sum(), "mio_fn": g["mio_fn"].sum(),
        "viejo_fp": g["viejo_fp"].sum(), "viejo_fn": g["viejo_fn"].sum(),
        "cuadro_ok_mio": g["cuadro_ok_mio"].sum(),
    })
    print(tab.to_string())

    print("\n--- LAS DOS CIFRAS QUE PIDE EL ENCARGO, POR SEPARADO ---")
    for conj in ("ajuste", "prueba"):
        s = d[d["conjunto"] == conj]
        print(f"  {conj.upper():7s} sedes={len(s):2d}  "
              f"conjunto exacto (canal+lado): mio {s['mio_exacto'].sum()}/{len(s)} "
              f"= {100.0 * s['mio_exacto'].mean():5.1f} %   "
              f"viejo {s['viejo_exacto'].sum()}/{len(s)} = {100.0 * s['viejo_exacto'].mean():5.1f} %")
        print(f"          {'':7s}"
              f"solo los canales de COLOR:    mio {s['mio_color'].sum()}/{len(s)} "
              f"= {100.0 * s['mio_color'].mean():5.1f} %   "
              f"viejo {s['viejo_color'].sum()}/{len(s)} = {100.0 * s['viejo_color'].mean():5.1f} %")
    caida = 100.0 * (d[d["conjunto"] == "ajuste"]["mio_exacto"].mean()
                     - d[d["conjunto"] == "prueba"]["mio_exacto"].mean())
    print(f"\n  CAIDA de ajuste a prueba: {caida:+.1f} puntos.  Criterio declarado: no mas de 20.")
    print(f"  VEREDICTO: {'PASA' if caida <= 20.0 else 'NO PASA'}")

    print("\n--- fallos, uno a uno ---")
    for _, r in d[~d["mio_exacto"] | ~d["viejo_exacto"]].iterrows():
        print(f"  {r['sede']:28s} ojo={r['ojo']:14s} mio={r['mio']:14s} viejo={r['viejo']:14s}")
    print("\n--- cuadro de encendido fuera de tolerancia (mio) ---")
    for _, r in d[~d["cuadro_ok_mio"]].iterrows():
        print(f"  {r['sede']:28s} ojo=c{r['cuadro_ojo']} mio=c{r['cuadro_mio']} "
              f"error={r['error_cuadro_mio']} cuadros, tolerancia {r['tol'] + 4}")

    destino = os.path.join(SALIDA, "generalizacion_por_sede.csv")
    d.to_csv(destino, index=False)
    tab.to_csv(os.path.join(SALIDA, "generalizacion_por_familia.csv"))
    print(f"\ncsv: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
