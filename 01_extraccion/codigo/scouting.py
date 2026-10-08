"""SCOUTING: las series de eventos de cada clip, juntadas por TIRADOR y por ASALTO.

## POR QUE ESTO EXISTE

La etapa 7 deja una fila por evento (`serie_eventos_<clip>.csv`) y el catalogo dice quien es quien,
quien marco, quien inicio, en que zona y en que blanco. Nada de eso estaba junto: para saber
"cuantos fondos tira X por segundo y con que amplitud" habia que cruzar a mano tres sitios. Este
modulo no mide nada nuevo; **reune** lo que ya esta medido y conserva sus marcas de fiabilidad.

## LAS TRES TABLAS, en `03_datos/scouting/`

    clip_tirador.csv       una fila por (clip, lado): resultado, etiquetas humanas, eventos
    tiradores.csv          una fila por tirador: perfil agregado sobre todos sus clips
    asaltos.csv            una fila por (asalto, tirador): el asalto como unidad

## LAS REGLAS, las mismas que el resto del proyecto

- **`None` no es cero.** Un tirador sin fondos medidos tiene `fondo_mag_p50_m = None`, no 0.
- **Solo lados validos.** Un lado con `lado_valido = False` (el arbitro) no entra en el perfil de
  nadie: se cuenta en `lados_descartados` para que el descarte se vea.
- **Los metros llevan su marca.** La escala a lo largo de la pista **no esta verificada**
  (`publicable` en el 2,6 % de los clips). Por eso cada magnitud en metros va acompaniada de
  `frac_escala_publicable`: la fraccion de los clips del tirador cuya escala pasa el control.
  Los conteos y las duraciones **no** dependen de la escala y son la parte solida.
- **Las etiquetas humanas no se mezclan con las del modelo.** `inicia`, `zona` y `blanco` salen de
  la anotacion del catalogo, nunca de `etiquetas_<clip>.csv`.
- **Lado 0 es `fotl`.** El catalogo llama `fotl` al tirador de la izquierda de la imagen y la
  etapa 7 llama 0 al de la izquierda: es el mismo convenio, y `0 cruces en 2.896 cuadros` lo
  sostiene.

    python scouting.py               reconstruye las tres tablas LEYENDO EL DISCO
"""
from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import catalogo as CA  # noqa: E402
import rutas as RU  # noqa: E402

SALIDA = RU.DATOS / "scouting"
TIPOS_PIES = ("marcha", "romper", "fondo", "alto")
"""Los eventos de pies que se resumen. `bote` sale con 0 por balistica y `accion` es un tramo, no
un gesto, asi que van aparte."""


def _p50(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _max(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.max()) if len(s) else None


def _resultado(touch, lado):
    """QUE DEVUELVE: (marca, recibe) para ese lado. Un doble cuenta para los dos."""
    yo, otro = ("left", "right") if lado == 0 else ("right", "left")
    return touch in (yo, "double"), touch in (otro, "double")


def lee_series(patron=None):
    """QUE DEVUELVE: todas las series de eventos y todas las filas por clip de la etapa 7, leidas
    del disco. Nunca de lo que se acaba de calcular: un agregado escrito desde memoria ya dejo una
    vez 40 filas de un clip donde tenia que haber 284 de once."""
    fs = sorted(glob.glob(str(RU.PISTA / (patron or "serie_eventos_*.csv"))))
    ev = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    fp = sorted(glob.glob(str(RU.PISTA / "posicion_clip_*.csv")))
    pc = pd.concat([pd.read_csv(f) for f in fp], ignore_index=True) if fp else pd.DataFrame()
    return ev, pc


def clip_tirador(ev, pc, man):
    """QUE DEVUELVE: una fila por (clip, lado) con resultado, etiquetas humanas y eventos."""
    man = man.set_index("clip_id")
    pc = pc.drop_duplicates("clip").set_index("clip") if len(pc) else pc
    filas = []
    for clip, e in ev.groupby("clip"):
        if clip not in man.index:
            continue
        m = man.loc[clip]
        su = float(pc.loc[clip, "s_utiles"]) if clip in pc.index else None
        pub = bool(pc.loc[clip, "publicable"]) if clip in pc.index else False
        for lado in (0, 1):
            el = e[e["lado"] == lado]
            valido = not (len(el) and (el["lado_valido"] == False).any())  # noqa: E712
            nombre = m["fotl_name"] if lado == 0 else m["fotr_name"]
            rival = m["fotr_name"] if lado == 0 else m["fotl_name"]
            marca, recibe = _resultado(m["touch"], lado)
            yo = "fotl" if lado == 0 else "fotr"
            ini = m["initiated"] if isinstance(m["initiated"], str) and m["initiated"] else None
            f = {"clip": clip, "lado": lado, "tirador": nombre, "rival": rival,
                 "torneo": m["torneo"], "arma": m["weapon"],
                 "marcador_propio": m["left_score"] if lado == 0 else m["right_score"],
                 "marcador_rival": m["right_score"] if lado == 0 else m["left_score"],
                 "marca": marca, "recibe": recibe, "doble": m["touch"] == "double",
                 "inicia": None if ini is None else ini == yo,
                 "zona_humana": m["strip_location"] or None,
                 "blanco_humano": m["body_location"] or None,
                 "lado_valido": valido, "s_utiles": su, "escala_publicable": pub}
            if valido and len(el):
                f["mano"] = el["mano"].dropna().mode().iat[0] if el["mano"].notna().any() else None
                f["guardia_m"] = _p50(el["guardia_m"])
                for t in TIPOS_PIES + ("brazo",):
                    et = el[el["tipo"] == t]
                    f["n_" + t] = int(len(et))
                    f[t + "_por_s"] = len(et) / su if su else None
                    f[t + "_dur_p50_s"] = _p50(et["duracion_s"])
                    if t != "brazo":
                        mm = et[et["unidad_magnitud"] == "m a lo largo de la pista"]
                        f[t + "_mag_p50_m"] = _p50(mm["magnitud_m"])
                        f[t + "_vel_p50_m_s"] = _p50(mm["velocidad_m_s"])
                f["brazo_ext_p50_torso"] = _p50(el.loc[el["tipo"] == "brazo", "ext_torso"])
                f["brazo_ext_max_torso"] = _max(el.loc[el["tipo"] == "brazo", "ext_torso"])
                f["frac_eventos_pose_fiable"] = float(el["pose_fiable"].fillna(False).mean())
                pies = el[el["tipo"].isin(TIPOS_PIES)].sort_values("cuadro_inicio")
                f["ultimo_gesto_pies"] = pies["tipo"].iat[-1] if len(pies) else None
            filas.append(f)
    return pd.DataFrame(filas)


def _perfil(g):
    """QUE DEVUELVE: el resumen de un grupo de filas (clip, lado) de UN tirador."""
    v = g[g["lado_valido"]]
    r = {"clips": int(len(g)), "lados_descartados": int((~g["lado_valido"]).sum()),
         "marca_frac": float(g["marca"].mean()), "recibe_frac": float(g["recibe"].mean()),
         "dobles": int(g["doble"].sum())}
    ini = g["inicia"].dropna()
    r["inicia_frac_humana"] = float(ini.mean()) if len(ini) else None
    r["n_inicia_anotados"] = int(len(ini))
    sa = g.loc[g["inicia"] == True]  # noqa: E712
    r["marca_cuando_inicia"] = float(sa["marca"].mean()) if len(sa) else None
    sn = g.loc[g["inicia"] == False]  # noqa: E712
    r["marca_cuando_no_inicia"] = float(sn["marca"].mean()) if len(sn) else None
    z = g.loc[g["marca"] & g["zona_humana"].notna(), "zona_humana"]
    for k in CA.ZONAS:
        r["marca_en_" + k] = int((z == k).sum())
    b = g.loc[g["marca"] & g["blanco_humano"].notna(), "blanco_humano"]
    r["blanco_mas_tocado_al_rival"] = b.mode().iat[0] if len(b) else None
    b2 = g.loc[g["recibe"] & g["blanco_humano"].notna(), "blanco_humano"]
    r["blanco_mas_recibido"] = b2.mode().iat[0] if len(b2) else None
    if len(v):
        su = v["s_utiles"].sum()
        r["s_utiles_total"] = float(su)
        for t in TIPOS_PIES + ("brazo",):
            n = v.get("n_" + t)
            r["n_" + t] = int(n.sum()) if n is not None else 0
            r[t + "_por_s"] = float(n.sum() / su) if n is not None and su else None
            r[t + "_dur_p50_s"] = _p50(v.get(t + "_dur_p50_s", pd.Series(dtype=float)))
            if t != "brazo":
                r[t + "_mag_p50_m"] = _p50(v.get(t + "_mag_p50_m", pd.Series(dtype=float)))
        for c in ("guardia_m", "brazo_ext_p50_torso", "brazo_ext_max_torso"):
            r[c] = _p50(v.get(c, pd.Series(dtype=float)))
        r["frac_escala_publicable"] = float(v["escala_publicable"].mean())
        u = v["ultimo_gesto_pies"].dropna() if "ultimo_gesto_pies" in v else pd.Series()
        for t in TIPOS_PIES:
            r["acaba_en_" + t + "_frac"] = float((u == t).mean()) if len(u) else None
    return pd.Series(r)


def _agrupa(ct, claves):
    """QUE DEVUELVE: una fila por grupo con su perfil. Se construye fila a fila porque los perfiles
    no traen todos las mismas columnas (un tirador sin lados validos no tiene eventos)."""
    filas = []
    for k, g in ct.groupby(claves):
        k = k if isinstance(k, tuple) else (k,)
        filas.append({**dict(zip(claves, k)), **_perfil(g).to_dict()})
    return pd.DataFrame(filas)


def asalto_id(df):
    """QUE DEVUELVE: un identificador de asalto: torneo + la pareja ORDENADA de nombres. Dos
    tiradores que se cruzan dos veces en el mismo torneo (poule y directa) quedan juntos: es un
    limite declarado, el catalogo no trae la ronda."""
    par = df.apply(lambda r: " vs ".join(sorted([str(r["tirador"]), str(r["rival"])])), axis=1)
    return df["torneo"].astype(str) + " | " + par


def construye(verboso=True):
    ev, pc = lee_series()
    man = CA.manifiesto()
    ct = clip_tirador(ev, pc, man)
    if not len(ct):
        print("sin series de eventos en", RU.PISTA)
        return None
    ct["asalto"] = asalto_id(ct)
    ti = _agrupa(ct, ["tirador"])
    fie = RU.CATALOGO / "mano_fie.csv"
    if fie.exists():
        mf = pd.read_csv(fie).drop_duplicates("nombre_catalogo")
        mf["clave"] = mf["nombre_catalogo"].str.upper()
        ti["clave"] = ti["tirador"].astype(str).str.upper()
        ti = ti.merge(mf[["clave", "fie_id", "pais", "hand", "genero", "altura"]], on="clave",
                      how="left").drop(columns="clave").rename(columns={"hand": "mano_fie"})
    asl = _agrupa(ct, ["asalto", "tirador"])
    mx = ct.groupby(["asalto", "tirador"])["marcador_propio"].max().rename("marcador_final_propio")
    asl = asl.merge(mx.reset_index(), on=["asalto", "tirador"], how="left")
    SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre, d in (("clip_tirador", ct), ("tiradores", ti), ("asaltos", asl)):
        tmp = SALIDA / (nombre + ".csv.part")
        d.to_csv(tmp, index=False)
        tmp.replace(SALIDA / (nombre + ".csv"))
    if verboso:
        print("clips con series: %d | filas clip-tirador: %d (validas %d) | tiradores: %d | "
              "asaltos: %d" % (ct["clip"].nunique(), len(ct), int(ct["lado_valido"].sum()),
                               len(ti), ct["asalto"].nunique()))
        print("escrito en", SALIDA)
    return ct, ti, asl


if __name__ == "__main__":
    construye()
