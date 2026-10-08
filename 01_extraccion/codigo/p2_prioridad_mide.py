"""MIDE el detector de prioridad contra la anotacion a mano, contra la linea base y contra si mismo.

CUATRO MEDIDAS, y las cuatro estaban declaradas antes de correr esto:

  1. ACIERTO POR CUADRO contra la anotacion a mano de 21 ventanas. La unidad de comparacion es el
     cuadro DENTRO de las ventanas anotadas, no el clip entero, porque fuera de las ventanas no hay
     anotacion y contar esos cuadros como acierto seria regalarse el 90 % del corpus en «ninguna».
  2. LA LINEA BASE TONTA, «tiene prioridad el que empieza a avanzar primero» con la x de la caja y
     nada mas. Si el detector de 308 puntos no la bate, se dice.
  3. EL INSTANTE DEL CAMBIO, con tolerancia de +-3 cuadros, que es la resolucion de la anotacion: las
     laminas van de 3 en 3 cuadros y los de en medio no se han visto. Dar el instante exacto seria
     fingir una precision que la anotacion no tiene.
  4. LOS CAMBIOS POR SEGUNDO con y sin intervalo de gracia, contra los que hay de verdad en la
     anotacion. Es la medida que justifica el margen; sin ella `PERMANENCIA` seria inventado.

QUIEN HIZO LA ANOTACION, dicho claro: la hizo este agente mirando laminas de fotogramas, **no un
arbitro FIE**. Cuatro de las 21 ventanas llevan confianza «media» o «baja» y una lleva «SIN DATO».
El acierto contra esta referencia mide acuerdo con un aficionado atento, no correccion reglamentaria.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import p1_variables as P1
import prioridad as PR

PRUEBAS = P1.PRUEBAS
SALIDA = P1.SALIDA
COD = {"ninguna": -1, "izquierda": 0, "derecha": 1, "": -1}
TOL = 3


def carga_anotacion(fich="anotacion_a_mano.csv"):
    filas = []
    with open(SALIDA / fich, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["clip"]:
                continue
            r["ini"], r["fin"] = int(r["ini"]), int(r["fin"])
            r["fps"] = float(r["fps"])
            filas.append(r)
    return filas


def verdad_por_cuadro(r):
    """QUE HACE: expande una fila de la anotacion a la etiqueta por cuadro de su ventana."""
    y = {}
    act = COD[r["prio_ini"]]
    for k in range(r["ini"], r["fin"] + 1):
        for i in (1, 2):
            c = r.get("cambio_%d" % i) or ""
            if c and int(c) == k:
                act = COD[r.get("prio_%d" % i) or "ninguna"]
        y[k] = act
    return y


def cambios_de(prio, ini, fin):
    """Cuadros donde la etiqueta cambia, dentro de [ini, fin]."""
    out = []
    for k in range(ini + 1, fin + 1):
        if prio[k] != prio[k - 1]:
            out.append((k, int(prio[k])))
    return out


def main():
    an = carga_anotacion()
    clips = sorted({r["clip"] for r in an})
    res_por_clip = {}
    print("=== 1 y 2 · ACIERTO POR CUADRO en las ventanas anotadas ===")
    print("%-38s %5s %6s %6s %6s" % ("clip", "cuadros", "detec", "base", "trivial"))
    tot = {"n": 0, "det": 0, "base": 0, "triv": 0}
    detalle = []
    for cid in clips:
        with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        s = PR.senales(d)
        mask, T = d["mask"], int(d["T"])
        r = PR.decide(s, mask, T)
        rb, _q = PR.linea_base(d["B"], d["pres"], mask, T)
        rsg = PR.decide_sin_gracia(s, mask, T)
        par = PR.paradas(s, mask, T, r)
        res_por_clip[cid] = {"s": s, "r": r, "rb": rb, "rsg": rsg, "par": par,
                             "mask": mask, "T": T, "d": d}
        n = det = base = triv = 0
        for fila in [f for f in an if f["clip"] == cid]:
            y = verdad_por_cuadro(fila)
            for k, v in y.items():
                if k >= T or not mask[k]:
                    continue
                n += 1
                det += int(r["prio"][k] == v)
                base += int(rb[k] == v)
                triv += int(v == -1)
                detalle.append({"clip": cid, "cuadro": k, "verdad": v,
                                "detector": int(r["prio"][k]),
                                "base": int(rb[k]), "conf": float(r["conf"][k]),
                                "regla": r["regla"][k], "confianza_anotacion": fila["confianza"]})
        if n:
            print("%-38s %7d %5.0f%% %5.0f%% %5.0f%%" % (cid[:38], n, 100 * det / n,
                                                         100 * base / n, 100 * triv / n))
            tot["n"] += n
            tot["det"] += det
            tot["base"] += base
            tot["triv"] += triv
    print("%-38s %7d %5.1f%% %5.1f%% %5.1f%%" % ("TOTAL", tot["n"], 100 * tot["det"] / tot["n"],
                                                 100 * tot["base"] / tot["n"],
                                                 100 * tot["triv"] / tot["n"]))
    print()
    print('"trivial" es responder SIEMPRE «ninguna». Es la linea base mas tonta posible y hay que')
    print("batirla tambien, porque 12 de las 21 ventanas anotadas son «ninguna» de principio a fin.")
    print()
    print("=== 2b · EL MISMO ACIERTO PARTIDO EN DOS, que es lo que de verdad informa ===")
    print("El numero agrupado esta dominado por las ventanas sin accion: 12 de 21 son «ninguna» de")
    print("principio a fin, o sea el 73,8 % de los cuadros. Un detector que no dispare nunca saca ese")
    print("73,8 % sin saber nada, y uno que dispare paga en esas ventanas. Partido:")
    con = [x for x in detalle if x["verdad"] >= 0]
    sin = [x for x in detalle if x["verdad"] < 0]
    print("  ventanas CON dueno anotado: %d cuadros, acierto detector %.1f %%, base %.1f %%, trivial 0 %%" % (
        len(con), 100 * np.mean([x["detector"] == x["verdad"] for x in con]),
        100 * np.mean([x["base"] == x["verdad"] for x in con])))
    print("  ventanas SIN dueno anotado: %d cuadros, acierto detector %.1f %%, base %.1f %%, trivial 100 %%" % (
        len(sin), 100 * np.mean([x["detector"] == x["verdad"] for x in sin]),
        100 * np.mean([x["base"] == x["verdad"] for x in sin])))
    amb = [x for x in con if x["detector"] >= 0]
    if amb:
        print("  y cuando LOS DOS nombran dueno (%d cuadros), el LADO coincide en %.1f %%" % (
            len(amb), 100 * np.mean([x["detector"] == x["verdad"] for x in amb])))
    print("  falsa alarma: cuadros sin dueno anotado donde el detector inventa uno: %.1f %%" % (
        100 * np.mean([x["detector"] >= 0 for x in sin])))
    print()
    print("=== 2c · CUANTO CUESTA LA CONDICION DE ORDEN DE t.83.2.a ===")
    print("Es la condicion mas dudosa del detector, porque `pies` no separa el a fondo de la marcha.")
    print("Apagarla no es la solucion —seria dejar de aplicar la regla— pero SI localiza el problema.")
    for nom, v in (("con la condicion (como esta)", True), ("sin la condicion", False)):
        PR.EXIGE_ORDEN_FONDO = v
        try:
            n = d1 = nc2 = d2 = 0
            for cid in clips:
                R = res_por_clip[cid]
                rr = PR.decide(R["s"], R["mask"], R["T"])
                for fila in [f for f in an if f["clip"] == cid]:
                    for k, ver in verdad_por_cuadro(fila).items():
                        if k >= R["T"] or not R["mask"][k]:
                            continue
                        n += 1
                        d1 += int(rr["prio"][k] == ver)
                        if ver >= 0:
                            nc2 += 1
                            d2 += int(rr["prio"][k] == ver)
        finally:
            PR.EXIGE_ORDEN_FONDO = True
        print("  %-30s acierto global %.1f %%   en ventanas CON dueno %.1f %%" % (
            nom, 100.0 * d1 / n, 100.0 * d2 / max(nc2, 1)))
    print()
    print("=== 3 · EL INSTANTE DEL CAMBIO, tolerancia +-%d cuadros ===" % TOL)
    ok_c, tot_c, falsos = 0, 0, 0
    for fila in an:
        cid = fila["clip"]
        if cid not in res_por_clip:
            continue
        R = res_por_clip[cid]
        esperados = []
        for i in (1, 2):
            c = fila.get("cambio_%d" % i) or ""
            if c:
                esperados.append((int(c), COD[fila.get("prio_%d" % i) or "ninguna"]))
        vistos = cambios_de(R["r"]["prio"], fila["ini"], min(fila["fin"], R["T"] - 1))
        usados = set()
        for k, v in esperados:
            tot_c += 1
            hit = [j for j, (kk, vv) in enumerate(vistos)
                   if j not in usados and abs(kk - k) <= TOL and vv == v]
            if hit:
                usados.add(hit[0])
                ok_c += 1
                print("  ACIERTA  %-34s esperado c%d->%d  detectado c%d" % (
                    cid[:34], k, v, vistos[hit[0]][0]))
            else:
                print("  FALLA    %-34s esperado c%d->%d  detectado %s" % (
                    cid[:34], k, v, vistos if vistos else "nada"))
        falsos += len([j for j in range(len(vistos)) if j not in usados])
    print("  cambios anotados %d, acertados %d (%.0f %%), cambios detectados de mas %d" % (
        tot_c, ok_c, 100.0 * ok_c / max(tot_c, 1), falsos))
    print()
    print("=== 4 · CAMBIOS POR SEGUNDO, con y sin intervalo de gracia ===")
    print("%-38s %8s %8s %8s" % ("clip", "con", "sin", "segundos"))
    tc = ts = seg = 0.0
    for cid in clips:
        R = res_por_clip[cid]
        fps = float([f for f in an if f["clip"] == cid][0]["fps"])
        u = int(R["mask"].sum())
        if not u:
            continue
        c1 = len(cambios_de(R["r"]["prio"], 0, R["T"] - 1))
        c2 = len(cambios_de(R["rsg"]["prio"], 0, R["T"] - 1))
        sg = u / fps
        print("%-38s %8.2f %8.2f %8.1f" % (cid[:38], c1 / sg, c2 / sg, sg))
        tc += c1
        ts += c2
        seg += sg
    nc = sum(1 for f in an for i in (1, 2) if (f.get("cambio_%d" % i) or ""))
    sa = sum((f["fin"] - f["ini"] + 1) / f["fps"] for f in an)
    print("%-38s %8.2f %8.2f %8.1f" % ("TOTAL", tc / seg, ts / seg, seg))
    print("  ANOTADO A MANO: %d cambios en %.1f s = %.2f cambios por segundo" % (nc, sa, nc / sa))
    print()
    print("=== 4b · QUE APORTA CADA MECANISMO CONTRA LA OSCILACION, apagandolos uno a uno ===")
    print("Sin esta tabla, `PERMANENCIA` y la histeresis serian dos numeros inventados.")
    variantes = [("completo", {}),
                 ("sin intervalo de gracia", {"PERMANENCIA": 0}),
                 ("sin histeresis (un solo umbral)", {"EXT_SIGUE": PR.EXT_DISPARA,
                                                      "PIES_SIGUE": PR.PIES_DISPARA}),
                 ("sin filtro de amplitud/altura", {"EXT_ALTO": 0.0, "EXT_AMPL": 0.0,
                                                    "PIES_ALTO": 0.0, "PIES_AMPL": 0.0}),
                 ("sin fusion de mesetas", {"RETRAE": -1.0}),
                 ("sin nada de lo anterior", {"PERMANENCIA": 0, "EXT_SIGUE": PR.EXT_DISPARA,
                                              "PIES_SIGUE": PR.PIES_DISPARA, "EXT_ALTO": 0.0,
                                              "EXT_AMPL": 0.0, "PIES_ALTO": 0.0,
                                              "PIES_AMPL": 0.0, "RETRAE": -1.0})]
    print("%-34s %10s %10s" % ("variante", "cambios/s", "x lo real"))
    for nom, ajustes in variantes:
        viejos = {k: getattr(PR, k) for k in ajustes}
        for k, v in ajustes.items():
            setattr(PR, k, v)
        try:
            c = 0.0
            for cid in clips:
                R = res_por_clip[cid]
                rr = PR.decide(R["s"], R["mask"], R["T"])
                c += len(cambios_de(rr["prio"], 0, R["T"] - 1))
        finally:
            for k, v in viejos.items():
                setattr(PR, k, v)
        print("%-34s %10.2f %10.1f" % (nom, c / seg, (c / seg) / (nc / sa)))
    print()
    print("=== 5 · PARADAS: candidatos del criterio de t.86 ===")
    npar = 0
    with open(SALIDA / "paradas_candidatas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "cuadro", "paraba", "atacaba", "manos", "umbral_manos", "caida_rel"])
        for cid in clips:
            for c in res_por_clip[cid]["par"]:
                w.writerow([cid, c["cuadro"], c["paraba"], c["atacaba"],
                            "%.3f" % c["manos"], "%.3f" % c["umbral_manos"],
                            "%.3f" % c["caida_rel"]])
                print("  %-34s c%-4d para el %d, atacaba el %d, manos %.2f (umbral %.2f)" % (
                    cid[:34], c["cuadro"], c["paraba"], c["atacaba"], c["manos"],
                    c["umbral_manos"]))
                npar += 1
    print("  candidatos: %d en %d clips" % (npar, len(clips)))
    print("  PARADAS ANOTADAS A MANO EN LAS 21 VENTANAS: %d" % sum(
        1 for f in an if f["parada"] == "si"))
    print()
    print("=== 6 · DONDE FALLA: cuadros mal clasificados, agrupados por regla ===")
    mal = [x for x in detalle if x["detector"] != x["verdad"]]
    por = {}
    for x in mal:
        por.setdefault(x["regla"][:64], []).append(x)
    for r, xs in sorted(por.items(), key=lambda kv: -len(kv[1])):
        print("  %4d cuadros  %s" % (len(xs), r))
        print("        ejemplo: %s c%d verdad %d detector %d conf %.2f" % (
            xs[0]["clip"][:30], xs[0]["cuadro"], xs[0]["verdad"], xs[0]["detector"], xs[0]["conf"]))
    print()
    print("=== 7 · LA CONFIANZA, ¿separa acierto de fallo? ===")
    ok = np.array([x["conf"] for x in detalle if x["detector"] == x["verdad"]])
    ko = np.array([x["conf"] for x in detalle if x["detector"] != x["verdad"]])
    if len(ok) and len(ko):
        print("  conf mediana en aciertos %.3f, en fallos %.3f" % (
            float(np.median(ok)), float(np.median(ko))))
        print("  criterio declarado: separa si la mediana de aciertos es >= 1,2x la de fallos")
        v = float(np.median(ok)) >= 1.2 * max(float(np.median(ko)), 1e-9)
        print("  veredicto: %s" % ("SEPARA" if v else "NO SEPARA, y se dice"))
    print()
    print("=== 8 · LOTE DE CONTROL: clips que NO se miraron al arreglar el detector ===")
    print("Es la unica defensa contra haberme ajustado a las 21 ventanas. Los arreglos de definicion")
    print("se encontraron mirando la senal en 5 de esas ventanas, asi que su acierto esta inflado.")
    print("Estas ventanas se anotaron DESPUES de congelar el codigo y sobre clips sin tocar.")
    ho = carga_anotacion("anotacion_holdout.csv")
    n = d1 = nc2 = d2 = 0
    for fila in ho:
        cid = fila["clip"]
        with np.load(PRUEBAS / ("poses_%s.npz" % cid), allow_pickle=True) as z:
            dd = {k: z[k] for k in z.files}
        s2 = PR.senales(dd)
        m2, T2 = dd["mask"], int(dd["T"])
        r2 = PR.decide(s2, m2, T2)
        for k, ver in verdad_por_cuadro(fila).items():
            if k >= T2 or not m2[k]:
                continue
            n += 1
            d1 += int(r2["prio"][k] == ver)
            if ver >= 0:
                nc2 += 1
                d2 += int(r2["prio"][k] == ver)
        vistos = cambios_de(r2["prio"], fila["ini"], min(fila["fin"], T2 - 1))
        print("  %-30s c%d-c%d  cambios anotados %s  detectados %s" % (
            cid[:30], fila["ini"], fila["fin"],
            [(fila.get("cambio_%d" % i), fila.get("prio_%d" % i)) for i in (1, 2)
             if fila.get("cambio_%d" % i)], vistos))
    if n:
        print("  ventanas de control: %d cuadros, acierto global %.1f %%, en cuadros CON dueno %.1f %%"
              % (n, 100.0 * d1 / n, 100.0 * d2 / max(nc2, 1)))
    with open(SALIDA / "p2_por_cuadro.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detalle[0].keys()))
        w.writeheader()
        w.writerows(detalle)
    print()
    print("tabla por cuadro -> %s" % (SALIDA / "p2_por_cuadro.csv"))


if __name__ == "__main__":
    main()
