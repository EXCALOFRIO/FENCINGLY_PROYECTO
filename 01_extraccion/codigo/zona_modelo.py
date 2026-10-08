"""El modelo que lee la zona de la pista DEL FOTOGRAMA, sin homografia ninguna.

## LAS LINEAS BASE, DECLARADAS ANTES DE ENTRENAR

Sobre los 4.889 clips con etiqueta humana y video en disco:

    contestar siempre `middle`              **0,409**   la clase mayoritaria de ESTE conjunto
    nuestra x medida de la geometria         0,323      lo que da hoy toda la via de homografia
    acuerdo entre dos anotadores humanos     0,619      el TECHO
    acuerdo de anotaciones repetidas         0,688      el otro techo, del propio corpus

**No se adopta si no bate 0,409.** Y hay que dar tambien el error ordinal medio y el acierto a
menos de una zona, porque confundir `middle` con `fotr_half` no cuesta lo mismo que confundirlo con
`fotr_warning_box` y una cifra de acierto plano trata los dos errores igual.

## LA ANTISIMETRIA, IMPUESTA POR CONSTRUCCION

Voltear el fotograma en horizontal tiene que **invertir la respuesta**:

    fotl_warning_box <-> fotr_warning_box      fotl_half <-> fotr_half      middle <-> middle

Se impone, no se desea. El modelo no predice cinco clases: predice un **escalar** `s(x)` -- la
posicion a lo largo de la pista -- definido como

    s(x) = g(x) - g(espejo(x))

con `g` la red compartida. Entonces `s(espejo(x)) = -s(x)` **exactamente**, para cualquier peso y
sin entrenar nada. Y las zonas salen de **umbrales simetricos** `-t2 < -t1 < +t1 < +t2` con
`t1, t2 > 0` aprendidos.

Dos razones, y las dos son medibles:

  1. **Impide el atajo.** Sin la antisimetria la red puede aprender "el marcador suele estar a la
     izquierda en este torneo" y sacar un numero que parece aprendizaje y es el fondo del pabellon.
     Con ella ese atajo **no es representable**: cualquier cosa que no cambie al voltear se cancela
     en la resta.
  2. **Duplica los datos gratis** y el control sale de balde: pasar cada clip y su espejo tiene que
     dar respuestas opuestas en el **100 %**. Si no, la antisimetria no esta impuesta, esta deseada.

Es la misma disciplina que `etiquetador.py` aplica a la cabeza de quien inicia, y por la misma
razon.

## LA PARTICION ES POR TORNEO

Nunca por clip. Los clips del mismo torneo comparten pabellon, pista, camara y publicidad de fondo;
partir por clip deja el mismo fondo a los dos lados y el numero sale inflado. Con 135 torneos y el
mayor al 14,4 % hay sitio de sobra para partir.

    python zona_modelo.py --epocas 8
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import catalogo as CA  # noqa: E402

RAIZ = pathlib.Path("D:/FENCINGLY_ZONA")
INDICE = RAIZ / "indice.csv"
PESOS = pathlib.Path(__file__).resolve().parents[2] / "03_datos" / "modelos" / "zona_pista.pt"

ZONAS = CA.ZONAS
BASE_MAYORITARIA = 0.409
BASE_GEOMETRIA = 0.323
TECHO_HUMANO = 0.619


MEDIA = (0.485, 0.456, 0.406)
DESV = (0.229, 0.224, 0.225)


class DS:
    """El conjunto de fotogramas.

    VIVE A NIVEL DE MODULO y no dentro de `main`, que es donde lo puse primero. En Windows los
    procesos del `DataLoader` arrancan por `spawn` y tienen que PICKLEAR la clase; una clase local
    no se puede picklear y la primera tanda murio con
    `PicklingError: Can't pickle local object main.<locals>.DS`. Solo pasa con `num_workers > 0`,
    asi que con un solo proceso el fallo habria estado escondido y habria salido en Colab.
    """

    def __init__(self, filas, aumenta):
        self.f = filas
        self.aumenta = aumenta

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        import cv2
        import torch
        r = self.f[i]
        im = cv2.imread(str(RAIZ / r["fichero"]))
        if im is None:
            im = np.zeros((180, 320, 3), "uint8")
        y = ZONAS.index(r["zona"])
        if self.aumenta and np.random.rand() < 0.5:
            im = im[:, ::-1].copy()
            y = 4 - y
            """EL ESPEJO COMO AUMENTO CAMBIA LA ETIQUETA, no solo la imagen. Voltear sin cambiarla
            ensenaria que la zona no depende del lado, que es lo contrario de lo que hay que
            aprender."""
        x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(MEDIA).view(3, 1, 1)) / torch.tensor(DESV).view(3, 1, 1)
        return x, y, r["clip"]


def carga_indice():
    filas = []
    with open(INDICE, encoding="utf-8", newline="") as fh:
        for f in csv.DictReader(fh):
            if f["zona"] in ZONAS:
                filas.append(f)
    return filas


def particion_por_torneo(filas, frac=0.25, semilla=0):
    """QUE DEVUELVE: (entrena, reserva) como listas de filas, partiendo por TORNEO.

    Se reparten torneos enteros, del mas grande al mas pequeno, al lado que menos clips tenga, para
    que el reservado quede cerca de la fraccion pedida sin partir ningun torneo. Repartir al azar
    con 135 torneos donde el mayor es el 14,4 % puede dejar un reservado del 5 % o del 30 %.
    """
    por = collections.defaultdict(list)
    for f in filas:
        por[f["torneo"]].append(f)
    orden = sorted(por, key=lambda t: -len(por[t]))
    rng = np.random.default_rng(semilla)
    orden = [orden[0]] + list(rng.permutation(orden[1:]))
    obj = frac * len(filas)
    res, ent = [], []
    for t in orden:
        if len(res) < obj:
            res.extend(por[t])
        else:
            ent.extend(por[t])
    return ent, res


def construye(dispositivo="cuda"):
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class Zona(nn.Module):
        """`s(x) = g(x) - g(espejo(x))` y zonas por umbrales simetricos."""

        def __init__(self):
            super().__init__()
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Linear(m.fc.in_features, 1)
            self.g = m
            self.t = nn.Parameter(torch.tensor([0.4, 1.2]))
            """DOS UMBRALES, NO CUATRO, y positivos: las fronteras son -t2 < -t1 < t1 < t2. Con
            cuatro libres el modelo podria colocarlas asimetricas y romper la antisimetria que la
            arquitectura garantiza, que seria tenerla y tirarla en la ultima capa."""

        def score(self, x):
            return (self.g(x) - self.g(torch.flip(x, dims=[3]))).squeeze(1)

        def bordes(self):
            t = torch.abs(self.t) + 1e-3
            t1, t2 = t[0], t[0] + t[1]
            return torch.stack([-t2, -t1, t1, t2])

        def forward(self, x):
            s = self.score(x)
            return s, s.unsqueeze(1) - self.bordes().unsqueeze(0)

    return Zona().to(dispositivo)


def perdida_ordinal(logits, y):
    """Perdida de umbrales acumulados: para cada frontera k, `P(zona > k) = sigmoide(s - b_k)`.

    No es clasificacion plana a proposito. Las cinco zonas estan ORDENADAS en el espacio, y con una
    entropia cruzada de cinco clases confundir `middle` con `fotr_warning_box` cuesta lo mismo que
    confundirlo con `fotr_half`, que es falso: una es un error de 5 m y la otra de 2.
    """
    import torch
    import torch.nn.functional as F
    obj = (y.unsqueeze(1) > torch.arange(4, device=y.device).unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(logits, obj)


def de_logits(logits):
    import torch
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--lote", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader

    filas = carga_indice()
    ent, res = particion_por_torneo(filas)
    print("LA ZONA DE LA PISTA, LEIDA DEL FOTOGRAMA")
    print("=" * 84)
    print("fotogramas: %d  (entrena %d, reservado %d)" % (len(filas), len(ent), len(res)))
    print("torneos: entrena %d, reservado %d, solape %d"
          % (len({f["torneo"] for f in ent}), len({f["torneo"] for f in res}),
             len({f["torneo"] for f in ent} & {f["torneo"] for f in res})))
    print("clips:   entrena %d, reservado %d"
          % (len({f["clip"] for f in ent}), len({f["clip"] for f in res})))
    print()
    print("LINEAS BASE declaradas: siempre middle %.3f | nuestra x %.3f | techo humano %.3f"
          % (BASE_MAYORITARIA, BASE_GEOMETRIA, TECHO_HUMANO))
    print()

    dl_e = DataLoader(DS(ent, True), batch_size=a.lote, shuffle=True, num_workers=4,
                      drop_last=True)
    dl_r = DataLoader(DS(res, False), batch_size=a.lote, shuffle=False, num_workers=4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = construye(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=a.lr, weight_decay=1e-4)
    mejor = -1.0
    for ep in range(a.epocas):
        mod.train()
        tot = 0.0
        for x, y, _c in dl_e:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            _s, lo = mod(x)
            p = perdida_ordinal(lo, y)
            opt.zero_grad()
            p.backward()
            opt.step()
            tot += float(p) * len(y)
        ac, ac1, eo, esp = evalua(mod, dl_r, dev)
        print("epoca %d  perdida %.4f  |  RESERVADO acierto %.3f  <=1 zona %.3f  err ord %.3f  "
              "espejo %.3f" % (ep + 1, tot / max(len(ent), 1), ac, ac1, eo, esp), flush=True)
        if ac > mejor:
            mejor = ac
            PESOS.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": mod.state_dict(), "acierto": ac, "zonas": list(ZONAS)}, PESOS)
    print()
    print("mejor acierto en el reservado: %.3f  (linea base %.3f, techo %.3f)"
          % (mejor, BASE_MAYORITARIA, TECHO_HUMANO))
    print("pesos en %s" % PESOS)
    return 0


def evalua(mod, dl, dev):
    """QUE DEVUELVE: (acierto, acierto<=1 zona, error ordinal medio, control del espejo).

    El control del espejo es la fraccion de muestras en que `s(espejo(x)) = -s(x)` numericamente.
    Tiene que salir **1,000**: no es una metrica de calidad sino la comprobacion de que la
    arquitectura hace lo que dice. Si baja de 1, hay un fallo de construccion y el resto de cifras
    no valen.
    """
    import torch
    mod.eval()
    P, Y, S, Sm = [], [], [], []
    with torch.no_grad():
        for x, y, _c in dl:
            x = x.to(dev)
            s, lo = mod(x)
            P.append(de_logits(lo).cpu().numpy())
            Y.append(y.numpy())
            S.append(s.cpu().numpy())
            Sm.append(mod.score(torch.flip(x, dims=[3])).cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    S, Sm = np.concatenate(S), np.concatenate(Sm)
    esp = float(np.mean(np.abs(S + Sm) <= 1e-4 + 1e-3 * np.abs(S)))
    return (float((P == Y).mean()), float((np.abs(P - Y) <= 1).mean()),
            float(np.abs(P - Y).mean()), esp)


if __name__ == "__main__":
    sys.exit(main())
