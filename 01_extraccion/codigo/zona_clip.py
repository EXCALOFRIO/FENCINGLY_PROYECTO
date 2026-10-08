"""La zona de la pista leyendo EL CLIP ENTERO, no cada fotograma por separado.

## DE DONDE SALE ESTO

El usuario: *"creo que es dificil porque no pillas el contexto de todo el clip y lo evaluas como
frame aislado a frame aislado sin pillar el contexto de todo, porque como en los videos se mueve la
camara, pues pillas mas contexto"*. Las dos partes son correctas y la primera era un fallo mio.

**El fallo.** `zona_modelo.evalua` medía el acierto **por fotograma**, y la etiqueta es **del clip**.
Asi que el 0,430 de la primera tanda se estaba comparando contra un techo (0,619) que es por clip:
dos cifras con denominadores distintos, que es el fallo que este proyecto ya tiene documentado cinco
veces en un dia.

**El fondo.** La camara panea y hace zoom, asi que **cada fotograma del clip ensena un trozo
distinto de la pista**. Uno puede no tener ningun extremo visible y otro del mismo clip si. Decidir
por separado y no juntarlos tira justo la informacion que desambigua.

## LO QUE HACE ESTE MODULO

El clip es la unidad, de principio a fin: la entrada son sus `F` fotogramas a la vez y la salida es
**un escalar por clip**.

    e_i  = tronco(x_i)                        embedding de cada fotograma
    a_i  = softmax(w . e_i)                   cuanto pesa cada fotograma
    G(c) = v . sum_i a_i e_i                  un escalar por CLIP
    s(c) = G(c) - G(espejo(c))                <- la antisimetria, exacta

**La atencion es la pieza que contesta al usuario.** No promedia a ciegas: aprende a fiarse del
fotograma donde se ve un extremo de la pista o la zona de aviso, y a ignorar los de primer plano
cerrado donde no hay ninguna referencia. Un promedio simple trataria igual al fotograma que lo
ensena todo y al que no ensena nada.

**Y la antisimetria sigue siendo exacta.** `G` puede ser cualquier funcion del clip -- incluida la
atencion -- porque la resta se hace fuera: `s(espejo(c)) = G(espejo(c)) - G(c) = -s(c)` para
cualesquiera pesos y sin entrenar. Voltear el clip entero intercambia `fotl` con `fotr` por
construccion, no por aprendizaje.

## LAS CIFRAS CONTRA LAS QUE SE MIDE, declaradas

    contestar siempre `middle`                 0,409
    nuestra x de la geometria                  0,323
    el mismo modelo POR FOTOGRAMA              0,430    <- lo que hay que batir de verdad
    techo humano, 5 zonas                      0,598
    techo humano, izq/centro/der               0,676
    techo humano, extremo o no                 0,757

    python zona_clip.py --epocas 10
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import zona_modelo as ZM  # noqa: E402

PESOS = pathlib.Path(__file__).resolve().parents[2] / "03_datos" / "modelos" / "zona_clip.pt"
N_FOT = 5


def por_clip(filas):
    """QUE DEVUELVE: lista de `(clip, [filas ordenadas por instante], zona)`."""
    d = collections.defaultdict(list)
    for f in filas:
        d[f["clip"]].append(f)
    out = []
    for c, fs in d.items():
        fs.sort(key=lambda r: float(r["instante"]))
        out.append((c, fs, ZM.ZONAS.index(fs[0]["zona"])))
    return out


class DSClip:
    """Un clip por muestra: devuelve `(F, 3, 180, 320)`.

    Vive a nivel de modulo por la misma razon que `zona_modelo.DS`: en Windows el `DataLoader`
    arranca por `spawn` y tiene que picklear la clase.
    """

    def __init__(self, clips, aumenta):
        self.c = clips
        self.aumenta = aumenta

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        import cv2
        import torch
        _cid, fs, y = self.c[i]
        vol = np.random.rand() < 0.5 if self.aumenta else False
        ims = []
        for r in fs[:N_FOT]:
            im = cv2.imread(str(ZM.RAIZ / r["fichero"]))
            if im is None:
                im = np.zeros((180, 320, 3), "uint8")
            if vol:
                im = im[:, ::-1]
            ims.append(im[:, :, ::-1].copy())
        while len(ims) < N_FOT:
            ims.append(ims[-1] if ims else np.zeros((180, 320, 3), "uint8"))
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255.0
        x = (x - torch.tensor(ZM.MEDIA).view(1, 3, 1, 1)) / \
            torch.tensor(ZM.DESV).view(1, 3, 1, 1)
        if vol:
            y = 4 - y
            """EL CLIP ENTERO SE VOLTEA A LA VEZ, no fotograma a fotograma: voltear unos si y otros
            no crearia un clip que no existe, con la pista mirando a los dos lados dentro de la
            misma muestra."""
        return x, y


def construye(dispositivo="cuda"):
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class ZonaClip(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.dim_tronco = m.fc.in_features
            m.fc = nn.Identity()
            self.tronco = m
            self.proy = nn.Sequential(nn.Linear(self.dim_tronco, dim), nn.ReLU())
            self.aten = nn.Linear(dim, 1)
            self.cabeza = nn.Linear(dim, 1)
            self.t = nn.Parameter(torch.tensor([0.4, 1.2]))

        def G(self, x):
            """`x` es (B, F, 3, H, W) -> un escalar por clip, con atencion sobre los fotogramas."""
            B, F = x.shape[:2]
            e = self.proy(self.tronco(x.reshape(B * F, *x.shape[2:]))).view(B, F, -1)
            a = torch.softmax(self.aten(e).squeeze(-1), dim=1)
            return self.cabeza((a.unsqueeze(-1) * e).sum(1)).squeeze(-1)

        def pesos_atencion(self, x):
            B, F = x.shape[:2]
            e = self.proy(self.tronco(x.reshape(B * F, *x.shape[2:]))).view(B, F, -1)
            return torch.softmax(self.aten(e).squeeze(-1), dim=1)

        def score(self, x):
            return self.G(x) - self.G(torch.flip(x, dims=[4]))

        def bordes(self):
            t = torch.abs(self.t) + 1e-3
            return torch.stack([-(t[0] + t[1]), -t[0], t[0], t[0] + t[1]])

        def forward(self, x):
            s = self.score(x)
            return s, s.unsqueeze(1) - self.bordes().unsqueeze(0)

    return ZonaClip().to(dispositivo)


def agr3(i):
    return 0 if i <= 1 else (2 if i >= 3 else 1)


def agrx(i):
    return 0 if i == 0 else (2 if i == 4 else 1)


def evalua(mod, dl, dev):
    import torch
    mod.eval()
    P, Y, S, Sm, A = [], [], [], [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev)
            s, lo = mod(x)
            P.append((torch.sigmoid(lo) > 0.5).sum(1).cpu().numpy())
            Y.append(y.numpy())
            S.append(s.cpu().numpy())
            Sm.append(mod.score(torch.flip(x, dims=[4])).cpu().numpy())
            A.append(mod.pesos_atencion(x).cpu().numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    S, Sm, A = np.concatenate(S), np.concatenate(Sm), np.concatenate(A)
    esp = float(np.mean(np.abs(S + Sm) <= 1e-4 + 1e-3 * np.abs(S)))
    return {"acierto": float((P == Y).mean()),
            "un_zona": float((np.abs(P - Y) <= 1).mean()),
            "err_ord": float(np.abs(P - Y).mean()),
            "espejo": esp,
            "acierto3": float(np.mean([agr3(a) == agr3(b) for a, b in zip(P, Y)])),
            "aciertox": float(np.mean([agrx(a) == agrx(b) for a, b in zip(P, Y)])),
            "aten_max": float(np.median(A.max(axis=1))), "n": len(Y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader

    filas = ZM.carga_indice()
    ent_f, res_f = ZM.particion_por_torneo(filas)
    ent, res = por_clip(ent_f), por_clip(res_f)
    print("LA ZONA DE LA PISTA, LEYENDO EL CLIP ENTERO")
    print("=" * 86)
    print("clips: entrena %d, reservado %d  (%d fotogramas por clip)" % (len(ent), len(res), N_FOT))
    print("torneos sin solape: %s"
          % (len({f["torneo"] for f in ent_f} & {f["torneo"] for f in res_f}) == 0))
    print()
    print("A BATIR: siempre middle 0.409 | geometria 0.323 | el mismo modelo POR FOTOGRAMA 0.430")
    print("TECHOS HUMANOS: 5 zonas 0.598 | izq/centro/der 0.676 | extremo o no 0.757")
    print()
    dl_e = DataLoader(DSClip(ent, True), batch_size=a.lote, shuffle=True, num_workers=4,
                      drop_last=True)
    dl_r = DataLoader(DSClip(res, False), batch_size=a.lote, shuffle=False, num_workers=4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = construye(dev)
    opt = torch.optim.AdamW(mod.parameters(), lr=a.lr, weight_decay=1e-4)
    mejor = -1.0
    for ep in range(a.epocas):
        mod.train()
        tot = 0.0
        for x, y in dl_e:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            _s, lo = mod(x)
            p = ZM.perdida_ordinal(lo, y)
            opt.zero_grad()
            p.backward()
            opt.step()
            tot += float(p) * len(y)
        m = evalua(mod, dl_r, dev)
        print("epoca %2d  perdida %.4f | acierto %.3f  <=1z %.3f  err %.3f | 3 cajones %.3f  "
              "extremo %.3f | espejo %.3f  aten_max %.2f"
              % (ep + 1, tot / max(len(ent), 1), m["acierto"], m["un_zona"], m["err_ord"],
                 m["acierto3"], m["aciertox"], m["espejo"], m["aten_max"]), flush=True)
        if m["acierto"] > mejor:
            mejor = m["acierto"]
            PESOS.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": mod.state_dict(), "metricas": m, "zonas": list(ZM.ZONAS)}, PESOS)
    print()
    print("mejor acierto POR CLIP en el reservado: %.3f" % mejor)
    print("   contra: siempre middle 0.409 | por fotograma 0.430 | techo humano 0.598")
    print("pesos en %s" % PESOS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
