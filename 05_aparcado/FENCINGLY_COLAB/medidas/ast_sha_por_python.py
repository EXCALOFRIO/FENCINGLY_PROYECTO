"""Es `source_ast_sha256` estable entre versiones de Python?

`measurement_artifacts._producer()` mete en la huella `source_ast_sha256`, un digest del AST de los
20 ficheros de MEASUREMENT_SOURCES, y en `runtime_versions` SOLO declara numpy, pandas y scipy. Si el
digest del AST cambia con la version de Python, la huella cambia sin que quede constancia de por que.

Este script reimplementa `_ast_sha` y `_sha` sin importar el paquete (asi corre en cualquier
interprete, aunque no tenga numpy), y se ejecuta con dos Pythons distintos para comparar.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

MEASUREMENT_SOURCES = ("coach_panel.py", "clip_stats.py", "footwork.py", "armwork.py", "distance_control.py",
                       "attack_continuity.py", "calibration.py", "calibration_reference.py", "stature.py",
                       "rules_fie.py", "dataset.py", "features.py", "sample_time.py",
                       "context.py", "augment.py", "bout_formats.py", "psychology.py", "fencer_stats.py",
                       "analysis_payloads.py", "measurement_artifacts.py")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


class _NoDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def _ast_sha(text):
    tree = _NoDocstrings().visit(ast.parse(text))

    def stable(node):
        if isinstance(node, ast.AST):
            return [type(node).__name__, [[key, stable(value)] for key, value in ast.iter_fields(node)
                                          if not (key == "type_params" and not value)]]
        if isinstance(node, list):
            return [stable(value) for value in node]
        if isinstance(node, bytes):
            return {"bytes": node.hex()}
        if isinstance(node, complex):
            return {"complex": [node.real, node.imag]}
        if node is Ellipsis:
            return {"ellipsis": True}
        return node
    return _sha(stable(tree))


root = Path(sys.argv[1])
out = {name: _ast_sha((root / name).read_text(encoding="utf-8-sig")) for name in MEASUREMENT_SOURCES}
print(json.dumps({"python": sys.version.split()[0], "agregado": _sha(out), "por_fichero": out}, indent=1))
