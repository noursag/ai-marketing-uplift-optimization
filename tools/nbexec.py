"""Minimal notebook executor (fallback when nbconvert / a Jupyter kernel is not available).

    python tools_nbexec.py notebooks/01_eda.ipynb [more.ipynb ...]

Runs each code cell in order in one namespace (cwd = the notebook's folder, like Jupyter),
captures printed text, `display()` calls, the value of a cell's last expression and
matplotlib figures shown with plt.show(), and writes them back into the .ipynb as
standard nbformat-4 outputs. run_all.py uses nbconvert when it is installed and falls
back to this file otherwise.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _mime(obj) -> dict:
    """Best-effort rich representation, the way IPython would do it."""
    data = {}
    if hasattr(obj, "_repr_html_"):
        try:
            html = obj._repr_html_()
            if html:
                data["text/html"] = html
        except Exception:
            pass
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            data["text/plain"] = obj.to_string(max_rows=60, max_cols=30)
        elif hasattr(obj, "data") and hasattr(obj, "_repr_html_"):   # pandas Styler
            data["text/plain"] = obj.data.to_string(max_rows=60, max_cols=30)
    except Exception:
        pass
    data.setdefault("text/plain", repr(obj))
    return data


def _figure_outputs() -> list:
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
        outs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<Figure>"}})
    plt.close("all")
    return outs


def run_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    outputs_cell: list = []

    def display(*objs):
        for o in objs:
            outputs_cell.append({"output_type": "display_data", "metadata": {}, "data": _mime(o)})

    def show(*a, **k):
        outputs_cell.extend(_figure_outputs())

    ns["display"] = display
    plt.show = show
    old_cwd = os.getcwd()
    os.chdir(path.parent)
    count = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            count += 1
            cell["execution_count"] = count
            outputs_cell = []
            stdout = io.StringIO()
            try:
                tree = ast.parse(src)
                last_expr = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(tree.body[-1].value)
                    tree.body = tree.body[:-1]
                with contextlib.redirect_stdout(stdout):
                    exec(compile(tree, str(path), "exec"), ns)
                    result = eval(compile(last_expr, str(path), "eval"), ns) if last_expr is not None else None
                    outputs_cell.extend(_figure_outputs())   # figures not explicitly shown
            except Exception:
                tb = traceback.format_exc()
                text = stdout.getvalue()
                outs = ([{"output_type": "stream", "name": "stdout", "text": text}] if text else []) + outputs_cell
                outs.append({"output_type": "error", "ename": "Error", "evalue": tb.strip().splitlines()[-1], "traceback": tb.splitlines()})
                cell["outputs"] = outs
                path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
                raise
            outs = []
            text = stdout.getvalue()
            if text:
                outs.append({"output_type": "stream", "name": "stdout", "text": text})
            outs.extend(outputs_cell)
            if result is not None:
                outs.append({"output_type": "execute_result", "metadata": {}, "execution_count": count, "data": _mime(result)})
            cell["outputs"] = outs
    finally:
        os.chdir(old_cwd)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t0 = time.time()
        run_notebook(Path(p).resolve())
        print(f"OK  {Path(p).name}  ({time.time() - t0:.0f}s)", flush=True)
