"""Run the whole project from scratch, in order (works on Windows, macOS, Linux):

    python run_all.py            # notebooks 01-05 (outputs kept in the .ipynb files) + unit tests
    python run_all.py --fresh    # also delete the cached split, results and models first
    python run_all.py --guide    # also rebuild reports/project_guide.pdf (needs Chrome or Edge)

Notebooks are executed with nbconvert when it is installed; otherwise with the small
fallback executor in tools/nbexec.py (same result, no Jupyter kernel needed).
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ["01_eda.ipynb", "02_baseline_propensity.ipynb", "03_uplift_models.ipynb",
             "04_evaluation_economics.ipynb", "05_robustness_crossfit.ipynb"]


def run_with_nbconvert(path: Path) -> None:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)


def main():
    if "--fresh" in sys.argv:
        for p in [ROOT / "data" / "processed", ROOT / "reports" / "results", ROOT / "models"]:
            for f in p.glob("*"):
                if f.name != ".gitkeep":
                    f.unlink() if f.is_file() else shutil.rmtree(f)
        print("Deleted cached split, models and results.")
    try:
        import nbconvert, nbformat  # noqa: F401
        runner = run_with_nbconvert
    except ImportError:
        sys.path.insert(0, str(ROOT / "tools"))
        from nbexec import run_notebook as runner
        print("nbconvert not installed: using tools/nbexec.py")
    for name in NOTEBOOKS:
        t0 = time.time()
        runner(ROOT / "notebooks" / name)
        print(f"OK  {name}  ({time.time() - t0:.0f}s)", flush=True)
    code = subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT)
    if "--guide" in sys.argv:
        subprocess.check_call([sys.executable, str(ROOT / "reports" / "build_guide.py")], cwd=ROOT)
    sys.exit(code)


if __name__ == "__main__":
    main()
