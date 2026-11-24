"""Build the optional Cython search kernel as engine._xiangqi."""

from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
EXT_DIR = ROOT / "ext"

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cython is required to build the native search kernel.\n"
        "  pip install 'Cython>=3.0'\n"
        "  python setup.py build_ext --inplace"
    ) from exc

_WIN = __import__("sys").platform == "win32"
extensions = [
    Extension(
        "engine._xiangqi",
        sources=[str(EXT_DIR / "_xiangqi.pyx")],
        include_dirs=[str(EXT_DIR)],
        extra_compile_args=["/O2", "/openmp"] if _WIN else ["-O3", "-fopenmp"],
        extra_link_args=[] if _WIN else ["-fopenmp"],
    )
]

setup(
    name="chees",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        annotate=False,
    ),
    packages=["engine"],
    zip_safe=False,
)
