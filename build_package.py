import shutil
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


def build_cython_extensions() -> None:
    EXT = ".pyx"
    MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    EXTRA_COMPILE_ARGS = ["-O3"]

    extensions = [
        Extension(
            "frechetlib.frechet_utils",
            ["src/frechetlib/frechet_utils" + EXT],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[np.get_include()],
            define_macros=MACROS,
        ),
        Extension(
            "frechetlib.geometry_utils",
            ["src/frechetlib/geometry_utils" + EXT],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[np.get_include()],
            define_macros=MACROS,
        ),
        Extension(
            "frechetlib.retractable_frechet",
            ["src/frechetlib/retractable_frechet" + EXT],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[np.get_include()],
            define_macros=MACROS,
        ),
        Extension(
            "frechetlib.continuous_frechet",
            ["src/frechetlib/continuous_frechet" + EXT],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            include_dirs=[np.get_include()],
            define_macros=MACROS,
        ),
    ]

    extensions = cythonize(
        extensions,
        annotate=True,
        compiler_directives={"language_level": "3str"},
    )

    dist = Distribution({"ext_modules": extensions})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = "src" / output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build_cython_extensions()
