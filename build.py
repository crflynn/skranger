import os
from setuptools import Extension
from setuptools import setup

import numpy as np
from Cython.Build import cythonize

# skranger project directory
top = os.path.dirname(os.path.abspath(__file__))
# the cpp source code
ranger_src = os.path.join("ranger", "cpp_version", "src")

# include skranger, ranger, and numpy headers
include_dirs = [
    ".",
    os.path.join(top, ranger_src),
    os.path.join(top, ranger_src, "Forest"),
    os.path.join(top, ranger_src, "Tree"),
    os.path.join(top, ranger_src, "utility"),
    np.get_include(),
]


def find_pyx_files(directory, files=None):
    """Recursively find all Cython extension files.

    :param str directory: The directory in which to recursively crawl for .pyx files.
    :param list files: A list of files in which to append discovered .pyx files.
    """
    if files is None:
        files = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            find_pyx_files(path, files)
    return files


def create_extension(module_name):
    """Create a setuptools build extension for a Cython extension file.

    :param str module_name: The name of the module
    """
    path = module_name.replace(".", os.path.sep) + ".pyx"
    return Extension(
        module_name,
        sources=[path],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++11", "-Wall"],
        extra_link_args=["-std=c++11", "-g"],
    )


ext_modules = [create_extension(name) for name in find_pyx_files("skranger")]

setup(ext_modules=cythonize(ext_modules, gdb_debug=True))


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": ext_modules})
