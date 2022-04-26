import os
import shutil

top = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(top, "ranger", "cpp_version")
dst = os.path.join(top, "skranger", "ranger")


def copy_ranger_source():
    """Copy the ranger cpp source, following symlinks."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=False)


def disambiguate_ranger_make_unique():
    """Rewrite ranger calls to ``make_unique``.

    This enables us to compile on Windows by ensuring we call ``ranger::make_unique``
    explicitly via the namespace. This removes ambiguity since windows compiles with
    C++14 which is when ``make_unique`` was added to ``std``.
    """
    for root, dirs, files in os.walk(dst, topdown=False):
        for file in files:
            # don't rewrite the definition
            if file != "utility.h":
                with open(os.path.join(root, file), "r") as f:
                    contents = f.read()
                contents = contents.replace("make_unique", "ranger::make_unique")
                with open(os.path.join(root, file), "w") as f:
                    f.write(contents)


copy_ranger_source()
disambiguate_ranger_make_unique()
