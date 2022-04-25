import os
import shutil

top = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(top, "ranger", "cpp_version")
dst = os.path.join(top, "skranger", "ranger")


def copy_ranger_source():
    """Copy the ranger cpp source, following symlinks."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=False)


def patch_make_unique():
    """Patch ranger calls to ``make_unique``.

    This enables us to compile on windows by ensuring we call ``ranger::make_unique``
    rather than ``std::make_unique``
    """
    for root, dirs, files in os.walk(dst, topdown=False):
        for file in files:
            if file != "utility.h":
                with open(os.path.join(root, file), "r") as f:
                    contents = f.read()
                contents = contents.replace("make_unique", "ranger::make_unique")
                with open(os.path.join(root, file), "w") as f:
                    f.write(contents)


copy_ranger_source()
patch_make_unique()
