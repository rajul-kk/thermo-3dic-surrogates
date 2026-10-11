import shutil
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_support as bs

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:                                 # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class build_py(_build_py):
    def run(self):
        super().run()
        bs.build_native(self.build_lib)


class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        up = Path(base_dir) / "upstream"             # full corresponding source, for the GPL
        up.mkdir(exist_ok=True)
        for name in bs.PINS:
            shutil.copy2(bs.source_tarball(name), up / bs.tarball_name(name))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):                       # platlib wheel: it carries a compiled binary
        return True


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"build_py": build_py, "sdist": sdist, "bdist_wheel": bdist_wheel})
