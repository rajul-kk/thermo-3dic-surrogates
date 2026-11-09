# threedice

The [3D-ICE](https://github.com/esl-epfl/3d-ice) 4.0 thermal simulator (`3D-ICE-Emulator`) as a pip-installable
package. It ships one compiled binary, built from upstream commit
`4953952a1ef6d38807ff307212a6f15e5b2ef935` (the build the Thermo benchmark is pinned to), so a pinned,
reproducible 3D-ICE needs no compiler, no SuperLU and no `make`.

Licence: **GPL-3.0-or-later** (3D-ICE's own licence). Please cite the four 3D-ICE papers listed in `NOTICE`.
This is a repackaging; it is not affiliated with EPFL or the 3D-ICE authors.

```
pip install threedice            # Linux x86_64 (manylinux_2_28) wheel
python -c "import threedice; print(threedice.executable(), threedice.ice_commit())"
3d-ice-emulator model.stk        # same arguments as 3D-ICE-Emulator; run it in the stk's directory
```

API: `threedice.executable()` (path of the bundled binary), `threedice.ice_commit()` (the full upstream sha),
`threedice.build_info()` (contents of `threedice/_build_info.json`: upstream and SuperLU_MT commits, build date,
toolchain, applied patches with sha256), `3d-ice-emulator` (console script, execs the binary).

Not included: the SystemC wrapper, `3D-ICE-Server`/`3D-ICE-Client`, the Python heat-sink plugin.

## What is built, and from what

| piece | source | note |
|---|---|---|
| 3D-ICE | `esl-epfl/3d-ice` @ `4953952a1ef6d38807ff307212a6f15e5b2ef935` | patches in `patches/` |
| SuperLU_MT (OpenMP), static | `xiaoyeli/superlu_mt` @ `e60ed2c1dd491e1832d783f519c1851288f9d422` | patch in `patches/superlu_mt/` |
| OpenBLAS, libgomp, libgfortran | from the manylinux_2_28 container (OpenBLAS from AlmaLinux 8 `openblas-devel`) | bundled by `auditwheel repair` |

Note on SuperLU_MT: 3D-ICE's own tree ships `superlu_mt-4.0.0.zip`, but the pinned WSL build (`~/3d-ice-4.0`)
actually linked a git checkout of SuperLU_MT master (`e60ed2c`, 2026-03-29, which fixes the 64-bit-int prototype
mismatches of the zip). The wheel reproduces that tree, not the zip.

Patches (all build fixes, none changes the numerics of the algorithm):

1. `0001-drop-werror.patch`: removes `-Werror` (a newer compiler turns upstream warnings into errors). Same as the local patch in `docs/installation.md`, plus `bin/Makefile`.
2. `0002-luindex-int.patch`: `LUIndex_t` from `long` to `int`, to match SuperLU_MT's `int_t` (the second patch in `docs/installation.md`).
3. `superlu_mt/0001-openmp-openblas-make-inc.patch`: SuperLU_MT `make.inc` for OpenMP + `-lopenblas` (what `install-superlumt.sh` does by hand).

## Source and the GPL

The upstream source is **not** vendored in this git repository. `upstream.json` pins both repositories to full
shas. `build_support.py` fetches each at exactly that sha (`git fetch <sha>`), fails if `HEAD` differs, and makes a
`git archive` tarball. **The sdist contains both tarballs** under `upstream/` (complete corresponding source) plus
`patches/`, and building from the sdist needs no network. Whoever receives a wheel can get the matching source
from the sdist of the same version (or from the two upstream commits plus `patches/`).

## Building

Linux x86_64 wheel (the official manylinux_2_28 container; Docker on Windows: use PowerShell, Git Bash mangles paths):

```
docker run --rm -v <repo>/tools/threedice:/src:ro -v <outdir>:/out quay.io/pypa/manylinux_2_28_x86_64 bash /src/build_manylinux.sh
```

This builds the sdist, then builds the wheel *from the sdist*, then runs `auditwheel repair`. The wheel is tagged
`py3-none-manylinux_2_28_x86_64`. Build dependencies: gcc, make, bison, flex, patch, git, OpenBLAS (with `cblas.h`).
From a source checkout on any Linux with those: `pip install ./tools/threedice`.

Verification record (commands, numbers, environment): `VERIFICATION.md`.

## Platforms

* **Linux x86_64**: built and verified (see `VERIFICATION.md`).
* **Linux aarch64** and **macOS** (libomp, Homebrew gcc, delocate): only described in
  `.github-workflows-draft/build.yml` (cibuildwheel). **That workflow has never been run and macOS is untested.**
  It is deliberately outside `.github/` so nothing triggers it.
* **Windows**: no native build. Install it inside WSL and let iceforge's WSL backend call it:

  ```
  wsl -e python3 -m venv ~/ice-venv && wsl -e ~/ice-venv/bin/pip install threedice
  # then point iceforge at it (path inside WSL):
  set ICE_EXECUTABLE=wsl /home/<you>/ice-venv/lib/python3.X/site-packages/threedice/bin/3D-ICE-Emulator
  ```

  (`wsl -e ~/ice-venv/bin/python -c "import threedice; print(threedice.executable())"` prints the exact path.) The pin
  check reads the commit from the package's `_build_info.json` for such paths.

## iceforge

`iceforge` has a `pip` backend: if `import threedice` works in the Python that runs iceforge, `--backend pip` (and
`auto`) use `threedice.executable()`, and the pin check takes the commit from `threedice.ice_commit()`.
Order of `auto`: `$ICE_EXECUTABLE`, **pip**, native `3D-ICE-Emulator` on PATH, WSL `~/3d-ice-4.0`, docker.

Licence note: iceforge's own licence field is unchanged. The wheel being GPL-3.0-or-later does not affect
iceforge, which only runs the binary as a separate process; whether to state iceforge's licence is the maintainer's call.
