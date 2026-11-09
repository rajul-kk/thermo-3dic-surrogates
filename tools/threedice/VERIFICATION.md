# Verification record: threedice 4.0.0.post1 (Linux x86_64)

Date 2026-10-12. Nothing here was run for aarch64 or macOS.

## Environment

| item | value |
|---|---|
| host | Windows 11 Home 10.0.26200, Docker 29.5.2, WSL2 (kernel 6.6.114.1, Ubuntu, glibc 2.41, gcc 15.2.0) |
| build container | `quay.io/pypa/manylinux_2_28_x86_64` (digest `sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae`), gcc 14.2.1, bison 3.0.4, flex 2.6.1, OpenBLAS 0.3.15 (AlmaLinux 8 `openblas-devel`), setuptools >= 77, auditwheel (pip latest of the day) |
| fresh install env | `python:3.11-slim` container (Python 3.11.16), `python -m venv`, `pip install <wheel>` |
| pinned reference build | WSL `~/3d-ice-4.0` @ 4953952, gcc 15.2.0, system OpenBLAS 0.3.32 (pthread), SuperLU_MT e60ed2c |

## Build

```
docker run -d -v <repo>\tools\threedice:/src:ro -v <out>:/out quay.io/pypa/manylinux_2_28_x86_64 bash /src/build_manylinux.sh
```

The script builds the sdist (upstream fetched at the pinned shas, sha verified), builds the wheel *from the sdist* (offline),
runs `auditwheel repair --plat manylinux_2_28_x86_64`. The first attempts failed and were fixed: `cblas.h` is in
`/usr/include/openblas` on AlmaLinux (`CPATH`), SuperLU_MT's makefile races its `ar` rules under `-j` (built serially), and the
wheel had to be marked platlib for auditwheel.

| artefact | size | sha256 |
|---|---|---|
| `threedice-4.0.0.post1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | **12,284,802 bytes (11.7 MiB)** | `65d8961b2fa0c5e78be3b0dc98901c1e691b36eaff30685479ff30ecd26fbddb` |
| `threedice-4.0.0.post1.tar.gz` (sdist, with both upstream source tarballs) | 12,323,243 bytes | `5c6e231001cb13ea89e100d3e952378047db4cda75176683afd314b1f274e7bf` |

The tag is the legitimate one auditwheel found (symbols need glibc >= 2.27), listed as both manylinux_2_27 and manylinux_2_28.
Most of the size is the bundled OpenBLAS (37 MB uncompressed); the binary is 364 kB stripped. Bundled by auditwheel:
libopenblas, libgomp, libgfortran, libquadmath (`threedice.libs/`). SuperLU_MT is linked statically.

## Checks

1. **Fresh install, import** (python:3.11-slim venv):
   `python -c "import threedice; print(threedice.executable(), threedice.ice_commit(), threedice.__version__)"` printed
   `/venv/lib/python3.11/site-packages/threedice/bin/3D-ICE-Emulator 4953952a1ef6d38807ff307212a6f15e5b2ef935 4.0.0.post1`.
   `ldd` of the binary resolves libopenblas/libgomp/libgfortran/libquadmath inside `threedice.libs/` through `$ORIGIN/../../threedice.libs`.
2. **Console script**: `/venv/bin/3d-ice-emulator example_steady.stk` (upstream `bin/example_steady.stk`) exited 0
   ("Emulation took 2.739 sec", wrote output1..5.txt, xaxis/yaxis).
3. **Same-input comparison with the pinned build**: the same `example_steady.stk` run with the wheel binary (extracted in WSL) and with
   `~/3d-ice-4.0/bin/3D-ICE-Emulator`; `cmp` of all 7 text outputs (output1..5, xaxis, yaxis): byte-identical.
4. **Re-solve of stored data** (`scripts/resolve_from_metadata.py`, wheel binary through
   `THERMO_ICE_EXECUTABLE="wsl /home/rajul/tdverify/sp/threedice/bin/3D-ICE-Emulator"`, the pin check accepted it through
   `_build_info.json`; then the same files through the default pinned build; one CPU job at a time). Nine files:
   layout g1 (test_041, train_040), g4, g6, g7 (train_040), interface pilot (geometry5 tim_top_80), interface pilot median (geometry4 tim_sink_8),
   geometry7 pilot test_016, moving-geometry1 test_041. Result file: `results/3dice_version_check/threedice_wheel_resolve.json`.

   | comparison | max abs dT over the 9 files (float32 temperature arrays as stored) |
   |---|---|
   | wheel vs stored | **0.0** |
   | wheel vs pinned WSL build | **0.0** (`np.array_equal` true for all 9) |
   | pinned WSL build vs stored | 0.0 |

   **Bit-identical on every file**, although the wheel uses gcc 14.2 + OpenBLAS 0.3.15 and the pinned build gcc 15.2 + OpenBLAS 0.3.32.
   Limits of this statement: 9 files, float32 as stored (a difference below float32 resolution would not show), 3D-ICE ran single-threaded
   ("Used threads (1)"), and all are steady solves. TSV-density files, leakage/throttle feedback and the microchannel pilot were not re-solved (resolve_from_metadata refuses them).
5. **Tests**
   * `python -m pytest tools/iceforge -q` (Windows, WSL backend): 140 passed, 7 skipped on the first full run, 1 failure
     (`test_doctor_runs` asserted the backend list without `pip`; test updated, 8/8 of test_cli pass); full re-run after the fix: **141 passed, 7 skipped, 0 failed** (the skipped ones are HotSpot / real-wheel tests absent on this host; the real-wheel test is run in the container, below).
   * `python -m pytest tests/test_ice_pin.py -q`: 12 passed (9 existing + 3 new wheel/sidecar tests).
   * iceforge inside the python:3.11-slim venv with the wheel installed (pip backend live): `tests/test_pip_backend.py` all 11 passed including the real-wheel
     solve of the `aligned` repro case (max rise 11.130 K), and `tests/test_run.py` 10 passed, 1 failed: `test_wsl_path_translation_with_spaces`, a Windows-path
     test that cannot pass on Linux (unrelated to this change). `iceforge doctor` there: `pip ... (threedice 4.0.0.post1); smoke test ok; HEAD 4953952 (pinned)`, auto backend = pip.
   * `tools/threedice/tests/test_threedice.py` in the venv: 3 passed.

## Not verified

* macOS and aarch64 (only the draft workflow exists, never run); Windows has no native build.
* Behaviour on a glibc older than 2.27 (outside manylinux_2_28 policy anyway), OpenMP thread scaling (3D-ICE used 1 thread).
* A byte-reproducible wheel: the build embeds the build date; the binary was not compared across two builds.
