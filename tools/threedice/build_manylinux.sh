#!/bin/bash
# Build the sdist and the manylinux_2_28 x86_64 wheel inside quay.io/pypa/manylinux_2_28_x86_64.
# Usage (PowerShell on Windows, or any Docker host):
#   docker run --rm -v <path to tools/threedice>:/src:ro -v <output dir>:/out quay.io/pypa/manylinux_2_28_x86_64 bash /src/build_manylinux.sh
set -euxo pipefail
dnf install -y epel-release || true
dnf install -y bison flex patch git openblas-devel gcc-gfortran
PY=/opt/python/cp311-cp311/bin
export CPATH=/usr/include/openblas      # cblas.h lives here on AlmaLinux
mkdir -p /work && cd /work
cp -r /src/. /work/proj && cd proj && rm -rf build upstream dist
$PY/pip install build auditwheel 2>&1 | tail -1
$PY/python -m build --sdist --outdir /work/dist .          # fetches upstream at the pinned shas, bundles them
mkdir -p /work/sdist_x && tar -xzf /work/dist/threedice-*.tar.gz -C /work/sdist_x
cd /work/sdist_x/threedice-*/                                # the wheel is built from the sdist, offline
$PY/pip install "setuptools>=77" wheel 2>&1 | tail -1
$PY/pip wheel --no-deps --no-build-isolation -w /work/raw .
$PY/auditwheel show /work/raw/*.whl
$PY/auditwheel repair --plat manylinux_2_28_x86_64 -w /out /work/raw/*.whl
cp /work/dist/threedice-*.tar.gz /out/
ls -l /out
