"""Fetch, verify, patch and build the pinned 3D-ICE source. Used by setup.py; no third-party imports.

Source handling:
  * an sdist carries upstream/<name>-<sha>.tar.gz (git archive of the pinned commit), so building it needs no network;
  * a source checkout clones the repositories listed in upstream.json at exactly that sha and fails if HEAD differs.
Every tarball's commit is verified against upstream.json (git writes it into the archive's pax header).
"""
import datetime
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "upstream.json").read_text())
PATCH_DIRS = {"3d-ice": HERE / "patches", "superlu_mt": HERE / "patches" / "superlu_mt"}


def _run(cmd, cwd=None, capture=False, **kw):
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None, **kw)


def tarball_name(name):
    return f"{name}-{PINS[name]['commit']}.tar.gz"


def tarball_commit(path):
    with tarfile.open(path) as tf:
        return tf.pax_headers.get("comment", "")


def fetch_tarball(name, dest_dir):
    """Return a verified git-archive tarball of PINS[name] in dest_dir (cloning if not already there)."""
    pin = PINS[name]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / tarball_name(name)
    if out.exists() and tarball_commit(out) == pin["commit"]:
        return out
    work = HERE / "build" / "_fetch" / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    _run(["git", "init", "-q"], cwd=work)
    _run(["git", "fetch", "-q", "--depth", "1", pin["url"], pin["commit"]], cwd=work)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=work)
    head = _run(["git", "rev-parse", "HEAD"], cwd=work, capture=True).stdout.strip()
    if head != pin["commit"]:
        raise RuntimeError(f"{name}: fetched {head}, but upstream.json pins {pin['commit']}")
    _run(["git", "archive", "--format=tar.gz", "-o", out, "HEAD"], cwd=work)
    if tarball_commit(out) != pin["commit"]:
        raise RuntimeError(f"{name}: archive does not record commit {pin['commit']}")
    shutil.rmtree(work, ignore_errors=True)
    return out


def source_tarball(name):
    """The sdist-bundled tarball when present (verified), else fetch it into build/."""
    bundled = HERE / "upstream" / tarball_name(name)
    if bundled.exists():
        if tarball_commit(bundled) != PINS[name]["commit"]:
            raise RuntimeError(f"{bundled} does not record the pinned commit {PINS[name]['commit']}")
        return bundled
    return fetch_tarball(name, HERE / "build" / "upstream")


def _extract(tarball, dest):
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(tarball) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:                           # Python < 3.12 without the filter argument
            tf.extractall(dest)


def _apply_patches(name, tree):
    applied = []
    for p in sorted(PATCH_DIRS[name].glob("*.patch")):
        _run(["patch", "-p1", "--forward", "--batch", "-i", p], cwd=tree)
        applied.append({"name": f"{name}/{p.name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return applied


def _first_line(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).splitlines()[0].strip()
    except Exception:
        return "unavailable"


def toolchain():
    t = {"gcc": _first_line(["gcc", "--version"]), "bison": _first_line(["bison", "--version"]),
         "flex": _first_line(["flex", "--version"]), "make": _first_line(["make", "--version"]),
         "platform": platform.platform(), "machine": platform.machine()}
    try:
        t["openblas"] = subprocess.run(["pkg-config", "--modversion", "openblas"], capture_output=True,
                                       text=True, timeout=30).stdout.strip() or "unavailable"
    except Exception:
        t["openblas"] = "unavailable"
    return t


def build_native(build_lib):
    """Compile 3D-ICE-Emulator and put it, with licences and _build_info.json, into <build_lib>/threedice/."""
    if os.name != "posix":
        raise RuntimeError("threedice has no native Windows build. Install it inside WSL (see README.md); "
                           "iceforge's WSL backend then calls it.")
    root = HERE / "build" / "native"
    ice, slu = root / "3d-ice", root / "superlu_mt"
    patches = []
    _extract(source_tarball("3d-ice"), ice)
    _extract(source_tarball("superlu_mt"), slu)
    patches += _apply_patches("3d-ice", ice)
    patches += _apply_patches("superlu_mt", slu)
    extra = shlex.split(os.environ.get("THREEDICE_MAKE_ARGS", ""))      # e.g. CC=gcc-14 on macOS
    jobs = [f"-j{os.cpu_count() or 2}"]
    (slu / "lib").mkdir()
    _run(["make", "-C", slu / "SRC", *extra])                    # libsuperlu_mt_OPENMP.a (static); serial: its ar rules race under -j
    lib_a = slu / "lib" / "libsuperlu_mt_OPENMP.a"
    if not lib_a.exists():
        raise RuntimeError(f"SuperLU_MT build did not produce {lib_a}")
    blas = os.environ.get("THREEDICE_BLAS_LIBS", "-lopenblas")
    mk = [f"SLU_MAIN={slu}", f"SLU_LIBS=-L{slu / 'lib'} -lsuperlu_mt_OPENMP {blas}", *extra]
    _run(["make", "-C", ice / "flex", *mk])
    _run(["make", "-C", ice / "bison", *mk])
    _run(["make", *jobs, "-C", ice / "sources", *mk])
    _run(["make", "-C", ice / "lib", *mk])
    _run(["make", "-C", ice / "bin", "3D-ICE-Emulator", *mk])
    exe = ice / "bin" / "3D-ICE-Emulator"
    if not exe.exists():
        raise RuntimeError("3D-ICE-Emulator was not built")
    pkg = Path(build_lib) / "threedice"
    (pkg / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, pkg / "bin" / "3D-ICE-Emulator")
    if shutil.which("strip"):
        subprocess.run(["strip", str(pkg / "bin" / "3D-ICE-Emulator")], check=False)
    (pkg / "licenses").mkdir(exist_ok=True)
    shutil.copy2(HERE / "COPYING", pkg / "licenses" / "COPYING")
    shutil.copy2(HERE / "NOTICE", pkg / "licenses" / "NOTICE")
    for cand in ("License.txt", "LICENSE", "License"):
        if (slu / cand).exists():
            shutil.copy2(slu / cand, pkg / "licenses" / "SuperLU_MT-License.txt")
            break
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    when = datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc) if epoch else \
        datetime.datetime.now(datetime.timezone.utc)
    info = {"package": "threedice",
            "ice_commit": PINS["3d-ice"]["commit"], "ice_repo": PINS["3d-ice"]["url"],
            "superlu_mt_commit": PINS["superlu_mt"]["commit"], "superlu_mt_repo": PINS["superlu_mt"]["url"],
            "build_date_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "toolchain": toolchain(), "patches": patches,
            "built_targets": ["3D-ICE-Emulator"],
            "skipped": ["SystemC wrapper", "3D-ICE-Server", "3D-ICE-Client", "heatsink plugin"],
            "make_args": extra, "blas_libs": blas}
    (pkg / "_build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("built", pkg / "bin" / "3D-ICE-Emulator", flush=True)
