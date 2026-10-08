"""The Dockerfile builds the pinned 3D-ICE and the pin check can read its commit inside the image.
The image itself is not built here: these tests read the Dockerfile and exercise verify_ice_executable on a throwaway
install tree laid out like the container's (<root>/bin/3D-ICE-Emulator next to <root>/.git)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulators import ice_simulator as ice

DOCKERFILE = (Path(__file__).resolve().parent.parent / 'Dockerfile').read_text(encoding='utf-8')


def test_clone_is_pinned_to_the_commit_the_code_enforces():
    sha = re.search(r'^ARG ICE_COMMIT=([0-9a-f]{40})$', DOCKERFILE, re.M).group(1)
    assert sha.startswith(ice.PINNED_ICE_COMMIT)
    assert 'fetch --depth 1 origin ${ICE_COMMIT}' in DOCKERFILE
    assert 'rev-parse HEAD)" = "${ICE_COMMIT}"' in DOCKERFILE                  # the build fails on a mismatch
    assert 'git clone' not in DOCKERFILE                                        # no more unpinned clone of HEAD


def test_env_var_is_the_one_the_code_reads():
    exe = '/opt/3d-ice/bin/3D-ICE-Emulator'
    assert f'ENV THERMO_ICE_EXECUTABLE={exe}' in DOCKERFILE
    assert f'COPY --from=builder {exe} {exe}' in DOCKERFILE
    # the default the code falls back to is read from this very variable
    assert "os.environ.get(\n    'THERMO_ICE_EXECUTABLE'" in Path(ice.__file__).read_text().replace('\r\n', '\n')


def test_runtime_image_can_read_the_commit():
    runtime = DOCKERFILE[DOCKERFILE.rindex('FROM ubuntu:22.04'):]                # the last stage
    assert re.search(r'^\s+git \\$', runtime, re.M)                              # git installed in the runtime stage
    assert 'COPY --from=builder /opt/3d-ice/.git /opt/3d-ice/.git' in runtime    # next to bin/, where the lookup looks


@pytest.mark.skipif(shutil.which('git') is None, reason='git not available')
def test_commit_lookup_and_pin_check_work_on_a_container_style_tree(tmp_path, monkeypatch):
    root = tmp_path / 'opt' / '3d-ice'
    (root / 'bin').mkdir(parents=True)
    exe = root / 'bin' / '3D-ICE-Emulator'
    exe.write_text('#!/bin/sh\n')

    def git(*a):
        return subprocess.run(['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', *a],
                              check=True, capture_output=True, text=True).stdout.strip()

    git('init', '-q')
    git('add', 'bin')
    git('commit', '-q', '-m', 'x')
    head = git('rev-parse', 'HEAD')
    posix = exe.as_posix()
    monkeypatch.setattr(ice, '_COMMIT_CACHE', {})
    monkeypatch.delenv(ice.ALLOW_ANY_ENV, raising=False)
    assert ice.ice_commit(posix) == head
    with pytest.raises(RuntimeError, match='pinned'):                           # a different commit is refused ...
        ice.verify_ice_executable(posix)
    monkeypatch.setattr(ice, 'PINNED_ICE_COMMIT', head[:7])
    assert ice.verify_ice_executable(posix) == {'ice_commit': head, 'ice_executable': posix}   # ... the pinned one passes

    shutil.rmtree(root / '.git', onerror=lambda fn, path, _: (Path(path).chmod(0o700), fn(path)))   # no .git: 'unknown'
    monkeypatch.setattr(ice, '_COMMIT_CACHE', {})
    assert ice.ice_commit(posix) == 'unknown'
