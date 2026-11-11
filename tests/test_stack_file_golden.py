"""Golden test for ICESimulator's stack.stk: the text for every geometry and the special paths, byte for byte.

No solver is needed: generate_config_files only writes files. Paths to the config and output directories are
replaced by placeholders, so the digests do not depend on where the test runs. A digest changes only when the
.stk text changes; if that is intended, regenerate with `python tests/test_stack_file_golden.py` and say why in
the commit.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.geometry_builders import get_geometry_by_name
from src.scenario.generator import ScenarioGenerator
from src.simulators.ice_simulator import ICESimulator

GEOMETRIES = ["geometry1", "geometry2a", "geometry3", "geometry4", "geometry5", "geometry6", "geometry7"]


def _first_train(geom):
    gen = ScenarioGenerator()
    sc = gen.generate_all_scenarios(geom)
    gen.attach_tsv_maps(sc, geom)
    return sc[0].to_dict()


def _cases():
    """{case name: (geometry, scenario dict, executable)}."""
    out = {}
    for n in GEOMETRIES:
        g = get_geometry_by_name(n)
        out[n] = (g, _first_train(g), "x")
    g6 = get_geometry_by_name("geometry6")
    gen = ScenarioGenerator()
    extra = gen.generate_extra_training_scenarios(g6, 1, pool_start_idx=0)   # carries a tim_top override
    gen.attach_tsv_maps(extra, g6, seed_base=900_000)
    out["geometry6_k_override"] = (g6, extra[0].to_dict(), "x")
    g5 = get_geometry_by_name("geometry5")
    sc = _first_train(g5)
    fp_layer = g5.die_footprints[0].die_layer_name
    sc["layer_k_overrides"] = {fp_layer: 120.0, "tim_top": 10.0}
    sc["num_cores"] = 2
    out["geometry5_footprint_override_cores"] = (g5, sc, "x")
    out["geometry4_wsl_paths"] = (get_geometry_by_name("geometry4"), _first_train(get_geometry_by_name("geometry4")),
                                  "wsl /opt/3D-ICE-Emulator")
    sys.path.insert(0, str(ROOT / "scripts"))
    from gen_microchannel_pilot import build_microchannel_geometry6, microchannel_cooling
    gm = build_microchannel_geometry6()
    sc = _first_train(gm)
    sc.update(microchannel_cooling(1))
    out["geometry6_microchannel"] = (gm, sc, "x")
    return out


def stack_text(tmp: Path, geometry, scenario, executable) -> str:
    cfg, outd = tmp / "cfg", tmp / "out"
    cfg.mkdir(parents=True, exist_ok=True)
    outd.mkdir(parents=True, exist_ok=True)
    sim = ICESimulator(config_dir=cfg, output_dir=outd, executable=executable)
    sim.generate_config_files(geometry, scenario)
    text = (cfg / "stack.stk").read_text()
    for d, tag in ((cfg, "<CFG>"), (outd, "<OUT>")):
        r = d.resolve()
        for form in (sim._to_wsl_path(r), str(r)):
            text = text.replace(form, tag)
    return text.replace("\\", "/")  # Windows path separators; the .stk grammar has no other backslashes


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Recorded 2026-11-11, before _generate_stack_file was split into sections.
GOLDEN = {
    "geometry1": "854f41bd721179ca9b082d093f6e4f1ebbdb6ff6055645941abbcb40229b2cc3",
    "geometry2a": "20ae2e456ef2c94f5c2a8a095b2fb8e2cbbbf5145d581141644688e4670a65e1",
    "geometry3": "413a7f7d458907a712a234ea8e7d6fae1f44249a08ae5f1c0879dc5c32925e59",
    "geometry4": "2308d53d042cea94c11ea01cddaa26c535c1000d1d832cde1fd3fcbd482add8f",
    "geometry4_wsl_paths": "2308d53d042cea94c11ea01cddaa26c535c1000d1d832cde1fd3fcbd482add8f",
    "geometry5": "b09f86b1d3104f90523a59a25f413d64555f1970b0e27966830b43ffb948c0d2",
    "geometry5_footprint_override_cores": "dabe800eb6df9c10bb123309a5d6ada4fcbac4f3e5a58297cfe92a1350566090",
    "geometry6": "9b25e766f6b4e310586dd46187409ecc9cda4c345de6087ce684079eb1a44418",
    "geometry6_k_override": "438783e09763d9c29d2b931793d19313f62375b5d304b8203b5d087a36ff56ca",
    "geometry6_microchannel": "eccf4c2bbd686955d231595c8414ec596d689275a0adcf448dd564835ec1a814",
    "geometry7": "2f85243b03b44b6b3805b8ed1f6fd9b0678bc677d4a3f42d8811586d2cbe8cfb",
}


@pytest.fixture(scope="module")
def cases():
    return _cases()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_stack_file_unchanged(name, cases, tmp_path, monkeypatch):
    monkeypatch.setenv("ICEFORGE_GUARD", "0")
    g, sc, exe = cases[name]
    assert digest(stack_text(tmp_path, g, sc, exe)) == GOLDEN[name], (
        f"{name}: stack.stk text changed; if intended, regenerate GOLDEN and say why in the commit")


def test_every_case_is_pinned(cases):
    assert sorted(GOLDEN) == sorted(cases)


if __name__ == "__main__":
    import os
    import tempfile
    os.environ["ICEFORGE_GUARD"] = "0"
    for name, (g, sc, exe) in sorted(_cases().items()):
        with tempfile.TemporaryDirectory() as t:
            print(f'    "{name}": "{digest(stack_text(Path(t), g, sc, exe))}",')
