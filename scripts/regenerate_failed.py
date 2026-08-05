"""
Regenerate NPZ files where 3D-ICE silently fell back to synthetic temperatures.

Detection: files whose point count != the 3D-ICE Tmap grid count
  (n_layers * nx * ny, i.e. 1 z-point per layer).

Each bad file's scenario parameters are preserved from its metadata so the
same scenario ID keeps the same power pattern, HTC, and ambient temperature.
The bad file is deleted only after the new 3D-ICE run succeeds.
"""
import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.geometry_builders import (
    build_geometry1,
    build_geometry2a,
    build_geometry3, build_geometry4, build_geometry5,
)
from src.simulators.ice_simulator import ICESimulator
from src.export.npz_exporter import NPZExporter
from src.core.mesh import generate_power_density_field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

ICE_EXECUTABLE = "wsl /home/rajul/3d-ice/bin/3D-ICE-Emulator"

# expected point count for each geometry (3D-ICE Tmap grid: 1 z per layer)
EXPECTED_PTS = {
    'geometry1':  60000,
    'geometry2a': 64000,
    'geometry3':  60000,
    'geometry4':  33600,
    'geometry5':  44800,
}

GEOMETRY_BUILDERS = {
    'geometry1':  build_geometry1,
    'geometry2a': build_geometry2a,
    'geometry3':  build_geometry3,
    'geometry4':  build_geometry4,
    'geometry5':  build_geometry5,
}


def find_bad_files(data_root: Path) -> list:
    """Return list of (npz_path, geometry_name, scenario_name, scenario_params)."""
    bad = []
    for gname, exp_pts in EXPECTED_PTS.items():
        gdir = data_root / gname
        if not gdir.exists():
            continue
        for f in sorted(gdir.glob('*.npz')):
            d = np.load(f, allow_pickle=True)
            npts = d['coords'].shape[0]
            if npts != exp_pts:
                meta = dict(d['metadata'][0])
                power_blocks = {
                    k[len('block_power_'):]: float(v)
                    for k, v in meta.items() if k.startswith('block_power_')
                }
                scenario_params = {
                    'power_blocks': power_blocks,
                    'htc':       float(meta['htc']),
                    't_ambient': float(meta['t_ambient_celsius']),
                    'pattern':   meta.get('pattern', 'unknown'),
                }
                bad.append((f, gname, meta['scenario_name'], scenario_params))
            d.close()
    return bad


def regenerate_one(npz_path: Path,
                   gname: str,
                   scenario_name: str,
                   scenario_params: dict,
                   geometry,
                   config_dir: Path,
                   output_dir: Path,
                   exporter: NPZExporter) -> bool:
    """Run 3D-ICE for this scenario and overwrite the bad NPZ. Returns True on success."""
    sim = ICESimulator(
        config_dir=config_dir,
        output_dir=output_dir,
        executable=ICE_EXECUTABLE,
    )
    try:
        parsed = sim.simulate(geometry, scenario_params, scenario_name)
    except Exception as e:
        log.error('  3D-ICE FAILED for %s: %s', scenario_name, e)
        return False
    finally:
        sim.cleanup_temp_files(scenario_name)

    coords = parsed['coords']
    temps  = parsed['temperature']

    # Verify point count matches expected
    exp = EXPECTED_PTS[gname]
    if len(coords) != exp:
        log.warning('  Unexpected pts %d (expected %d) for %s', len(coords), exp, scenario_name)

    # Re-derive power field and layer indices on 3D-ICE grid
    power = generate_power_density_field(coords, geometry, scenario_params['power_blocks'])
    layer_indices = np.array(
        [geometry.get_layer_index_at_z(z) for z in coords[:, 2]],
        dtype=np.int32,
    )

    # Delete bad file before saving new one (exporter writes to same path)
    if npz_path.exists():
        npz_path.unlink()

    exporter.export_scenario(
        scenario_name,
        geometry,
        scenario_params,
        temps,
        coords,
    )
    return True


def main():
    data_root = Path('data/3d-ice')
    config_dir = data_root / '_ice_configs'
    output_dir = data_root / '_ice_output'
    config_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    log.info('Scanning for synthetic/bad NPZ files...')
    bad_files = find_bad_files(data_root)
    log.info('Found %d bad files to regenerate', len(bad_files))

    # Pre-build all geometries (avoid rebuilding for every scenario)
    geometries = {name: builder() for name, builder in GEOMETRY_BUILDERS.items()}

    # One exporter per geometry directory
    exporters = {
        gname: NPZExporter(data_root / gname)
        for gname in GEOMETRY_BUILDERS
    }

    n_ok = n_fail = 0
    for i, (npz_path, gname, scenario_name, scenario_params) in enumerate(bad_files, 1):
        log.info('[%d/%d] %s  (htc=%.0f, pattern=%s, max_power=%.2f W/cm2)',
                 i, len(bad_files), scenario_name,
                 scenario_params['htc'],
                 scenario_params['pattern'],
                 max(scenario_params['power_blocks'].values()) if scenario_params['power_blocks'] else 0)

        ok = regenerate_one(
            npz_path, gname, scenario_name, scenario_params,
            geometries[gname], config_dir, output_dir, exporters[gname],
        )
        if ok:
            n_ok += 1
            log.info('  -> OK (%d pts)', EXPECTED_PTS[gname])
        else:
            n_fail += 1

    log.info('Done. Regenerated %d/%d. Failed: %d', n_ok, len(bad_files), n_fail)
    if n_fail:
        log.warning('Failed scenarios still have their original bad NPZ files.')


if __name__ == '__main__':
    main()
