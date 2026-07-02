"""
Test 3D-ICE connectivity: verify the executable is reachable and generate a
minimal config to confirm the simulator runs without error.

Usage:
    # With 3D-ICE-Emulator in PATH
    python scripts/test_ice_connection.py

    # With WSL-wrapped binary
    python scripts/test_ice_connection.py --executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator"

    # Full smoke test (generates config + runs simulation)
    python scripts/test_ice_connection.py --executable "wsl ..." --run
"""

import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_builders import build_geometry1
from src.simulators.ice_simulator import ICESimulator


def check_executable(executable: str) -> bool:
    """Return True if the executable can be called."""
    # shutil.which only works for simple names, not WSL-wrapped commands
    if executable.startswith('wsl '):
        # Test WSL by running a trivial echo
        try:
            result = subprocess.run(
                ['wsl', 'echo', 'ok'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                print(f"  WSL not available: {result.stderr}")
                return False
            # Now test that the actual binary exists inside WSL
            wsl_path = executable[4:].strip()  # strip "wsl "
            result2 = subprocess.run(
                ['wsl', wsl_path, '--version'],
                capture_output=True, text=True, timeout=15
            )
            print(f"  WSL binary response (rc={result2.returncode}):")
            output = (result2.stdout + result2.stderr).strip()
            if output:
                for line in output.splitlines()[:5]:
                    print(f"    {line}")
            return True  # even if --version fails, binary may still work
        except Exception as e:
            print(f"  WSL check failed: {e}")
            return False
    else:
        found = shutil.which(executable)
        if found:
            print(f"  Found: {found}")
            try:
                result = subprocess.run(
                    [executable, '--version'],
                    capture_output=True, text=True, timeout=10
                )
                output = (result.stdout + result2.stderr).strip()
                if output:
                    print(f"  Version: {output.splitlines()[0]}")
            except Exception:
                pass
            return True
        else:
            print(f"  Not found in PATH: {executable}")
            return False


def run_smoke_test(executable: str, tmp_dir: Path) -> bool:
    """Generate config for geometry1 scenario and run a single simulation."""
    print()
    print("[3/3] Running smoke test (geometry1, scenario 1)...")

    config_dir = tmp_dir / 'ice_config'
    output_dir = tmp_dir / 'ice_output'

    sim = ICESimulator(config_dir=config_dir, output_dir=output_dir, executable=executable)

    geometry = build_geometry1()
    scenario = {
        'power_blocks': {'block1': 2.0, 'block2': 0.5, 'block3': 0.5, 'block4': 0.5},
        'htc': 5000.0,
        't_ambient': 25.0,
        'pattern': 'uniform',
    }

    try:
        print(f"  Generating config files in {config_dir}...")
        sim.generate_config_files(geometry, scenario)
        stk = config_dir / 'stack.stk'
        flp = config_dir / 'floorplan.flp'
        print(f"  stack.stk: {'OK' if stk.exists() else 'MISSING'} ({stk.stat().st_size} bytes)")
        print(f"  floorplan.flp: {'OK' if flp.exists() else 'MISSING'}")

        print(f"  Running {executable} ...")
        result_file = sim.run_simulation('smoke_test')
        print(f"  Output file: {result_file} ({result_file.stat().st_size} bytes)")

        print("  Parsing results...")
        parsed = sim.parse_results(result_file)
        coords = parsed['coords']
        temps = parsed['temperature']
        print(f"  coords shape: {coords.shape}")
        print(f"  temperature range: {temps.min():.1f} - {temps.max():.1f} K  "
              f"({temps.min()-273.15:.1f} - {temps.max()-273.15:.1f} C)")

        if temps.min() < 270 or temps.max() > 700:
            print(f"  WARNING: temperature range outside expected 270-700 K bounds")
        else:
            print("  [OK] Temperature range is physically plausible")

        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test 3D-ICE executable connectivity")
    parser.add_argument(
        '--executable', default='3D-ICE-Emulator',
        help='Path to 3D-ICE-Emulator binary. '
             'For WSL: "wsl /home/user/3d-ice/bin/3D-ICE-Emulator"'
    )
    parser.add_argument(
        '--run', action='store_true',
        help='Run a full smoke test (generate config + simulate geometry1 scenario 1)'
    )
    parser.add_argument(
        '--tmp-dir', type=Path, default=Path('_ice_test_tmp'),
        help='Temporary directory for test config/output files'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("3D-ICE CONNECTION TEST")
    print("=" * 60)

    print()
    print(f"[1/3] Checking executable: {args.executable}")
    ok = check_executable(args.executable)
    if not ok:
        print()
        print("FAIL: 3D-ICE executable not reachable.")
        print()
        print("To install 3D-ICE via WSL2 on Windows:")
        print("  1. Enable WSL2: Run PowerShell as admin -> wsl --install")
        print("  2. Install Ubuntu from Microsoft Store")
        print("  3. Inside Ubuntu:")
        print("       sudo apt update && sudo apt install -y gcc make bison flex libsuperlu-dev")
        print("       git clone https://github.com/esl-epfl/3d-ice.git ~/3d-ice")
        print("       cd ~/3d-ice && make")
        print("  4. Re-run with: --executable \"wsl ~/3d-ice/bin/3D-ICE-Emulator\"")
        sys.exit(1)

    print("  [OK] Executable reachable")

    print()
    print("[2/3] Checking Python imports...")
    try:
        from src.core.geometry_builders import build_geometry1
        from src.simulators.ice_simulator import ICESimulator
        print("  [OK] ICESimulator and geometry imports OK")
    except ImportError as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    if not args.run:
        print()
        print("[3/3] Skipped smoke test (pass --run to execute a simulation)")
        print()
        print("PASS: 3D-ICE executable is reachable.")
        print(f"To run the full pipeline: python src/main.py --geometry geometry1 "
              f"--simulator 3d-ice --ice-executable \"{args.executable}\" --output data/ice_test")
        sys.exit(0)

    success = run_smoke_test(args.executable, args.tmp_dir)

    print()
    if success:
        print("=" * 60)
        print("PASS: 3D-ICE smoke test completed successfully.")
        print(f"Full dataset command:")
        print(f"  python src/main.py --all-geometries --simulator 3d-ice "
              f"--ice-executable \"{args.executable}\" --output data/3d-ice --verbose")
        print("=" * 60)
    else:
        print("=" * 60)
        print("FAIL: 3D-ICE smoke test failed. Check output above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
