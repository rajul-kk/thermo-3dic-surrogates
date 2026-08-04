# Installation Guide - 3D-IC Thermal PINN Benchmark System

## Overview

This guide covers installing all dependencies for the 3D-IC Thermal PINN Benchmark System, including:
1. Python 3.8+ with required packages
2. 3D-ICE compact thermal simulator (primary)
3. HotSpot thermal simulator (optional cross-validation)
4. Configuration and verification

**Estimated installation time**: 30-60 minutes (depending on system and internet speed)

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: Intel i5 or equivalent (dual-core, 2 GHz+)
- RAM: 4 GB
- Disk: 2 GB free space
- OS: Windows, macOS, or Linux

**Recommended**:
- CPU: Intel i7 or better (quad-core, 3+ GHz)
- RAM: 8 GB+
- Disk: 5 GB free space
- OS: Ubuntu 20.04 LTS or later (best 3D-ICE support)

### Development Tools

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y gcc g++ make
sudo apt-get install -y bison flex
sudo apt-get install -y python3 python3-pip python3-venv
```

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install gcc make bison flex
brew install python3
```

#### Windows
Download and install:
1. **MinGW-w64** (GCC compiler for Windows)
   - Download from: https://www.mingw-w64.org/
   - Or install via Chocolatey: `choco install mingw`

2. **MSYS2** (for Unix-like environment on Windows)
   - Download from: https://www.msys2.org/
   - Install build tools: `pacman -S base-devel mingw-w64-x86_64-toolchain`

3. **Make**
   - Included with MSYS2, or download from: `choco install make`

4. **Python 3.8+**
   - Download from: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation

---

## Step 1: Python Environment Setup

### Create Virtual Environment

This isolates project dependencies from system Python.

#### Linux/macOS
```bash
cd /path/to/3D-ICE Thermal-modelling/Thermo

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### Windows (Command Prompt)
```bash
cd "path\to\3D-ICE Thermal-modelling\Thermo"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
cd "path\to\3D-ICE Thermal-modelling\Thermo"

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1
```

### Verify Python Installation

```bash
python --version          # Should be 3.8+
pip --version             # Should be 20.0+
```

### Install Python Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**requirements.txt contents** (if not present, create it):
```txt
numpy>=1.21.0
scipy>=1.7.0
pyyaml>=5.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
pytest>=6.2.0
```

**Verify installation:**
```bash
python -c "import numpy; print('NumPy', numpy.__version__)"
python -c "import yaml; print('PyYAML OK')"
python -c "import matplotlib; print('Matplotlib OK')"
```

---

## Step 2: 3D-ICE Installation (Priority)

3D-ICE is the primary compact thermal simulator used for ground-truth data generation.

> **Windows users**: 3D-ICE is a Linux-native tool. The recommended path on Windows 10/11
> is **WSL2 + Ubuntu** (Section 2a below). MSYS2 is possible but more fragile.

---

### Step 2a: WSL2 Setup (Windows 10/11 — Recommended)

WSL2 (Windows Subsystem for Linux 2) runs a full Linux kernel inside Windows. This is the
most reliable way to run 3D-ICE and HotSpot on Windows.

#### Enable WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

This installs WSL2 and Ubuntu 22.04 LTS. After installation, **restart Windows**.

If WSL is already installed but on version 1, upgrade it:

```powershell
wsl --set-default-version 2
wsl --list --verbose    # verify VERSION = 2
```

If Ubuntu wasn't installed automatically:

```powershell
wsl --install -d Ubuntu-22.04
```

Launch Ubuntu from the Start Menu and complete the initial username/password setup.

#### Install Build Dependencies Inside WSL

Open the Ubuntu terminal and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y gcc g++ make bison flex
sudo apt install -y libsuperlu-dev libopenblas-dev
sudo apt install -y git curl
```

Verify:

```bash
gcc --version    # should show GCC 11 or 12
make --version
bison --version
flex --version
```

#### Compile 3D-ICE Inside WSL

```bash
# Clone the repository
git clone https://github.com/esl-epfl/3d-ice.git ~/3d-ice
cd ~/3d-ice

# Build (takes 2-5 minutes)
make

# Verify the binary was created
ls -la bin/3D-ICE-Emulator
./bin/3D-ICE-Emulator   # should print usage/version info
```

If `make` fails with a SuperLU error, try the bundled SuperLU:

```bash
make SUPERLU=local
```

#### Test WSL2 + 3D-ICE from Windows PowerShell

Windows can call WSL binaries directly:

```powershell
wsl ~/3d-ice/bin/3D-ICE-Emulator
# Should print version/usage info
```

#### Configure the Pipeline to Use WSL

When running `src/main.py` from Windows PowerShell, pass the WSL path via `--ice-executable`:

```powershell
python src/main.py --geometry geometry1 --simulator 3d-ice `
    --ice-executable "wsl ~/3d-ice/bin/3D-ICE-Emulator" `
    --output data/ice_test --verbose
```

The `wsl` prefix tells Windows to route the command through WSL2. The `~` expands
to the WSL home directory (e.g., `/home/yourname`).

**Test connectivity only (no simulation):**

```powershell
python scripts/test_ice_connection.py --executable "wsl ~/3d-ice/bin/3D-ICE-Emulator"
```

**Full smoke test (generates config + runs geometry1 scenario 1):**

```powershell
python scripts/test_ice_connection.py --executable "wsl ~/3d-ice/bin/3D-ICE-Emulator" --run
```

#### WSL2 Troubleshooting

**Issue**: `wsl: command not found` in PowerShell
- WSL not installed. Run `wsl --install` from an elevated PowerShell prompt.

**Issue**: `make: error: SuperLU not found`
- Try: `sudo apt install -y libsuperlu-dev` then `make clean && make`
- Or: `make SUPERLU=local`

**Issue**: `./bin/3D-ICE-Emulator: Permission denied`
- Run: `chmod +x ~/3d-ice/bin/3D-ICE-Emulator`

**Issue**: Long WSL startup time delays each scenario (~2-3 s per call)
- This is normal for WSL cold start. Subsequent calls in the same session are faster.
- For the full 320-scenario dataset it adds ~13 min total — acceptable.

**Issue**: File path errors (`/mnt/c/...` vs `C:\...`)
- All paths passed to the simulator run inside WSL and must use Linux paths.
- The pipeline handles this automatically via `subprocess`.

---

### Get 3D-ICE Source

Two options:

**Option A: GitHub Mirror (Recommended)**
```bash
git clone https://github.com/mchoi327/3d-ice.git 3d-ice-source
cd 3d-ice-source
```

**Option B: EPFL Official Repository**
```bash
# Download from: http://esl.epfl.ch/3d-ice
# Extract to 3d-ice-source directory
```

### Compile 3D-ICE

#### Linux/macOS
```bash
cd 3d-ice-source

# Configure build
./configure --prefix=$HOME/local/3d-ice

# Compile (may take 5-10 minutes)
make -j4

# Install
make install

# Verify installation
$HOME/local/3d-ice/bin/3d-ice-emulator --help
```

#### Windows (MSYS2)
```bash
# Launch MSYS2 terminal
# Navigate to extracted 3d-ice-source
cd /c/path/to/3d-ice-source

./configure --prefix=$HOME/local/3d-ice
make -j4
make install

# Verify
$HOME/local/3d-ice/bin/3d-ice-emulator.exe --help
```

### Add 3D-ICE to PATH

#### Linux/macOS
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export PATH="$HOME/local/3d-ice/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/3d-ice/lib:$LD_LIBRARY_PATH"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Windows
1. Open Environment Variables:
   - Press `Win + X`, select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"

2. Add to system PATH:
   - New Variable: `3D_ICE_PATH` = `C:\path\to\3d-ice\bin`
   - Edit PATH: Add `;%3D_ICE_PATH%`

3. Restart terminal for changes to take effect

### Verify 3D-ICE Installation

```bash
which 3d-ice-emulator          # Should show path to executable
3d-ice-emulator --version      # Should display version
3d-ice-emulator --help         # Should show command options
```

### Common 3D-ICE Build Issues

**Issue**: "make: command not found"
- Solution (Linux): `sudo apt-get install make`
- Solution (macOS): `xcode-select --install`
- Solution (Windows): Install via MSYS2 or MinGW

**Issue**: "bison: command not found" or "flex: command not found"
- Solution (Linux): `sudo apt-get install bison flex`
- Solution (macOS): `brew install bison flex`
- Solution (Windows): Via MSYS2: `pacman -S bison flex`

**Issue**: "configure: command not found"
- Solution: Ensure you're in the 3d-ice-source directory
- Also ensure autotools installed: `sudo apt-get install autoconf automake libtool`

**Issue**: Permission denied when running installed binary
- Solution: `chmod +x $HOME/local/3d-ice/bin/3d-ice-emulator*`

---

## Step 3: HotSpot Installation (Optional)

HotSpot is provided as an optional cross-validation tool. It's simpler to build than 3D-ICE.

### Get HotSpot Source

```bash
# Download from official site
# url: http://lava.cs.virginia.edu/HotSpot/
# Or via GitHub mirror (if available)
git clone https://github.com/yshuixi/HotSpot.git hotspot-source
cd hotspot-source
```

### Compile HotSpot

#### Linux/macOS
```bash
make

# Verify
./hotspot -help
```

#### Windows (MSYS2)
```bash
make

# Verify (if built as .exe)
./hotspot.exe -help
```

### Add HotSpot to PATH

#### Linux/macOS
```bash
# Copy executable to accessible location
cp hotspot $HOME/local/bin/
export PATH="$HOME/local/bin:$PATH"
```

#### Windows
Copy `hotspot.exe` to a directory in your PATH (e.g., `C:\Windows\System32` or custom bin directory)

### Verify HotSpot Installation

```bash
which hotspot         # Should show path
hotspot -help         # Should display help
```

---

## Step 4: Configure Benchmark System

### Create Configuration Directories

```bash
mkdir -p configs/3d-ice/geometry{1,2a,2b,2c,3,4,5,6}
mkdir -p configs/scenarios
mkdir -p data/3d-ice
mkdir -p results
```

### Generate Initial Scenarios

```bash
# Activate Python environment (if not already active)
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows

# Generate scenarios without running simulations
python src/main.py --generator-only --output configs/scenarios
```

This will create YAML scenario definitions across all 8 geometries (see
[`docs/geometry_reference.md`](geometry_reference.md) for the current per-geometry
scenario counts — 320 total NPZ files across the full dataset, not a fixed 80).

---

## Step 5: Verification and Testing

### Quick Verification (No Simulators)

Test with synthetic data (no 3D-ICE or HotSpot needed):

```bash
# Activate Python environment
source venv/bin/activate

# Run the test suite
pytest tests/ -v

# Expected: app API and smoke tests pass
```

**Expected duration**: a few seconds

### 3D-ICE Verification (If Installed)

```bash
# Activate Python environment
source venv/bin/activate

# Run single geometry with 3D-ICE (generates 1 scenario)
python src/main.py --simulator 3d-ice --geometry geometry1 \
    --ice-executable "wsl /home/user/3d-ice/bin/3D-ICE-Emulator" --output data/3d-ice_test

# Expected: Creates data/3d-ice_test/geometry1/geometry1_train_001.npz
# Expected duration: 10-30 seconds per scenario
```

### Full Benchmark Generation (Synthetic Data)

Generate the full benchmark using synthetic thermal data (no simulators required) —
covers all 8 geometries with mock ground truth:

```bash
# Activate Python environment
source venv/bin/activate

# Generate benchmark
python src/main.py --all-geometries --simulator mock --output data/3d-ice-mock

# Expected: 320+ .npz files across data/3d-ice-mock/<geometry>/
# Expected duration: 5-15 minutes on modern system
```

For real 3D-ICE ground truth (the actual benchmark dataset), see the "Data Generation
Commands" section in [`docs/geometry_reference.md`](geometry_reference.md).

**Monitor progress:**
```bash
# Count generated files
find data -name "*.npz" | wc -l   # Should reach 80
du -sh data                        # Should reach 600-800 MB
```

### Validation Checks

Verify generated data:

```bash
python -c "
import numpy as np
from pathlib import Path

data_dir = Path('data')
npz_files = list(data_dir.glob('**/*.npz'))

print(f'Generated {len(npz_files)} .npz files')

# Check first file
if npz_files:
    data = np.load(npz_files[0])
    print(f'Keys: {list(data.keys())}')
    print(f'Coords shape: {data[\"coords\"].shape}')
    print(f'Temp range: {np.min(data[\"temp\"]):.1f}-{np.max(data[\"temp\"]):.1f} K')
    print(f'Power range: {np.min(data[\"power\"]):.1f}-{np.max(data[\"power\"]):.1e} W/m3')
"
```

---

## Step 6: Usage Examples

### Generate Benchmark Data

```bash
# Activate environment
source venv/bin/activate

# Option 1: Single geometry with synthetic data
python src/main.py --geometry geometry1 --simulator mock --output data

# Option 2: All geometries (full benchmark)
python src/main.py --all-geometries --simulator mock --output data

# Option 3: With 3D-ICE (if installed)
python src/main.py --all-geometries --simulator 3d-ice --output data

# Option 4: Verbose output for debugging
python src/main.py --all-geometries --simulator mock --output data --verbose
```

### Load and Inspect Data

```python
import numpy as np
from pathlib import Path

# Load single scenario
data = np.load('data/geometry1/train/geometry1_train_001.npz')

coords = data['coords']      # (N, 3) coordinates in micrometers
temps = data['temp']         # (N,) temperatures in Kelvin
power = data['power']        # (N,) volumetric power density
layers = data['layer']       # (N,) layer indices
metadata = data['metadata'].item()  # Dict with scenario parameters

print(f"Scenario: {metadata['scenario_name']}")
print(f"Points: {len(coords):,}")
print(f"Temperature: {np.min(temps):.1f}-{np.max(temps):.1f} K")
print(f"HTC: {metadata['htc']} W/m²·K")
print(f"Ambient: {metadata['t_ambient_celsius']}°C")
```

### Use in PINN Training

```python
import numpy as np
import torch
from pathlib import Path

# Load dataset
train_files = list(Path('data/geometry1/train').glob('*.npz'))
test_files = list(Path('data/geometry1/test').glob('*.npz'))

# Combine all training data
train_coords = []
train_temps = []
train_power = []

for npz_file in train_files:
    data = np.load(npz_file)
    train_coords.append(data['coords'])
    train_temps.append(data['temp'])
    train_power.append(data['power'])

# Convert to tensors
X_train = torch.tensor(np.vstack(train_coords), dtype=torch.float32)
y_train = torch.tensor(np.hstack(train_temps), dtype=torch.float32)
f_train = torch.tensor(np.hstack(train_power), dtype=torch.float32)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Now use X_train, y_train, f_train in PINN training loop
```

---

## Step 7: Troubleshooting

### Installation Issues

| Problem | Solution |
|---------|----------|
| `pip: command not found` | Ensure Python 3.8+ installed and in PATH |
| `No module named numpy` | Ensure virtual environment activated, then `pip install numpy` |
| `ImportError: cannot import name 'build_geometry1'` | Ensure you're in correct directory with `sys.path.insert(0, str(Path(__file__).parent))` |
| 3D-ICE compilation fails | Check prerequisites installed: `gcc --version`, `make --version`, `bison --version`, `flex --version` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| `3d-ice-emulator: command not found` | Add 3D-ICE bin to PATH (see Step 2) |
| `Out of memory` during large scenario | Reduce mesh resolution or process fewer scenarios at once |
| `.npz` file too large | Files are normal at 5-10 MB; use compression: `np.savez_compressed()` |
| Plots not generating | Install matplotlib: `pip install matplotlib seaborn` |

### Performance Issues

| Issue | Optimization |
|-------|--------------|
| Slow scenario generation | Use `--skip-train` or `--skip-test` to process only one type |
| Slow 3D-ICE | Reduce mesh resolution in geometry builders (temporary) |
| Memory usage | Process one geometry at a time: `--geometry geometry1` |

---

## System Verification Checklist

After completing installation, verify each component:

```bash
# Python environment
[ ] python --version                    # 3.8+
[ ] pip --version                       # 20.0+

# Python packages
[ ] python -c "import numpy"            # NumPy installed
[ ] python -c "import yaml"             # PyYAML installed
[ ] python -c "import matplotlib"       # Matplotlib installed

# 3D-ICE (if installed)
[ ] which 3d-ice-emulator               # In PATH
[ ] 3d-ice-emulator --version           # Executable works

# HotSpot (optional)
[ ] which hotspot                       # In PATH
[ ] hotspot -help                       # Executable works

# Project structure
[ ] ls src/main.py                      # Main script present
[ ] ls tests/                           # Tests present
[ ] ls configs/scenarios/               # Config directory exists
[ ] ls data/                            # Data directory exists

# Run tests
[ ] pytest tests/ -v                    # All tests pass
[ ] python src/main.py --generator-only # Generates scenarios
```

---

## Next Steps

1. **Generate Benchmark Data**
   ```bash
   python src/main.py --all-geometries --simulator mock --output data/3d-ice-mock
   ```

2. **Inspect Generated Data**
   ```bash
   find data/3d-ice-mock -name "*.npz" | head -5 | xargs ls -lh
   python -c "import numpy as np; d=np.load('data/3d-ice-mock/geometry1/geometry1_train_001.npz', allow_pickle=True); print(list(d.keys()))"
   ```

3. **Review Statistics**
   ```bash
   find results -name "*_stats.json" | head -5 | xargs head -20
   ```

4. **Train a model**
   - See the root [`README.md`](../README.md) Quick Start section for PINN/FNO/WHNO/
     DeepONet/ARO/Therm-FM training commands.

---

## Support and Resources

- **3D-ICE Documentation**: https://www.epfl.ch/labs/esl/research/open-source-tools-datasets/3d-ice/
- **HotSpot Documentation**: http://lava.cs.virginia.edu/HotSpot/
- Project documentation: see [`docs/`](.) and the root [`README.md`](../README.md)

---

## License and Attribution

This benchmark system builds on:
- **3D-ICE**: Sridhar et al. (2010), ICCAD
- **HotSpot**: Skadron et al. (2004), TACO
- **Material properties**: Glassbrenner & Slack (1964), Physical Review

---

## 3D-ICE 4.0 (added 2026-08-04)

Built at `/home/rajul/3d-ice-4.0`; the 3.0.0 install at `/home/rajul/3d-ice` is left
in place so existing results stay reproducible.

```bash
git clone --depth 1 https://github.com/esl-epfl/3d-ice.git 3d-ice-4.0
cd 3d-ice-4.0
cp -r ../3d-ice/superlu_mt-4.0.0 .        # reuse the already-built SuperLU MT
sed -i 's/-Wall -Wextra -Werror/-Wall -Wextra/' sources/Makefile
sed -i 's/-Werror//g' makefile.def
sed -i 's/typedef long LUIndex_t ;/typedef int LUIndex_t ;/' include/types.h
make -j$(nproc)
```

Two build fixes were needed and both are deliberate:

- **`-Werror` removed.** Upstream promotes warnings to errors, and `printf("%ld", ...)`
  against a 32-bit `int_t` fails on this toolchain. The warnings are in error-reporting
  paths only.
- **`LUIndex_t` changed from `long` to `int`.** 4.0 declares 64-bit LU indices while our
  SuperLU MT is built with 32-bit `int_t`; the two must agree. Our largest system is
  ~400k nodes, far below the 2^31 limit, so 32-bit indices are ample. The alternative is
  rebuilding SuperLU with `-D_LONGINT` and using `long long`.

**Validated against 3.0.0: bit-identical.** Three geometry1 scenarios gave
max |ΔT| = 0.0000 K, so the existing dataset remains valid and the upgrade alone does
not require regeneration.

### What 4.0 unlocks

Per-floorplan-element material, which 3.0.0 could not express (one material per layer):

```
Background1 :
  position       0,    0 ;
  dimension   5000, 5000 ;
  material    SILICON ;        # <-- per-element material
  discretization  10, 10 ;     # <-- sub-grid inside the element
  power values 12.5, 14.0 ;
```

plus anisotropic conductivity (`thermal conductivity kx, ky, kz ;`) and non-uniform
grids (`non-uniform true;` in `dimensions`).

This makes spatially varying lateral `k(x,y)` expressible for the first time, which
unblocks two things previously recorded as impossible:

- Spatially varying TSV density / conductivity maps — the canonical operator-learning
  benchmark structure (cf. Darcy flow with a GRF permeability field).
- The geometry4/5/6 underfill inconsistency in `assumptions.md` §6.1, where the PDE loss
  uses a heterogeneous `k` that the ground truth does not contain.
