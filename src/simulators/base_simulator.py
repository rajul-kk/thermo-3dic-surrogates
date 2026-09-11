"""Abstract base class for thermal simulator wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import numpy as np
from ..core.geometry import Geometry


class ThermalSimulator(ABC):
    """Abstract base class for thermal simulation wrappers."""

    def __init__(self, config_dir: Path, output_dir: Path, executable: str = ""):
        """Initialize simulator wrapper."""
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.executable = executable

        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_config_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """Generate simulator-specific configuration files."""
        pass

    @abstractmethod
    def run_simulation(self, scenario_name: str) -> Path:
        """Run thermal simulation and return output file path."""
        pass

    @abstractmethod
    def parse_results(self, result_file: Path) -> Dict[str, np.ndarray]:
        """Parse simulation results into standard format."""
        pass

    def simulate(self,
                 geometry: Geometry,
                 scenario: Dict[str, Any],
                 scenario_name: str) -> Dict[str, np.ndarray]:
        """Complete simulation workflow: config → run → parse."""
        # Generate configuration files
        self.generate_config_files(geometry, scenario)

        # Run simulation
        result_file = self.run_simulation(scenario_name)

        # Parse results
        results = self.parse_results(result_file)

        return results

    def cleanup_temp_files(self, scenario_name: str) -> None:
        """Clean up temporary configuration and output files."""
        # Default implementation: subclasses can override
        pass

    def check_executable(self) -> bool:
        """Check if simulator executable exists and is accessible."""
        import shutil
        return shutil.which(self.executable) is not None

    def get_version(self) -> str:
        """Get simulator version information."""
        import subprocess

        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return ""
