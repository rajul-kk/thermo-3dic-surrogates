"""
Abstract base class for thermal simulator wrappers.

Defines the interface that all thermal simulator wrappers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import numpy as np
from ..core.geometry import Geometry


class ThermalSimulator(ABC):
    """
    Abstract base class for thermal simulation wrappers.

    Provides a unified interface for different thermal simulators (3D-ICE, HotSpot, etc.).
    Each concrete simulator implements config generation, execution, and result parsing.
    """

    def __init__(self, config_dir: Path, output_dir: Path, executable: str = ""):
        """
        Initialize simulator wrapper.

        Args:
            config_dir: Directory for configuration files
            output_dir: Directory for output files
            executable: Path to simulator executable
        """
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.executable = executable

        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate_config_files(self, geometry: Geometry, scenario: Dict[str, Any]) -> None:
        """
        Generate simulator-specific configuration files.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters dictionary containing:
                - power_blocks: {block_name: power_density_W/cm2}
                - htc: Heat transfer coefficient (W/m²·K)
                - t_ambient: Ambient temperature (°C)
                - pattern: Power distribution pattern name
        """
        pass

    @abstractmethod
    def run_simulation(self, scenario_name: str) -> Path:
        """
        Run thermal simulation and return output file path.

        Args:
            scenario_name: Unique identifier for this scenario

        Returns:
            Path to output file containing simulation results

        Raises:
            RuntimeError: If simulation fails
        """
        pass

    @abstractmethod
    def parse_results(self, result_file: Path) -> Dict[str, np.ndarray]:
        """
        Parse simulation results into standard format.

        Args:
            result_file: Path to simulator output file

        Returns:
            Dictionary containing:
                - 'coords': (N, 3) array of (x, y, z) coordinates in μm
                - 'temperature': (N,) array of temperatures in K

        Raises:
            ValueError: If result file is invalid or cannot be parsed
        """
        pass

    def simulate(self,
                 geometry: Geometry,
                 scenario: Dict[str, Any],
                 scenario_name: str) -> Dict[str, np.ndarray]:
        """
        Complete simulation workflow: config → run → parse.

        Args:
            geometry: Geometry object
            scenario: Scenario parameters
            scenario_name: Unique scenario identifier

        Returns:
            Dictionary with simulation results (coords, temperature)
        """
        # Generate configuration files
        self.generate_config_files(geometry, scenario)

        # Run simulation
        result_file = self.run_simulation(scenario_name)

        # Parse results
        results = self.parse_results(result_file)

        return results

    def cleanup_temp_files(self, scenario_name: str) -> None:
        """
        Clean up temporary configuration and output files.

        Args:
            scenario_name: Scenario identifier
        """
        # Default implementation: subclasses can override
        pass

    def check_executable(self) -> bool:
        """
        Check if simulator executable exists and is accessible.

        Returns:
            True if executable is found, False otherwise
        """
        import shutil
        return shutil.which(self.executable) is not None

    def get_version(self) -> str:
        """
        Get simulator version information.

        Returns:
            Version string or empty string if unavailable
        """
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
