from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel


class ScenarioParams(BaseModel):
    power_blocks: Dict[str, float]   # block_name → W/cm²
    htc: float = 5000.0              # W/m²·K
    t_ambient: float = 45.0          # °C
    pattern: str = 'uniform'


class JobRequest(BaseModel):
    geometry: str
    scenario_name: str
    scenario_params: ScenarioParams


class JobResponse(BaseModel):
    job_id: str
    status: Literal['pending', 'running', 'done', 'failed']


class GeometryInfo(BaseModel):
    name: str
    stack_type: str
    layers: int
    mesh_points: int
    die_length_um: float
    die_width_um: float
    power_blocks: List[str]
    custom: bool
