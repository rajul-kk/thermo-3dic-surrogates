"""Material property library for thermal simulations."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Material:
    """Thermal material properties."""
    name: str
    k_thermal: float  # W/m·K
    volumetric_heat_capacity: float  # J/m³·K
    density: Optional[float] = None  # kg/m³
    description: str = ""

    def __post_init__(self):
        """Validate material properties."""
        if self.k_thermal <= 0:
            raise ValueError(f"Thermal conductivity must be positive: {self.k_thermal}")
        if self.volumetric_heat_capacity <= 0:
            raise ValueError(f"Heat capacity must be positive: {self.volumetric_heat_capacity}")


class MaterialLibrary:
    """Library of standard thermal materials for 3D-IC packaging."""

    # Standard materials
    _materials: Dict[str, Material] = {
        'silicon': Material(
            name='silicon',
            k_thermal=148.0,
            volumetric_heat_capacity=1.63e6,
            density=2330.0,
            description='Silicon wafer substrate'
        ),

        'copper': Material(
            name='copper',
            k_thermal=400.0,
            volumetric_heat_capacity=3.55e6,
            density=8960.0,
            description='Copper heat spreader/sink'
        ),

        'aluminum': Material(
            name='aluminum',
            k_thermal=237.0,
            volumetric_heat_capacity=2.42e6,
            density=2700.0,
            description='Aluminum heat sink (alternative)'
        ),

        'tim': Material(
            name='tim',
            k_thermal=4.0,
            volumetric_heat_capacity=4.0e6,
            density=2500.0,
            description='Thermal grease/paste'
        ),

        'tim_high_k': Material(
            name='tim_high_k',
            k_thermal=8.0,
            volumetric_heat_capacity=4.0e6,
            density=2500.0,
            description='High-performance TIM (e.g., liquid metal)'
        ),

        'bonding': Material(
            name='bonding',
            k_thermal=50.0,
            volumetric_heat_capacity=2.5e6,
            density=3000.0,
            description='Die-to-die bonding layer (micro-bumps, hybrid bonding)'
        ),

        # Tier 0/1 upgrade materials for geometry5/6
        'silicon_low_k': Material(
            name='silicon_low_k',
            k_thermal=80.0,
            volumetric_heat_capacity=1.63e6,
            density=2330.0,
            # ENGINEERING ESTIMATE, not from a specific cited source: real N5/N3
            # BEOL low-k dielectric stacks reduce through-thickness effective k
            # to roughly 20-100 W/m·K depending on BEOL fraction of layer
            # thickness; 80 W/m·K is a plausible mid-range value but has not
            # been validated against a specific published measurement. If this
            # matters for a paper, replace with a cited number or an explicit
            # derivation (bulk Si k=148 weighted by BEOL volume fraction).
            description='Si active device layer with low-k dielectric — effective k at N5/N3 node (engineering estimate, not literature-cited)'
        ),

        'tim_indium': Material(
            name='tim_indium',
            k_thermal=80.0,
            volumetric_heat_capacity=1.70e6,
            density=7300.0,
            description='Indium solder TIM1 — die-to-spreader interface (k=80 W/m·K)'
        ),

        'c4_bump_array': Material(
            name='c4_bump_array',
            k_thermal=15.0,
            volumetric_heat_capacity=1.70e6,
            density=8000.0,
            # NUMERICAL INCONSISTENCY, flagged not fixed: the arithmetic
            # (parallel-conductor) mixing rule used elsewhere in this file
            # (see compute_tsv_effective_k) applied to "~10% Cu fraction" would
            # give k_eff = 0.10*400 + 0.90*k_matrix >= 40 W/m·K for ANY
            # non-negative matrix conductivity — i.e. 10%-Cu arithmetic mixing
            # can NEVER produce a value as low as 15 W/m·K regardless of what
            # the solder/underfill matrix k is. So the stated "10% Cu fraction"
            # and the assigned 15 W/m·K value are mutually inconsistent under
            # the mixing rule used elsewhere: either the true Cu fraction is
            # much lower than 10%, a different (non-arithmetic, e.g. harmonic
            # or effective-medium) mixing rule was used, or 15 W/m·K came from
            # an unstated separate source. Needs reconciliation before relying
            # on this number for anything precision-sensitive.
            description='C4 bump array effective medium — Cu pillars in solder matrix (~10% Cu fraction; k=15 W/m·K is an independent estimate, NOT derived via the arithmetic-mean formula used for TSVs — see code comment)'
        ),

        'hybrid_bonding': Material(
            name='hybrid_bonding',
            k_thermal=60.0,
            volumetric_heat_capacity=3.30e6,
            density=8500.0,
            description='Cu-Cu hybrid bonding interface — 9µm pitch pillar array, k_eff=60 W/m·K'
        ),

        # geometry7 (CoWoS-L-style package, reduced scale)
        'organic_substrate': Material(
            name='organic_substrate',
            k_thermal=0.5,
            volumetric_heat_capacity=1.9e6,
            density=1900.0,
            # ENGINEERING ESTIMATE: ABF (Ajinomoto Build-up Film) / BT-resin organic
            # substrate build-up layers are reported in the literature at roughly
            # 0.3-0.8 W/m·K, three orders of magnitude below bulk silicon's 148
            # W/m·K -- this is the whole point of the CoWoS-L vs CoWoS-S distinction:
            # a CoWoS-L "interposer" is mostly this organic film, with silicon LSI
            # (local silicon interconnect) bridge islands only under the specific
            # die-to-die regions that need bridge-grade routing density. Not
            # validated against a specific cited measurement; 0.5 W/m·K is a
            # plausible mid-range value for planning purposes only.
            description='CoWoS-L organic (ABF/BT) substrate build-up film -- the low-k '
                        'field the LSI bridge/via islands sit in'
        ),
        'lsi_bridge_via': Material(
            name='lsi_bridge_via',
            k_thermal=60.0,
            volumetric_heat_capacity=3.0e6,
            density=6000.0,
            # ENGINEERING ESTIMATE, deliberately NOT bulk silicon's 148 W/m·K.
            # First version of geometry7 (2026-08-17) used pure silicon (148) for
            # these islands but covered only narrow strips at die-to-die seams --
            # under a power-concentrating scenario, chipA's die had almost no
            # island coverage beneath it, forcing ~all its heat through the bare
            # organic field and producing a simulated 1125C peak (back-of-envelope
            # ΔT=q·t/k through 300µm of k=0.5 organic matches the simulated value
            # within 1.5%, confirming the mechanism, not a solver bug -- see
            # docs/compute.md 2026-08-18). Two ways to fix an underpowered-cooling-
            # path bug: extend the high-k coverage, or lower the assumed k so a
            # narrower coverage stops being catastrophic. Real CoWoS-L packages do
            # both -- LSI bridges are true silicon (148) but narrow, while compute
            # dies additionally get dense copper power/ground via + plane coverage
            # through the organic substrate itself (real organic substrates
            # laminate in several copper reference/power planes; effective
            # through-thickness k with via-dense PDN routing is well above the
            # bare resin's 0.3-0.8 W/m·K but well below bulk silicon). This
            # material represents that intermediate, PDN-realistic value and is
            # used uniformly for every bridge/via island in geometry7 (this
            # engine's `layer.material`/footprint mechanism does not currently
            # support two different footprint materials on the same layer --
            # seeded here as one physically-coherent composite instead of a
            # narrower-but-still-inaccurate patchwork). 60 W/m·K matches this
            # library's existing 'hybrid_bonding' precedent for a dense Cu-based
            # interconnect composite, not independently re-derived.
            description='CoWoS-L LSI bridge / power-delivery-via composite -- clearly '
                        'better than the bare organic field, clearly worse than a '
                        'monolithic silicon interposer'
        ),

    }

    @classmethod
    def get(cls, name: str) -> Material:
        """Retrieve material by name."""
        if name not in cls._materials:
            raise KeyError(
                f"Material '{name}' not found. Available materials: "
                f"{', '.join(cls._materials.keys())}"
            )
        return cls._materials[name]

    @classmethod
    def add(cls, material: Material) -> None:
        """Add or update a material in the library."""
        cls._materials[material.name] = material

    @classmethod
    def list_materials(cls) -> Dict[str, Material]:
        """Return dictionary of all available materials."""
        return cls._materials.copy()

    @classmethod
    def compute_tsv_effective_k(cls,
                               k_silicon: float,
                               k_copper: float,
                               tsv_fraction: float) -> float:
        """Calculate effective thermal conductivity for TSV region."""
        if not (0.0 <= tsv_fraction <= 1.0):
            raise ValueError(f"TSV fraction must be 0-1, got {tsv_fraction}")

        return (1.0 - tsv_fraction) * k_silicon + tsv_fraction * k_copper

    @classmethod
    def create_tsv_material(cls,
                           tsv_density: float,
                           name: Optional[str] = None) -> Material:
        """Create a custom TSV-enhanced silicon material."""
        k_si = cls.get('silicon').k_thermal
        k_cu = cls.get('copper').k_thermal
        rho_cp_si = cls.get('silicon').volumetric_heat_capacity
        rho_cp_cu = cls.get('copper').volumetric_heat_capacity

        k_eff = cls.compute_tsv_effective_k(k_si, k_cu, tsv_density)
        rho_cp_eff = (1.0 - tsv_density) * rho_cp_si + tsv_density * rho_cp_cu

        if name is None:
            name = f"silicon_tsv_{int(tsv_density*100)}pct"

        return Material(
            name=name,
            k_thermal=k_eff,
            volumetric_heat_capacity=rho_cp_eff,
            description=f"Silicon with {tsv_density*100:.1f}% TSV density"
        )

    @classmethod
    def k_silicon_temp_dependent(cls, T_kelvin: float) -> float:
        """Temperature-dependent silicon thermal conductivity."""
        k_300K = 148.0  # W/m·K at room temperature
        alpha = 1.3  # Power law exponent
        T_ref = 300.0  # Reference temperature (K)

        # Guard against unphysical temperatures
        if T_kelvin < 100:
            T_kelvin = 100.0  # Below this, phonon scattering model breaks down
        if T_kelvin > 1685:
            T_kelvin = 1685.0  # Silicon melting point

        return k_300K * (T_ref / T_kelvin) ** alpha

    @classmethod
    def summary(cls) -> str:
        """Return formatted summary of all materials."""
        lines = [
            "Material Library:",
            "=" * 80,
            f"{'Material':<25} {'k (W/m*K)':<15} {'rho*Cp (J/m^3*K)':<20} {'Description':<30}"
        ]
        lines.append("-" * 80)

        for mat in cls._materials.values():
            lines.append(
                f"{mat.name:<25} {mat.k_thermal:<15.2f} {mat.volumetric_heat_capacity:<15.2e} "
                f"{mat.description:<30}"
            )

        return "\n".join(lines)
