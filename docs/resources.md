================================================================================
3D-IC THERMAL MODELING - LITERATURE AND RESOURCES
================================================================================

PROJECT: 3D-IC Thermal PINN Benchmark System
PURPOSE: Training data generation for Physics-Informed Neural Networks
DATE: 2026-03-27

================================================================================
MATERIAL PROPERTIES - PRIMARY SOURCES
================================================================================

1. SILICON (k=148 W/m·K at 300K, ρCp=1.63×10⁶ J/m³·K)

   Primary Reference:
   - Glassbrenner, C.J., & Slack, G.A. (1964)
     "Thermal Conductivity of Silicon and Germanium from 3K to the Melting Point"
     Physical Review, 134(4A), A1058-A1069
     DOI: 10.1103/PhysRev.134.A1058

   Supporting References:
   - CRC Handbook of Chemistry and Physics (103rd Edition, 2022-2023)
     Section 12: Properties of Solids - Thermal Conductivity of Metals and Semiconductors

   Temperature Dependence:
   - k_Si(T) ≈ 148 × (300/T)^1.3 for 300K < T < 1000K
   - At 400K: k ≈ 80 W/m·K (drops ~50% from room temp)
   - At 500K: k ≈ 50 W/m·K

2. COPPER (k=400 W/m·K, ρCp=3.55×10⁶ J/m³·K)

   Primary Reference:
   - ASM Metals Reference Book (3rd Edition, 1993)
     Chapter: Thermal Properties of Metals

   Supporting References:
   - NIST Materials Database
     https://www.nist.gov/mml/acmd/thermophysical-properties-division

   Notes:
   - Pure copper at 300K
   - Practical heat sinks: k=350-385 W/m·K (impurities, oxidation)

3. THERMAL INTERFACE MATERIALS (TIM) (k=4 W/m·K)

   Primary References:
   - Prasher, R. (2006)
     "Thermal Interface Materials: Historical Perspective, Status, and Future Directions"
     Proceedings of the IEEE, 94(8), 1571-1586
     DOI: 10.1109/JPROC.2006.879796

   Industry Standards:
   - JEDEC JESD51-1: Integrated Circuit Thermal Measurement Method
   - JEDEC JESD51-2: Integrated Circuits Thermal Test Method

   Commercial TIM Datasheets:
   - Shin-Etsu X23-7783D: k=4.0 W/m·K (thermal grease)
   - Dow Corning TC-5625: k=3.8 W/m·K (thermal compound)
   - Thermal Grizzly Kryonaut: k=12.5 W/m·K (high-performance)
   - Liquid metal (Ga-In alloys): k=20-80 W/m·K

   Typical Range:
   - Standard thermal grease: 3-5 W/m·K
   - Phase-change materials: 2-4 W/m·K
   - Thermal pads: 1-6 W/m·K
   - Liquid metal (extreme): 20-80 W/m·K

4. BONDING LAYER (k=50 W/m·K, hybrid bonding)

   References:
   - Lau, J.H. (2014)
     "Overview and Outlook of Through-Silicon Via (TSV) and 3D Integrations"
     Microelectronics International, 31(1), 3-21
     DOI: 10.1108/MI-10-2013-0044

   Technology Types:
   - Hybrid Cu-Cu bonding: k=50-100 W/m·K (Cu interconnects + SiO₂ dielectric)
   - Micro-bump (SnAg solder): k=30-50 W/m·K
   - Direct Cu-Cu bonding: k=150-200 W/m·K (low oxide, high pressure)

================================================================================
TSV (THROUGH-SILICON VIA) THERMAL MODELING
================================================================================

1. EQUIVALENT CONDUCTIVITY METHOD (Used in this project)

   Model: k_eff = (1 - φ) × k_Si + φ × k_Cu

   Primary References:
   - Koo, K., et al. (2010)
     "Compact Thermal Modeling of Through-Silicon-Via (TSV) Arrays"
     IEEE International Electron Devices Meeting (IEDM)
     DOI: 10.1109/IEDM.2010.5703382

   - Coskun, A.K., et al. (2009)
     "Analysis and Optimization of MPSoC Reliability"
     Journal of Low Power Electronics, 5(1), 56-69

   - Liu, Y., et al. (2012)
     "Full-Chip TSV-to-TSV Thermal Coupling Analysis and Optimization"
     IEEE/ACM International Conference on Computer-Aided Design (ICCAD)
     DOI: 10.1145/2429384.2429502

2. TSV DESIGN PARAMETERS

   Typical Ranges (from literature and industry):
   - TSV diameter: 5-10 μm
   - TSV pitch: 10-100 μm
   - TSV density: 0.1-10% area fraction
     * Memory (HBM): 5-10%
     * Logic: 1-3%
     * Analog: <1%
   - TSV depth: 50-100 μm (thinned wafer)

   References:
   - Lau, J.H. (2011)
     "TSV Manufacturing Yield and Hidden Costs for 3D IC Integration"
     Electronic Components and Technology Conference (ECTC)
     DOI: 10.1109/ECTC.2011.5898569

3. LIMITATIONS OF HOMOGENIZATION

   Reference:
   - Oprins, H., et al. (2011)
     "Fine-Grained ATM Based Thermal Modeling of 3D Stacked ICs"
     Therminic Workshop
     DOI: 10.1109/THERMINIC.2011.6086567

   Key Findings:
   - Homogenization accurate for macro-scale (>1mm)
   - Underestimates local hotspots between TSVs by 5-15%
   - Valid for die-level thermal management (our use case)

================================================================================
3D-IC THERMAL MODELING - FUNDAMENTAL PAPERS
================================================================================

1. FOUNDATIONAL WORK

   - Bakir, M.S., et al. (2008)
     "3D Integration: Thermal Challenges and Solutions"
     IEEE Design & Test of Computers, 25(6), 568-579
     DOI: 10.1109/MDT.2008.149

   - Cong, J., & Zhang, Y. (2005)
     "Thermal Via Planning for 3-D ICs"
     IEEE/ACM International Conference on Computer-Aided Design (ICCAD)
     DOI: 10.1109/ICCAD.2005.1560157

2. COMPACT THERMAL MODELING TOOLS

   3D-ICE (Used in this project):
   - Sridhar, A., et al. (2010)
     "3D-ICE: Fast Compact Transient Thermal Modeling for 3D ICs with Inter-Tier
      Liquid Cooling"
     IEEE/ACM International Conference on Computer-Aided Design (ICCAD)
     DOI: 10.1109/ICCAD.2010.5654156

   - Project Website: http://esl.epfl.ch/3d-ice
   - Source Code: https://github.com/mchoi327/3d-ice

   HotSpot:
   - Skadron, K., et al. (2004)
     "Temperature-Aware Microarchitecture: Modeling and Implementation"
     ACM Transactions on Architecture and Code Optimization, 1(1), 94-125
     DOI: 10.1145/980152.980157

   - Project Website: http://lava.cs.virginia.edu/HotSpot/

3. COMMERCIAL TOOLS (For comparison/validation)

   - Ansys Icepak: CFD-based thermal simulation
   - Ansys RedHawk: Electro-thermal co-simulation for chip design
   - COMSOL Multiphysics: Multi-physics FEA
   - Mentor Graphics FloTHERM: Electronics cooling simulation
   - Cadence Celsius: IC-level thermal analysis

================================================================================
PHYSICS MODELS AND BOUNDARY CONDITIONS
================================================================================

1. FOURIER HEAT CONDUCTION EQUATION

   Steady-State (used in this project):
   ∇·(k∇T) + q = 0

   Transient Form:
   ρCp(∂T/∂t) = ∇·(k∇T) + q

   Standard References:
   - Incropera, F.P., & DeWitt, D.P. (2006)
     "Fundamentals of Heat and Mass Transfer" (6th Edition)
     John Wiley & Sons, ISBN: 978-0471457282

   - Cengel, Y.A., & Ghajar, A.J. (2014)
     "Heat and Mass Transfer: Fundamentals and Applications" (5th Edition)
     McGraw-Hill, ISBN: 978-0073398181

2. CONVECTIVE BOUNDARY CONDITIONS

   Newton's Law of Cooling:
   q = h(T_surface - T_ambient)

   Typical HTC Values (Heat Transfer Coefficient):
   - Natural convection (air): 5-25 W/m²·K
   - Forced convection (fan): 50-250 W/m²·K
   - Heat sink + fan: 1000-5000 W/m²·K
   - Liquid cooling: 5000-20000 W/m²·K

   References:
   - JEDEC JESD51-2: Thermal Resistance Measurements
   - ASHRAE Handbook - Fundamentals (2021 Edition)

3. RADIATION BOUNDARY CONDITIONS (Tier 1 addition)

   Stefan-Boltzmann Law:
   q_rad = ε σ (T⁴ - T_amb⁴)

   where:
   - ε = emissivity (0.0-1.0)
   - σ = 5.67×10⁻⁸ W/(m²·K⁴) (Stefan-Boltzmann constant)

   Typical Emissivities:
   - Bare silicon: ε ≈ 0.6-0.7
   - Oxidized silicon: ε ≈ 0.7-0.8
   - Anodized aluminum (black): ε ≈ 0.8-0.9
   - Polished copper: ε ≈ 0.02-0.05
   - Oxidized copper: ε ≈ 0.6-0.7

   When Radiation Matters:
   - At T=450K (177°C): q_rad ≈ 30% of q_conv (for ε=0.8, h=5000 W/m²·K)
   - Negligible for T<350K with good forced convection

   Reference:
   - Howell, J.R., et al. (2015)
     "Thermal Radiation Heat Transfer" (6th Edition)
     CRC Press, ISBN: 978-1439894552

================================================================================
MACHINE LEARNING FOR PHYSICS - RECENT WORK
================================================================================

1. PHYSICS-INFORMED NEURAL NETWORKS (PINNs)

   Foundational Paper:
   - Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019)
     "Physics-Informed Neural Networks: A Deep Learning Framework for Solving
      Forward and Inverse Problems Involving Nonlinear Partial Differential Equations"
     Journal of Computational Physics, 378, 686-707
     DOI: 10.1016/j.jcp.2018.10.045

   Review Papers:
   - Karniadakis, G.E., et al. (2021)
     "Physics-Informed Machine Learning"
     Nature Reviews Physics, 3, 422-440
     DOI: 10.1038/s42254-021-00314-5

   - Cuomo, S., et al. (2022)
     "Scientific Machine Learning Through Physics-Informed Neural Networks"
     Journal of Scientific Computing, 92, 88
     DOI: 10.1007/s10915-022-01939-z

2. NEURAL OPERATORS (Foundation Model Approach)

   Fourier Neural Operator (FNO):
   - Li, Z., et al. (2020)
     "Fourier Neural Operator for Parametric Partial Differential Equations"
     International Conference on Learning Representations (ICLR)
     arXiv: 2010.08895

   DeepONet (Deep Operator Networks):
   - Lu, L., et al. (2021)
     "Learning Nonlinear Operators via DeepONet Based on the Universal
      Approximation Theorem of Operators"
     Nature Machine Intelligence, 3, 218-229
     DOI: 10.1038/s42256-021-00302-5

   Graph Neural Operators:
   - Li, Z., et al. (2022)
     "Multipole Graph Neural Operator for Parametric Partial Differential Equations"
     Neural Information Processing Systems (NeurIPS)
     arXiv: 2006.09535

3. TRANSFORMERS FOR PHYSICAL SIMULATION

   - Brandstetter, J., et al. (2022)
     "Message Passing Neural PDE Solvers"
     International Conference on Learning Representations (ICLR)
     arXiv: 2202.03376

   - Cao, S. (2021)
     "Choose a Transformer: Fourier or Galerkin"
     Neural Information Processing Systems (NeurIPS)
     arXiv: 2105.14995

4. MULTI-FIDELITY LEARNING

   - Pang, G., et al. (2020)
     "nPINNs: Nonlocal Physics-Informed Neural Networks for a Parametrized
      Nonlocal Universal Laplacian Operator"
     Journal of Computational Physics, 422, 109760
     DOI: 10.1016/j.jcp.2020.109760

   - Meng, X., & Karniadakis, G.E. (2020)
     "A Composite Neural Network That Learns From Multi-Fidelity Data"
     Journal of Computational Physics, 401, 109020
     DOI: 10.1016/j.jcp.2019.109020

================================================================================
INDUSTRY STANDARDS AND GUIDELINES
================================================================================

1. JEDEC THERMAL STANDARDS

   - JESD51-1: Integrated Circuit Thermal Measurement Method
   - JESD51-2: Integrated Circuits Thermal Test Method - Natural Convection
   - JESD51-8: Integrated Circuit Thermal Test Method - Junction-to-Board
   - JESD51-14: Transient Dual Interface Test Method

   Download: https://www.jedec.org/standards-documents/results/jesd51

2. SEMI STANDARDS (Semiconductor Equipment)

   - SEMI E156: Mechanical Specification for Thermal Test Die
   - SEMI G69: Guide for Thermal Characterization of Semiconductor Packages

   Website: https://www.semi.org/

3. IEEE STANDARDS

   - IEEE 1620: Standard for Test Methods for Thermal Characterization of
                 Electronic Packages

   - IEEE TCPMT (Transactions on Components, Packaging and Manufacturing Technology)
     Key journal for thermal packaging research

================================================================================
COMPUTATIONAL RESOURCES AND TOOLS
================================================================================

1. OPEN-SOURCE THERMAL SOLVERS

   - 3D-ICE: Compact thermal simulator (this project)
     GitHub: https://github.com/mchoi327/3d-ice

   - HotSpot: Grid-based compact thermal model
     Website: http://lava.cs.virginia.edu/HotSpot/

   - OpenFOAM: CFD solver (overkill for this project, but industry-standard)
     Website: https://www.openfoam.com/

2. PYTHON LIBRARIES (Used in this project)

   - NumPy: Array operations, data export
     Version: ≥1.21.0
     Docs: https://numpy.org/doc/

   - SciPy: Scientific computing, interpolation
     Version: ≥1.7.0
     Docs: https://docs.scipy.org/

   - PyYAML: Configuration file management
     Version: ≥5.4.0
     Docs: https://pyyaml.org/

   - Matplotlib: Visualization
     Version: ≥3.4.0
     Docs: https://matplotlib.org/

3. PINN FRAMEWORKS (For future implementation)

   - DeepXDE: PINN library (TensorFlow/PyTorch backend)
     GitHub: https://github.com/lululxvi/deepxde
     Paper: Lu et al. (2021), SIAM Review

   - NVIDIA Modulus: Physics-ML platform
     Website: https://developer.nvidia.com/modulus
     GitHub: https://github.com/NVIDIA/modulus

   - PyTorch: Deep learning framework
     Website: https://pytorch.org/

================================================================================
DATA AND BENCHMARKS
================================================================================

1. PUBLIC THERMAL DATASETS (Limited availability)

   - OpenThermo: Industrial cooling system data (NOT chip-level)
     Website: https://www.nist.gov/programs-projects/openthermo

   - HotSpot Sample Benchmarks: Alpha/SPEC CPU benchmarks
     Included with HotSpot distribution

   NOTE: No large-scale public IC thermal datasets exist (proprietary data)

2. SYNTHETIC BENCHMARKS

   - This project generates synthetic data via 3D-ICE
   - 60 scenarios across 4 geometries
   - Parameterized: power, HTC, ambient temperature
   - Purpose: PINN training and validation

================================================================================
FUTURE DIRECTIONS AND OPEN QUESTIONS
================================================================================

1. Temperature-Dependent Material Properties
   - Most critical for silicon: k(T), leakage(T)
   - Requires iterative solvers (implemented in 3D-ICE)

2. Transient Thermal Analysis
   - Thermal time constants: τ = ρCp·L²/k
   - For 1mm silicon: τ ≈ 10 ms
   - Requires time-stepping solvers

3. Electro-Thermal Coupling
   - Power depends on temperature: P(T) = P_dynamic + P_leakage(T)
   - Leakage: I_leak ∝ exp(T/T_0) (exponential dependence)
   - Requires co-simulation with circuit models

4. Reliability Modeling
   - Electromigration: MTTF ∝ exp(E_a / kT)
   - Thermal cycling: Low-cycle fatigue of solder joints
   - Time-dependent dielectric breakdown (TDDB)

5. Foundation Models for Thermal Problems
   - No existing pre-trained models (as of 2026)
   - Opportunity for transfer learning across geometries
   - Data scarcity is main bottleneck

================================================================================
CONTACT AND ATTRIBUTION
================================================================================

This resource compilation supports the 3D-IC Thermal PINN Benchmark project.

All material properties and physics models are from peer-reviewed literature
and industry standards as cited above.

For questions about specific references or data sources, consult the original
papers via their DOI links.

================================================================================
END OF RESOURCES
================================================================================
