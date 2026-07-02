"""
Physics-Informed DeepONet for multi-geometry 3D-IC thermal surrogate.

A single model trained across all 7 benchmark geometries.

Reference: Lu et al., "Learning Nonlinear Operators via DeepONet Based on
the Universal Approximation Theorem of Operators", Nature Machine Intelligence,
2021.  Physics-informed extension: Wang et al., "Improved Architectures and
Training Algorithms for Deep Operator Networks", 2022.

Modules:
    model       -- BranchNet, TrunkNet, PIDeepONet
    data_loader -- MultiGeomDataset (all geometries in one dataset)
    trainer     -- Training loop with optional PI loss
"""
