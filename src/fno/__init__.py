"""
Fourier Neural Operator (FNO) for 3D-IC thermal surrogate modelling.

Maps power density field Q(x,y,z) → temperature field T(x,y,z) for all four
benchmark geometries in a single model. Unlike the geometry-specific PINNs,
the FNO learns the thermal operator directly and generalises across TSV densities.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations", ICLR 2021. https://arxiv.org/abs/2010.08895

Modules:
    model       -- SpectralConv3d, FNOBlock, FNO3d
    data_loader -- FNODataset (flat .npz → 3D grid tensors)
    trainer     -- Training loop
"""
