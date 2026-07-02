"""
Physics-Informed Neural Network (PINN) for 3D-IC thermal surrogate modelling.

Predicts T(x,y,z) for the steady-state heat equation with temperature-dependent
silicon conductivity k(T) = 148*(300/T)^1.3, convective BC at the top surface,
and adiabatic walls on sides and bottom.

Modules:
    model       -- FourierPINN architecture (Fourier encoding + residual MLP)
    physics     -- k(T) evaluation and PDE/BC residual computation
    losses      -- Loss components and NTK-based adaptive weighting
    data_loader -- ThermalDataset, normalization, collocation sampling
    trainer     -- Curriculum training loop with Adam + cosine annealing
    evaluate    -- Metrics and comparison plots
"""
