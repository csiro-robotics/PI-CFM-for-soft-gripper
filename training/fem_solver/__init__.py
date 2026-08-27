"""
FEM Solver Package for 2D Linear Elasticity

Contains:
    - TorchFEMSolver: PyTorch-differentiable FEM solver for neural network training
"""

from .torch_fem_solver_fast import TorchFEMSolverFast as TorchFEMSolver

__all__ = ['TorchFEMSolver']