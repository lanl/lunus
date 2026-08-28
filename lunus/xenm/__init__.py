"""
Elastic Network Model (ENM) refinement against diffuse scattering data.

This module implements Anisotropic Network Model (ANM) calculations
according to Riccardi et al., Biophys J 2010, Vol. 99, pp. 2616-2625.

The xenm package provides tools for:
- Calculating elastic network models from protein structures
- Predicting diffuse X-ray scattering from ENM dynamics
- Refining ENM parameters against experimental diffuse scattering data
- Computing covariance matrices and normal modes

Main programs:
  xenm.py - Calculate ENM Hessian and predict diffuse scattering
  refine_enm.py - Optimize ENM parameters against experimental data
  calc_pinv.py - Compute pseudoinverse of Hessian matrices
  expand_V_to_all_atoms.py - Expand covariance from CA atoms to all atoms

Usage:
  See README.md for detailed usage instructions and examples.
"""

__version__ = "0.1.0"
__author__ = "Michael Wall"
__all__ = []
