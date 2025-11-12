# Third-party notices

This repository is released under the MIT License (see `LICENSE`). Parts of the code are
ported or adapted from other open-source projects, and the code depends on third-party
packages with their own licences. This file lists them. The repository itself does not
redistribute any third-party source tree; only the adapted fragments named below live here.

## Code adapted into this repository

### Shape As Points (Peng et al. 2021)

* Upstream: https://github.com/autonomousvision/shape_as_points
* Licence: MIT (Copyright (c) 2021 Songyou Peng et al.)
* Where: `mpu/dpsr.py` (the differentiable Poisson surface reconstruction: `fftfreqs`, `img`,
  `spec_gaussian_filter`, `point_rasterize`, `grid_interp`, the `DPSR` module) and `mpu/sap.py`
  (the marching-cubes surrogate gradient `PSR2Mesh`, dL/dchi = splat(-(dL/dV . n)), and the
  Adam update of oriented points).
* Existing attribution in the code: `mpu/dpsr.py` opens with "Differentiable Poisson Surface
  Reconstruction (Peng et al. 2021, Shape As Points), compact port." and `mpu/sap.py` with
  "Shape-As-Points style inverse rendering step (Peng et al. 2021), adapted to the MPU particle
  pipeline." The DPSR code was rewritten into a compact form; the spectral solve, the
  rasterisation and the sign convention follow the upstream implementation.

### Neural Implicit Evolution (Mehta et al. 2022)

* Upstream: https://github.com/ishit/nie
* Licence: the upstream repository declares no licence (no LICENSE file); the adapted routines are included here with attribution to the authors, who are asked to contact us if they object.
* Where: `mpu/render_utils.py`: the camera setup (`projection`, `translate`, `random_rotation`,
  the `Renderer` class with NIE-style random cameras and head-light diffuse shading through
  nvdiffrast), the uniform graph Laplacian helpers (`compute_edges`, `laplacian_uniform`), the
  normal operators (`face_normals_t`, `vertex_normals_t`) and the multi-scale image loss
  (`gauss_kernel`, `build_pyramid`, `img_loss`, "Multi-scale L2 as in NIE").
* Existing attribution in the code: the module docstring "Differentiable rendering utilities
  (nvdiffrast) and mesh operators, ported from NIE." and the comments "Multi-scale L2 as in NIE"
  and "NIE-style cameras".
* The evaluation meshes of NIE (armadillo, genus6, kangaroo, dino, vbunny, rind) are not
  redistributed; see `data/README.md`.

### Large Steps in Inverse Rendering of Geometry (Nicolet et al. 2021)

* Upstream: https://github.com/rgl-epfl/large-steps-pytorch
* Licence: BSD-3-Clause (Copyright (c) 2021 Baptiste Nicolet, Alec Jacobson, Wenzel Jakob)
* Where: `largesteps_solve` in `mpu/diff.py`: the (I + lambda L) preconditioner with the uniform
  graph Laplacian L = D - A, solved here by conjugate gradients instead of the upstream Cholesky
  factorisation. The docstring reads "Solve (I + lam * L) u = g with the uniform graph Laplacian
  L = D - A by conjugate gradients (Nicolet et al. 2021 'Large Steps' preconditioning)". The
  idea and the operator are from the Large Steps paper; no upstream file is copied verbatim.

## Dependencies (not redistributed)

| Package | Licence | Used for | Notes |
|---|---|---|---|
| nvdiffrast | NVIDIA Source Code License (non-commercial research use) | `mpu/render_utils.py`, inverse rendering (Table 4) | Needed only for the inverse-rendering task. Check the NVIDIA licence terms before any commercial use. |
| pymeshlab | GPL-3 | `scripts/data/make_watertight.py` only | Never imported by the `mpu` library or by the run scripts. The preprocessing script is a separate, optional tool. |
| libigl (python bindings, `igl`) | MPL-2.0 | winding-number IoU, curvature, point-mesh distance (`mpu/metrics.py`, `mpu/data.py`, `mpu/diff.py`) | |
| Taichi | Apache-2.0 | block-sparse grids and kernels (`mpu/core.py`, `mpu/recon.py`) | |
| PyTorch | BSD-3-Clause | everything | |
| PyTorch3D | BSD-3-Clause | `knn_points`, GPU marching cubes (`mpu/recon.py`, `mpu/diff.py`) | No PyPI wheels; built from source. |
| Open3D | MIT | screened Poisson resampling (`mpu/poisson_worker.py`, `mpu/diff.py`, `scripts/run_deform.py`) | |
| trimesh | MIT | mesh I/O, components, repair | |
| scikit-image | BSD-3-Clause | marching-cubes fallback, `make_watertight.py` | |
| NumPy, SciPy | BSD-3-Clause | | |
| Pillow | MIT-CMU (HPND) | figure composition (`tools/`) | |

The paper itself (ACM Transactions on Graphics 43(6), SIGGRAPH Asia 2024) and the data sets it
uses have their own terms; see `data/README.md` for the data.
