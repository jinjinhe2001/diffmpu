"""Thin Python facade over the direct MPU reconstruction (paper Algorithm 2).

Every function here mirrors a block of scripts/run_recon.py (the Table 1 / 6 / 7 driver); the mirrored
lines are cited in comments ("run_recon.py L..") so the facade and the script cannot drift apart silently.
The intended calling order is::

    mpu.init(...)                                                    # once per process
    pos, nrm, cur, meta = mpu.particles_from_mesh(mesh_or_VF, n=...)
    R = mpu.build_mpu(pos, nrm, cur, cfg=mpu.recon_config(...), surface_area=meta["surface_area"])
    f, grad = R.eval(P)          # or  R.field(res)  /  V, F = R.marching_cubes(res)
    V_world = mpu.to_world(V, meta)

Frames and units. Everything is reconstructed in the normalised frame of mpu.data.normalize(): the mesh is
centred on its bounding-box centre and scaled so that max |coord| = 1 / scale_factor (0.8333 for the default
1.2), i.e. the surface lies inside the MPU domain [-1, 1]^3. Curvatures, thresholds, feature-particle radii,
field values and gradients are all in these normalised units; to_world() maps coordinates back to the input
frame (distances scale by meta["scale"]).

Tensor shapes. pos / nrm are torch float32 CUDA tensors of shape (N, 3), cur is (N,) (curvature estimate
2|H| compared against cfg["thresholds"] to activate the grid levels). R.eval(P) takes an (M, 3) CUDA tensor
and returns (f (M,), grad (M, 3) or None); R.field(res) returns a (res, res, res) tensor sampled on the cell
centres of [-1, 1]^3; R.marching_cubes(res) returns numpy V (K, 3) float64 and F (T, 3) int64.
Sign convention: f < 0 inside the surface, f > 0 outside.

torch and taichi are imported lazily inside the functions, so importing this module (for help() or
documentation) works on a machine without CUDA.
"""
import os

import numpy as np

__all__ = ["init", "recon_config", "default_thresholds", "particles_from_mesh", "build_mpu", "to_world"]


def init(ti_mem_gb=8.0, seed=0, debug=False):
    """Initialise taichi on CUDA and seed torch / numpy exactly like scripts/run_recon.py.

    Call once per process before Reconstructor / build_mpu. Taichi fields are allocated from a device pool of
    ti_mem_gb GB (run_recon.py --device_memory_GB defaults to 14; 8 GB is enough for the Table 1 recipes with
    max_fp = 2**20, 2 GB for scripts/smoke_test.py). seed feeds taichi random_seed, torch.manual_seed and
    np.random.seed (run_recon.py --seed, default 0). debug=True enables taichi bound checks (slow).
    """
    import torch
    import taichi as ti
    # run_recon.py
    ti.init(arch=ti.cuda, debug=bool(debug), device_memory_GB=float(ti_mem_gb), random_seed=int(seed),
            log_level=ti.WARN)
    torch.manual_seed(int(seed))    # run_recon.py
    np.random.seed(int(seed))       # run_recon.py


def default_thresholds(num_levels, top=85.0):
    """Per-level curvature activation thresholds: [0] + [top * 2**-(num_levels-1-l) for l in 1..num_levels-1].

    Level 0 (coarsest) is always active, the threshold doubles per level up to `top` at the finest level.
    4 levels -> [0, 21.25, 42.5, 85] (recipes/table1_stanford.jsonl), 3 levels -> [0, 42.5, 85] (Table 4).
    """
    num_levels = int(num_levels)
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    return [0.0] + [float(top) * 2.0 ** -(num_levels - 1 - l) for l in range(1, num_levels)]


def recon_config(fine_res=512, num_levels=4, thresholds=None, max_fp=2 ** 20, aux_shells=(0.25, 0.5, 0.75),
                 n_proj=2, n_sp=3_000_000, **overrides):
    """Config dict for Reconstructor: mpu.recon.default_config() updated with the Table 1 recipe.

    The defaults match recipes/table1_stanford.jsonl, i.e. run_recon.py
    --set aux_shells=[0.25,0.5,0.75] base_res=64 num_levels=4 thresholds=[0,21.25,42.5,85] n_proj=2 max_fp=1048576
    (run_recon.py apply the --set overrides on top of default_config(); n_sp stays at its default 3M).

    fine_res: resolution of the finest grid level; base_res = fine_res // 2**(num_levels-1) (512 -> 64 for 4 levels).
    thresholds: per-level curvature thresholds in normalised units; None -> default_thresholds(num_levels).
    max_fp: feature-particle capacity (build raises "too many feature particles" beyond it; build_mpu retries
      with doubled thresholds). aux_shells: auxiliary least-squares points at +-s * radius along the three axes.
    n_proj: Newton projection rounds (moving feature particles). n_sp: sample-particle capacity (build_mpu
      raises it to len(pos) if needed). **overrides: any other default_config() key (n_fix=..., far_winding=...,
      base_res=... wins over the computed one); unknown keys raise KeyError listing the valid keys.
    """
    from .recon import default_config
    cfg = default_config()                                           # run_recon.py
    num_levels = int(num_levels)
    fine_res = int(fine_res)
    if num_levels < 1 or fine_res % 2 ** (num_levels - 1) != 0:
        raise ValueError(f"fine_res={fine_res} must be a multiple of 2**(num_levels-1)={2 ** (num_levels - 1)}")
    cfg["base_res"] = fine_res // 2 ** (num_levels - 1)
    cfg["num_levels"] = num_levels
    th = default_thresholds(num_levels) if thresholds is None else [float(t) for t in thresholds]
    if len(th) != num_levels:
        raise ValueError(f"thresholds has {len(th)} entries, num_levels is {num_levels}")
    cfg["thresholds"] = th
    cfg["max_fp"] = int(max_fp)
    cfg["aux_shells"] = [float(s) for s in aux_shells]
    cfg["n_proj"] = int(n_proj)
    cfg["n_sp"] = int(n_sp)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise KeyError(f"unknown config key(s) {unknown}; valid keys: {sorted(cfg)}")
    cfg.update(overrides)                                            # run_recon.py (--set key=value)
    return cfg


def particles_from_mesh(mesh, n=3_000_000, seed=0, scale_factor=1.2, curv_mode="robust", normal_mode="vertex",
                        curv_smooth=0, normal_smooth=1, curv_scale=0.004, device="cuda"):
    """Load, normalise and sample a mesh into oriented sample particles (run_recon.py).

    mesh: a mesh file path (anything trimesh loads; load_mesh flips the faces if the signed volume is negative)
      or a tuple (V, F) of numpy arrays, V (n_v, 3) float, F (n_f, 3) int, in any frame (same orientation check).
    n: number of sample particles (area-weighted surface sampling with a torch generator seeded by `seed`).
    scale_factor: normalisation, max |coord| = 1 / scale_factor afterwards (run_recon.py --scale_factor 1.2).
    curv_mode: "robust" (igl quadric fitting over k-rings spanning ~curv_scale, run_recon.py default), "mesh"
      (cotangent Laplacian), "none" (cur = zeros) or "pca" (same as "robust" here; run_recon.py --curv_mode pca
      samples with "robust" at and replaces the curvature inside the build, = build_mpu(curvature="pca")).
    normal_mode: "vertex" = interpolated vertex normals smoothed over normal_smooth 1-ring passes (defaults
      "vertex", 1), "face" = flat face normals. curv_smooth: 1-ring smoothing passes of the curvature (default 0).

    Returns pos (n, 3), nrm (n, 3), cur (n,) torch float32 tensors on `device` in the normalised frame, and
    meta = {"center" (3,), "scale" float, "scale_factor", "surface_area" (triangle area sum of the normalised
    mesh, for build_mpu(surface_area=...)), "V" (normalised vertices), "F"}; to_world(X, meta) maps back.
    """
    from .data import load_mesh, normalize, make_sample_particles
    if isinstance(mesh, (str, bytes, os.PathLike)):
        V, F = load_mesh(os.fspath(mesh))                               # run_recon.py
    else:
        V, F = mesh
        V = np.ascontiguousarray(V, dtype=np.float64)
        F = np.ascontiguousarray(F, dtype=np.int64)
        # same orientation check as data.load_mesh (outward normals give the negative-inside convention)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol = float((np.cross(v1, v2) * v0).sum() / 6.0)
        if vol < 0:
            print(f"[load] negative signed volume ({vol:.3g}): flipping face orientation", flush=True)
            F = np.ascontiguousarray(F[:, [0, 2, 1]])
    V, center, scale = normalize(V, scale_factor=scale_factor)          # run_recon.py
    # run_recon.py ("pca" samples with "robust"; the PCA replacement happens in build_mpu)
    mode = "robust" if curv_mode == "pca" else curv_mode
    pos, nrm, cur = make_sample_particles(V, F, int(n), curv_smooth=int(curv_smooth), seed=int(seed),
                                          curv_mode=mode, normal_mode=normal_mode, normal_smooth=int(normal_smooth),
                                          curv_scale=float(curv_scale), device=device)
    # run_recon.py: triangle area sum of the normalised mesh (winding-number far field, cfg["surface_area"])
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    surface_area = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    meta = {"center": np.asarray(center, dtype=np.float64), "scale": float(scale), "scale_factor": float(scale_factor),
            "surface_area": surface_area, "V": V, "F": F}
    return pos, nrm, cur, meta


def build_mpu(pos, nrm, cur=None, cfg=None, curvature="pca", pca_K=16, curvature_max=400.0, surface_area=None,
              retries=3, verbose=True):
    """Build the MPU implicit surface from oriented sample particles (run_recon.py).

    pos, nrm: (N, 3) torch CUDA tensors in the normalised frame [-1, 1]^3; cur: (N,) curvature or None.
    cfg: recon_config() dict (default recon_config(), the Table 1 recipe). It is copied; the effective config
      (after the n_sp / surface_area / threshold-retry updates) is R.cfg. cfg["n_sp"] is raised to len(pos) if
      needed (Reconstructor allocates n_sp + 16 sample-particle slots, recon.py; run_recon.py samples
      exactly cfg["n_sp"] particles at, so no further margin is added).
    curvature: "pca" = run_recon.py --curv_mode pca (all recipes): bin the particles once with the given cur
      (zeros if None), then cur = R.mpu.sp_curvature_pca(K=pca_K).clamp(0, curvature_max) (). The binning
      read by the PCA does not depend on the curvature values, so cur=None gives the same result as the robust
      curvature run_recon.py passes. "given" = use cur as is (run_recon.py --curv_mode robust / mesh).
    surface_area: area of the sampled surface in the normalised frame (meta["surface_area"]); sets
      cfg["surface_area"], the per-point area of the winding-number far field (recon.py). None keeps
      cfg["surface_area"] (default 0.0 -> the far_winding_area = 4.0 fallback).
    retries: on RuntimeError "too many feature particles" (n_fp >= max_fp) every threshold is doubled and the
      build is retried, up to `retries` times (run_recon.py: range(4) attempts = 3 retries).
    Returns the Reconstructor R with R.last_stats = fp_stats() of the final build ({"n_fp", "n_fp_alloc",
      "per_level", "params"}), R.last_curvature = the (N,) curvature actually used and R.threshold_retries.
    """
    import torch
    from .recon import Reconstructor
    if curvature not in ("pca", "given"):
        raise ValueError(f"curvature must be 'pca' or 'given', got {curvature!r}")
    cfg = dict(recon_config() if cfg is None else cfg)
    n = int(pos.shape[0])
    cfg["n_sp"] = max(int(cfg["n_sp"]), n)
    if surface_area is not None:
        cfg["surface_area"] = float(surface_area)                      # run_recon.py
    if cur is None:
        if curvature != "pca":
            raise ValueError("cur is required when curvature='given'")
        cur = torch.zeros(n, dtype=torch.float32, device=pos.device)
    R = Reconstructor(cfg)                                              # run_recon.py (R.cfg is `cfg`)
    if curvature == "pca":                                              # run_recon.py
        R.mpu.bin_only(pos, nrm, cur, cfg["thresholds"])
        cur = R.mpu.sp_curvature_pca(K=int(pca_K), device=R.device).clamp(0, float(curvature_max))
        if verbose:
            print(f"[data] PCA curvature (K={int(pca_K)}): mean {float(cur.mean()):.2f} median {float(cur.median()):.2f} "
                  f"p90 {float(torch.quantile(cur, 0.9)):.2f}", flush=True)
    retries = int(retries)
    stats = None
    attempt = 0
    for attempt in range(retries + 1):                                  # run_recon.py
        try:
            stats = R.build(pos, nrm, cur, verbose=verbose)             # run_recon.py
            break
        except RuntimeError as e:
            if "too many feature particles" not in str(e) or attempt == retries:
                raise
            cfg["thresholds"] = [t * 2.0 for t in cfg["thresholds"]]    # run_recon.py (visible to R.build via R.cfg)
            print(f"[recon] {e}; retrying with thresholds x2 -> {cfg['thresholds']}", flush=True)
    R.last_stats = stats
    R.last_curvature = cur
    R.threshold_retries = attempt                                       # run_recon.py ("threshold_retries")
    return R


def to_world(V, meta):
    """Map normalised coordinates back to the input mesh frame: V * meta["scale"] + meta["center"].

    Inverse of normalize() (V_norm = (V_world - center) / scale). V is a numpy array or torch tensor of shape
    (..., 3): marching-cubes vertices, sample particles and feature-particle positions (R.mpu.fp_pos.to_numpy())
    are all in the normalised frame. Lengths (feature-particle radii, distances) scale by meta["scale"] alone.
    """
    center = np.asarray(meta["center"], dtype=np.float64).reshape(3)
    scale = float(meta["scale"])
    if isinstance(V, np.ndarray):
        return V * scale + center
    import torch
    if isinstance(V, torch.Tensor):
        return V * scale + torch.as_tensor(center, dtype=V.dtype, device=V.device)
    return np.asarray(V, dtype=np.float64) * scale + center
