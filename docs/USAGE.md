# Usage

The command-line scripts, their outputs, the MPU configuration keys and the Python API.

## The scripts

Three scripts are the entry points, one per task family. `--help` lists the flags in three groups: the ones the recipes
use, a `runtime` group (memory, logging, seeds) and an `advanced / experimental` group with variants that are not part
of any recipe.

### `scripts/run_recon.py` (Tables 1 and 6: direct reconstruction from oriented points)

```
python scripts/run_recon.py --mesh PATH [--out DIR] [--mc_res 512] [--iou_res 256] [--chamfer_n 1000000]
                            [--curv_mode robust|mesh|pca] [--no_metrics] [--world_coords] [--ti_mem 8]
                            [--set key=json ...]
```

Samples 3M oriented points from the mesh, builds the MPU (feature particle generation, least-squares quadric fit,
projection rounds, hole fixing, winding-number far field), extracts a marching-cubes mesh and evaluates IoU, Chamfer
and normal angle error against the input. Writes `config.json` (the MPU configuration after `--set`), `recon.ply`,
`fp_pos.npy` / `fp_radius.npy` (the feature particles) and `result.json` with `stats` (`n_fp`, `per_level`, `params`),
`iou`, `cl2_p2m`, `cl2_p2p_100k/500k/1m` (multiply by 1e5 for the convention of the tables), `nae_grad`, `nae_mesh_vn`,
`timings` and `normalize`. `--out` defaults to `runs/recon/<mesh stem>`.

### `scripts/run_diff.py` (Tables 2, 3, 4: differentiable particle flows)

```
python scripts/run_diff.py --task chamfer|sdfgrid|render --mesh PATH [--out DIR] [recipe flags ...] [--ti_mem 16] [--set key=json ...]
```

Starts from a sphere of sample particles, rebuilds the MPU every iteration, computes a velocity from the task loss
(Chamfer against target points; Newton projection onto an under-sampled SDF grid; nvdiffrast rendering of the
marching-cubes mesh against 100 target views) and periodically resamples the particles from a Poisson or DPSR surface.
The recipe flags per task are in the recipe files; the ones that matter most are `--iters`, `--n_sp`, `--fine_res`,
`--num_levels`, `--resample_every/--resample_mode`, `--lr_switch` (start of the fine phase), the step limits
`--max_step_cells(2)`, `--collide_cells`, and for the Chamfer task the dense final stage `--final_n_sp
--final_flow_iters --final_n_gt`. Writes `config.json` (`args` + `cfg`), `mesh_<it>.ply` every `--save_every`,
`particles.pt`, `final.ply` and `result.json` (`{"result": {iou, cl2_*, nae_*, psnr_train, psnr_test, n_fp,
time_total, torch_peak_gb, ...}, "log": [...]}`). `--out` defaults to `runs/<task>/<mesh stem>`. Table 4 has a second
stage, see below.

### `scripts/run_deform.py` (Table 5: deformation and rotation)

```
python scripts/run_deform.py --test deform|rotate [--out DIR] [--steps 500] [--reverse_at 250] [--fine_res 512]
                             [--n_sp 5000000] [--num_levels 4] [--eval_res 512] [--curvature_source pca] [--ti_mem 16]
                             [--set max_fp=1048576 init_scale=0.68 'init_mesh="data/Armadillo.ply"']
```

Advects sample particles with an analytic velocity field (Enright deformation, reversed at `--reverse_at`, or a rigid
rotation) and rebuilds the MPU every step; reports IoU and volume loss between the final and the initial surface.
Writes `mesh_0000.ply`, `mesh_<step>.ply`, `mesh_final.ply` and `result.json` (`iou`, `volume_loss_pct`, `n_fp_final`).

### MPU configuration (`--set key=json`)

The MPU itself is configured by a dictionary (`mpu.recon.default_config()`), overridden with `--set key=value` where the
value is JSON (no spaces inside lists). The keys a user touches:

| key | meaning |
|---|---|
| `base_res`, `num_levels` | coarsest grid resolution and number of levels; finest level = `base_res * 2^(num_levels-1)` |
| `thresholds` | per-level curvature threshold: a sample particle activates level l where its curvature exceeds `thresholds[l]` (0 = everywhere) |
| `n_sp` | capacity for sample particles (must be at least the number of points) |
| `max_fp` | capacity for feature particles |
| `n_proj` | projection rounds that move the feature particles onto the fitted surface (0 = fixed particles, Table 6) |
| `aux_shells` | off-surface auxiliary shells (fractions of the cell size) that stabilise the quadric fit |
| `weight_type`, `level_gain`, `alpha` | partition-of-unity kernel, level blending gain, support radius factor |
| `far_winding`, `far_stage1` | winding-number points for the sign of the empty cells, and the size of the coarse first pass |

### Outputs are in the normalised frame

All scripts centre the input mesh and scale it so that `max |coord| = 1 / 1.2` (`mpu.data.normalize`). Meshes written by
the scripts are in that frame; `result.json["normalize"]` holds `center` and `scale` (`x_world = x * scale + center`),
and `run_recon.py --world_coords` exports `recon.ply` in the input frame directly.

## Python API

```python
import torch, mpu
mpu.init(ti_mem_gb=4)                                           # Taichi init (CUDA); must precede any Reconstructor
pos, nrm, cur, meta = mpu.particles_from_mesh("data/Armadillo.ply", n=3_000_000)   # CUDA tensors (N,3), (N,3), (N,); meta: center, scale, surface_area
cfg = mpu.recon_config(fine_res=512, num_levels=4)              # the Table 1 recipe: base 64, thresholds [0,21.25,42.5,85], aux shells, n_proj 2
R = mpu.build_mpu(pos, nrm, cur, cfg=cfg, surface_area=meta["surface_area"])        # Reconstructor; PCA curvature, retries on "too many feature particles"
f, g = R.eval(torch.rand(10_000, 3, device="cuda") * 2 - 1, return_grad=True)      # f (N,) signed field, negative inside; g (N,3) analytic gradient
field = R.field(256)                                            # (256,256,256) values on the cell centres of [-1,1]^3
V, F = R.marching_cubes(512)                                    # numpy (M,3) float64 / (K,3) int64, normalised frame
V_world = mpu.to_world(V, meta)

# an oriented point cloud without a mesh (CUDA tensors in [-1,1]^3): the curvature is estimated by neighbourhood PCA
R = mpu.build_mpu(pos, nrm, cfg=mpu.recon_config(n_sp=len(pos)))
```

`mpu.Reconstructor(cfg)` is the class behind the facade: `build(pos, nrm, cur)` builds the MPU (`far=False` defers the
winding-number far field to the first query), `eval(P, return_grad)` evaluates field and gradient, `field(res)` samples
a grid, `marching_cubes(res)` extracts a mesh, and `R.mpu.fp_pos` / `R.mpu.fp_radius` are the feature particles (Taichi
fields; `R.mpu.fp_stats()` gives the counts). `mpu.DiffMPU` implements the differentiable tasks and takes the
`run_diff.py` argument namespace; it is driven through that script. `import mpu` itself imports neither torch nor
Taichi; the heavy modules load on first use.

## Runtime notes

* Run everything from the repository root: the scripts import `mpu` by path and the Poisson worker is spawned as
  `python -m mpu.poisson_worker`.
* Tables 2 and 3 spend about 70 % of their time in Open3D's screened Poisson reconstruction on the CPU (14-17 s per
  resample with 16 threads). `MPU_POISSON_THREADS=<n>` changes the thread count (more than 16 is slower);
  `--poisson_safe 0` runs it in-process (a rare Open3D crash then kills the run instead of one resample).
* `--profile 1` prints the per-phase wall time of the loop; `--lazy_far 0` restores the eager far field (the Chamfer and
  SDF-grid loops never read it, so the default only saves time).
* A warm MPU rebuild takes 0.1-0.2 s, a Table 2 / 3 iteration 0.1-0.15 s plus the Poisson resample, a Table 4 iteration
  with 100 views at 1536^2 about 0.5 s on an H100.

## Repository layout

```
mpu/        the library: core.py (Taichi multi-level grid, feature particles, partition-of-unity blending, far field),
            recon.py (Reconstructor), diff.py (DiffMPU, the differentiable tasks), api.py, data.py, metrics.py,
            render_utils.py, dpsr.py, sap.py, poisson_worker.py
scripts/    run_recon.py, run_diff.py, run_deform.py, launch.py, summarize.py, eval_final_mesh.py, smoke_test.py,
            data/make_watertight.py, setup_env.sh
tools/      resume_eval.py (second stage of the inverse-rendering runs), render_fps.py (feature-particle figures)
recipes/    one jsonl per table of the paper
data/       meshes (see data/README.md), ignored by git
runs/       outputs, ignored by git
```
