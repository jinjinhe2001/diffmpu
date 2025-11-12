"""Differentiable moving-particle optimisation with the MPU representation (paper Algorithm 3).

Three tasks (--task):
  chamfer   Chamfer-loss SDF optimisation from a sphere against GT surface samples (paper Table 2)
  sdfgrid   reconstruction from an under-sampled SDF grid (paper Table 3)
  render    inverse rendering from 100 silhouette + head-light images with nvdiffrast (paper Table 4)

python scripts/run_diff.py --task chamfer --mesh data/armadillo.ply [--out runs/x] [--iters 1500 ...]

--out defaults to runs/<task>/<mesh stem>[_s<seed>]. The output directory receives config.json ({"args", "cfg"}),
mesh_<it>.ply snapshots, particles.pt (the pre-final sample particles) and result.json ({"result": {...}, "log": [...]}).
The exact commands of Tables 2, 3 and 4 are in recipes/table2_chamfer.jsonl, table3_sdfgrid.jsonl and
table4_render.jsonl (run them with scripts/launch.py). Table 4 uses a two-stage protocol: the recipe run saves its
pre-final particles, and the corrected late flow of recipes/table4_final_flow.jsonl is then run on them with
tools/resume_eval.py, which re-uses this script's parser (build_parser()) so that every flag can be overridden.
Flags in the "advanced / experimental" group are not used by any recipe. The flags in the "runtime" group (memory, logging,
output paths) do not change the optimisation, apart from --seed and the floating-point summation order of --view_chunk
(results within run-to-run noise).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import taichi as ti


def build_parser():
    ap = argparse.ArgumentParser(description="Differentiable moving-particle MPU optimisation: --task chamfer (Table 2), sdfgrid (Table 3), render (Table 4).")
    rt = ap.add_argument_group("runtime (memory / logging / output; these do not change the optimisation, apart from --seed and the floating-point summation order of --view_chunk)")
    adv = ap.add_argument_group("advanced / experimental (not used by any recipe)")
    ap.add_argument("--task", choices=["chamfer", "render", "sdfgrid"], default="chamfer")
    ap.add_argument("--sdf_grid_res", type=int, default=50)
    ap.add_argument("--sdf_K", type=int, default=4)
    ap.add_argument("--sdf_mode", choices=["normal", "rfs", "sphere", "interp"], default="normal")
    adv.add_argument("--sdf_interp", choices=["linear", "cubic"], default="linear")
    adv.add_argument("--sdf_mode2", choices=["same", "normal", "rfs", "sphere", "interp"], default="same", help="SDF flow mode after lr_switch")
    adv.add_argument("--sdf_sphere_eps", type=float, default=0.02)
    adv.add_argument("--sdf_sphere_normal", type=int, default=1)
    adv.add_argument("--sdf_empty_w", type=float, default=1.0)
    adv.add_argument("--sdf_keep_band", type=float, default=0.0, help=">0: keep only grid samples with |sdf| below this")
    ap.add_argument("--mesh", required=True)
    rt.add_argument("--out", default=None, help="output directory (default: runs/<task>/<mesh stem>[_s<seed>])")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--n_sp", type=int, default=500_000)
    ap.add_argument("--n_gt", type=int, default=500_000)
    ap.add_argument("--init_radius", type=float, default=0.3)
    ap.add_argument("--fine_res", type=int, default=512)
    ap.add_argument("--num_levels", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    rt.add_argument("--ti_mem", type=float, default=16.0, help="Taichi up-front GPU allocation in GB (grid + particle fields; PyTorch memory comes on top); lower it if Taichi fails to allocate")
    adv.add_argument("--final_levels", type=int, default=0, help=">0: re-fit the final particles with this many levels (Table-1 style fine MPU) before evaluation")
    adv.add_argument("--final_fine_res", type=int, default=0, help="finest grid resolution of the final re-fit (0 = keep base_res, i.e. fine_res * 2^(final_levels - num_levels))")
    adv.add_argument("--final_thresholds", default="", help="comma-separated curvature thresholds of the final re-fit (default: 85 * 2^-(L-1-l))")
    adv.add_argument("--final_max_fp", type=int, default=2 ** 21)
    ap.add_argument("--final_n_sp", type=int, default=0, help=">0: number of particles drawn by the final resample")
    adv.add_argument("--final_recompute_curv", type=int, default=1)
    ap.add_argument("--sdf_bbox", type=float, default=0.0, help=">0: the coarse SDF grid spans the shape bounding box enlarged by this factor per axis (0: [-1,1]^3)")
    rt.add_argument("--view_chunk", type=int, default=0, help="render task: rasterise / back-propagate the views in chunks of this size with gradient accumulation (same gradient; peak memory / (views/chunk)); 0 = all views at once")
    rt.add_argument("--profile", type=int, default=0, help="1: print the mean per-phase wall time (synchronised) of the loop at every log line")
    rt.add_argument("--lazy_far", type=int, default=1, help="1: compute the winding-number far field only when an off-particle query needs it (identical results; saves the far field in every chamfer / sdfgrid iteration)")
    rt.add_argument("--poisson_safe", type=int, default=1, help="1: run Open3D Poisson in a child process and retry with jitter on a crash (bad average roots segfault)")
    ap.add_argument("--final_flow_iters", type=int, default=0, help=">0: flow iterations after the final resample (late-phase parameters, no resample)")
    ap.add_argument("--final_n_gt", type=int, default=0, help=">0: GT points used by the final flow (chamfer task)")
    ap.add_argument("--final_flow_anneal", type=float, default=0.0, help=">0: geometric decay of the step speed (speed_cells2 / lr2) over the final flow down to this factor")
    adv.add_argument("--final_flow_mst", type=int, default=1, help="re-run the MST normal orientation after the final flow")
    adv.add_argument("--final_poisson_depth", type=int, default=0, help=">0: Poisson depth of the final resample only")
    adv.add_argument("--final_poisson_trim", type=float, default=-1.0, help=">=0: density trim of the final resample only")
    adv.add_argument("--final_thin_boost", type=float, default=0.0, help=">0: curvature = max(curv, boost / local thickness) in the final re-fit (Table-1 thin-structure boost)")
    adv.add_argument("--final_thin_level_offset", type=int, default=3)
    adv.add_argument("--lr", type=float, default=0.1)
    adv.add_argument("--lr2", type=float, default=0.1)
    ap.add_argument("--lr_switch", type=int, default=600)
    ap.add_argument("--max_step_cells", type=float, default=1.0, help="max displacement per step in finest cells")
    ap.add_argument("--max_step_cells2", type=float, default=0.5, help="max displacement per step after lr_switch")
    ap.add_argument("--pull_weight", type=float, default=1.0)
    ap.add_argument("--far_delete_cells", type=float, default=0.0, help=">0: after lr_switch delete sample particles farther than this many cells from any target point")
    adv.add_argument("--clip_pct", type=float, default=98.0, help="clip velocity magnitudes above this percentile")
    ap.add_argument("--speed_cells", type=float, default=0.0, help=">0: normalise so the speed_pct percentile moves this many finest cells per step")
    ap.add_argument("--speed_cells2", type=float, default=0.1)
    adv.add_argument("--speed_pct", type=float, default=90.0)
    adv.add_argument("--smooth_K", type=int, default=16)
    ap.add_argument("--smooth_level_offset", type=int, default=3)
    ap.add_argument("--smooth_level_offset2", type=int, default=3, help="offset after lr_switch; -1 disables smoothing")
    adv.add_argument("--smooth_mode", type=int, default=2, help="1: K nearest, 2: all stored within radius")
    adv.add_argument("--smooth_radius_cells", type=float, default=1.5)
    adv.add_argument("--pca_normals", type=int, default=1)
    adv.add_argument("--normal_mode", choices=["pca", "field"], default="field")
    adv.add_argument("--pca_K", type=int, default=16)
    ap.add_argument("--collide_cells", type=float, default=1.5)
    ap.add_argument("--collide_dot", type=float, default=-0.5)
    adv.add_argument("--carry_curvature", type=int, default=1)
    adv.add_argument("--curv_source", choices=["pca", "mpu"], default="pca")
    adv.add_argument("--curvature_max", type=float, default=400.0)
    ap.add_argument("--resample_every", type=int, default=20)
    ap.add_argument("--resample_mode", choices=["mc", "poisson", "dpsr", "none"], default="mc")
    adv.add_argument("--resample_mode2", choices=["same", "mc", "poisson", "dpsr", "none"], default="same", help="resample mode after lr_switch")
    adv.add_argument("--resample_res", type=int, default=512)
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--poisson_trim", type=float, default=0.01)
    ap.add_argument("--dpsr_res", type=int, default=256)
    ap.add_argument("--dpsr_sig", type=float, default=2.0)
    ap.add_argument("--min_comp_frac", type=float, default=0.01)
    adv.add_argument("--kick_every", type=int, default=0, help="SAP-style random perturbation every N iterations (before lr_switch)")
    ap.add_argument("--densify_every", type=int, default=0, help="clone the fastest densify_frac particles ahead of the front every N iterations")
    ap.add_argument("--densify_frac", type=float, default=0.05)
    adv.add_argument("--densify_step", type=float, default=2.0, help="offset of clones in finest cells")
    adv.add_argument("--densify_cap", type=float, default=1.5, help="max particle count as a multiple of n_sp")
    ap.add_argument("--normals_from_field", type=int, default=0)
    ap.add_argument("--hull_w", type=float, default=0.0, help=">0: dense silhouette (visual hull) displacement weight for particles outside the target masks")
    adv.add_argument("--hull_max_cells", type=float, default=1.0)
    ap.add_argument("--hull_margin", type=float, default=1.0, help="pixels outside the target silhouette before the hull term acts")
    adv.add_argument("--hull_start", type=int, default=0)
    ap.add_argument("--orient_mst", type=int, default=0, help=">0: globally consistent normal orientation (MST propagation over the K-NN graph) before every Poisson resample and before the final fit")
    ap.add_argument("--orient_mst_resample", type=int, default=1)
    adv.add_argument("--orient_mst_signed", type=int, default=0, help="1: signed normal affinity in the MST weights (NOT recommended: fails to flip inverted patches)")
    adv.add_argument("--orient_thin", type=int, default=0, help=">0: flip normals that face a nearby parallel particle sheet (probe up to this many finest cells); fixes thin-shell orientation")
    adv.add_argument("--render_mode", choices=["mpu", "sap"], default="mpu", help="sap = Shape-As-Points path: DPSR -> MC (surrogate gradient) -> nvdiffrast, Adam on particle positions+normals")
    adv.add_argument("--sap_lr", type=float, default=2e-3, help="Adam lr on particle positions in [0,1] grid units before lr_switch")
    adv.add_argument("--sap_lr2", type=float, default=5e-4)
    adv.add_argument("--sap_nlr_scale", type=float, default=1.0, help="normal lr = sap_lr * scale")
    ap.add_argument("--hull_carve", type=int, default=0, help="1: intersect the implicit field with the visual hull of the target masks (space carving) for MC, resampling and evaluation")
    adv.add_argument("--hull_delete_px", type=float, default=0.0, help=">0: delete particles more than hull_margin+this many pixels outside the target silhouette in any view")
    ap.add_argument("--mesh_normals_from", type=int, default=0, help=">0: switch to mesh vertex normals (with gradient) from this iteration")
    adv.add_argument("--densify_remove", type=int, default=0, help="1: move (not clone) the fastest particles ahead")
    adv.add_argument("--kick_cells", type=float, default=0.5)
    adv.add_argument("--n_sp_start", type=int, default=0, help=">0: start with fewer sample particles and grow to n_sp at resamples")
    adv.add_argument("--grow_until", type=int, default=0)
    # rendering
    ap.add_argument("--views", type=int, default=100)
    ap.add_argument("--test_views", type=int, default=20)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--mc_res", type=int, default=175)
    adv.add_argument("--mc_jitter", type=int, default=1)
    adv.add_argument("--mc_jitter_range", type=int, default=3, help="MC resolution jitter +-range per iteration (dithers the MC discretisation)")
    adv.add_argument("--vel_K", type=int, default=5)
    adv.add_argument("--laplace_lam", type=float, default=0.0)
    ap.add_argument("--ls_lambda", type=float, default=19.0, help="Large-Steps preconditioner lambda (0 = off)")
    adv.add_argument("--ls_iters", type=int, default=50)
    ap.add_argument("--ls_lambda2", type=float, default=19.0)
    ap.add_argument("--mask_w", type=float, default=0.0, help="weight of an extra silhouette (mask) loss")
    ap.add_argument("--photo_w", type=float, default=1.0)
    ap.add_argument("--photo_start", type=int, default=0, help="iteration from which the photometric loss is enabled")
    ap.add_argument("--final_resample", choices=["none", "dpsr", "mc", "poisson"], default="dpsr")
    adv.add_argument("--laplace_lam2", type=float, default=0.0)
    # eval / io
    ap.add_argument("--eval_mc_res", type=int, default=512)
    ap.add_argument("--iou_res", type=int, default=256)
    ap.add_argument("--chamfer_n", type=int, default=1_000_000)
    rt.add_argument("--save_every", type=int, default=100)
    rt.add_argument("--save_particles_every", type=int, default=0, help=">0: save the sample particles (positions/normals/curvature) every N iterations as particles_<it>.pt (for evolution figures)")
    rt.add_argument("--log_every", type=int, default=10)
    rt.add_argument("--seed", type=int, default=0)
    adv.add_argument("--scale_factor", type=float, default=1.2)
    return ap


def main(args):
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.mesh))[0]
        args.out = os.path.join("runs", args.task, stem + (f"_s{args.seed}" if args.seed != 0 else ""))

    ti.init(arch=ti.cuda, device_memory_GB=args.ti_mem, log_level=ti.WARN, random_seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from mpu.recon import default_config
    from mpu.data import load_mesh, normalize
    from mpu.diff import DiffMPU
    from mpu.sap import step_sap
    import trimesh

    cfg = default_config()
    cfg["n_sp"] = max(int(args.n_sp * (args.densify_cap if args.densify_every > 0 else 1.0)), int(args.final_n_sp)) + 16
    cfg["num_levels"] = args.num_levels
    cfg["base_res"] = args.fine_res // (2 ** (args.num_levels - 1))
    cfg["thresholds"] = [0.0] + [85.0 * 2.0 ** (-(args.num_levels - 1 - l)) for l in range(1, args.num_levels)]
    cfg["aux_shells"] = [0.25, 0.5, 0.75]
    cfg["n_proj"] = 2
    cfg["max_fp"] = 2 ** 19
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except Exception:
            raise SystemExit(f"--set {k}: value must be JSON (no spaces inside lists, true/false), got {v!r}; valid keys: {sorted(default_config())}")
    os.makedirs(args.out, exist_ok=True)
    json.dump({"args": vars(args), "cfg": cfg}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print("[cfg]", json.dumps(cfg), flush=True)

    V, F = load_mesh(args.mesh)
    V, _, _ = normalize(V, scale_factor=args.scale_factor)
    print(f"[data] {args.mesh}: V {V.shape[0]} F {F.shape[0]}", flush=True)

    D = DiffMPU(args, cfg, V, F)
    D.profile = bool(args.profile)
    print(f"[init] sps {D.sp_pos.shape[0]} fps {D.R.mpu.fp_stats()}", flush=True)
    t0 = time.time()
    last_log = 0
    for it in range(1, args.iters + 1):
        info = step_sap(D, it) if (args.task == "render" and args.render_mode == "sap") else D.step(it)
        if it % args.log_every == 0 or it == 1:
            print(f"[it {it:5d}] loss {info['loss']:.3e} psnr {info['psnr'] if info['psnr'] is None else round(info['psnr'], 2)} "
                  f"sps {info['n_sp']} fps {info['n_fp']} del {info['n_del']} step {info['mean_step']:.2e} "
                  f"t_vel {info['t_vel']:.2f} t_build {info['t_build']:.2f} elapsed {time.time()-t0:.0f}s", flush=True)
            if args.profile and getattr(D, "prof", None):
                k = max(it - last_log, 1)
                print(f"[prof] mean s/it over {k} its: " + " ".join(f"{key} {v / k:.3f}" for key, v in D.prof.items()), flush=True)
                D.prof = {}
            last_log = it
        if args.save_particles_every and (it % args.save_particles_every == 0 or it == 1):
            torch.save({"sp_pos": D.sp_pos.cpu(), "sp_nrm": D.sp_nrm.cpu(), "sp_cur": D.sp_cur.cpu(), "it": it, "far_winding_area": D.cfg.get("far_winding_area")}, os.path.join(args.out, f"particles_{it:05d}.pt"))
        if args.save_every and it % args.save_every == 0:
            Vm, Fm = D.mesh(256)
            trimesh.Trimesh(Vm, Fm, process=False).export(os.path.join(args.out, f"mesh_{it:05d}.ply"))
    torch.save({"sp_pos": D.sp_pos.cpu(), "sp_nrm": D.sp_nrm.cpu(), "sp_cur": D.sp_cur.cpu(), "it": args.iters}, os.path.join(args.out, "particles.pt"))
    res = D.evaluate(args.out, mc_res=args.eval_mc_res, iou_res=args.iou_res, chamfer_n=args.chamfer_n)
    res["iters"] = args.iters
    res["time_total"] = time.time() - t0
    res["torch_peak_gb"] = float(torch.cuda.max_memory_allocated() / 2**30)
    res["torch_peak_reserved_gb"] = float(torch.cuda.max_memory_reserved() / 2**30)
    res["cuda_alloc_retries"] = int(torch.cuda.memory_stats().get("num_alloc_retries", 0))   # >0: the caching allocator hit the memory limit and had to free/re-malloc (slow)
    res["ti_mem_gb"] = float(args.ti_mem)
    res["task"] = args.task
    res["mesh"] = args.mesh
    json.dump({"result": res, "log": D.log}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    print("[done]", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main(build_parser().parse_args())
