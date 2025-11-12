"""Direct SDF reconstruction of one mesh + metrics (paper Table 1 / 6 / 7).

Usage: python scripts/run_recon.py --mesh path.ply [--out out_dir] [--set key=value ...]

--out defaults to runs/recon/<mesh stem>. The output directory receives config.json, recon.ply (the marching-cubes
mesh of the reconstructed field), fp_pos.npy / fp_radius.npy (the feature particles) and result.json (metrics, timings
and the normalisation used, so that recon.ply can be mapped back to the input frame: x * scale + center; pass
--world_coords to export recon.ply directly in that frame). The Table 1 / 6 commands are in recipes/table1_stanford.jsonl
and recipes/table6_fixed_vs_moving.jsonl.
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


def parse_value(v):
    try:
        return json.loads(v)
    except Exception:
        return v


def build_parser():
    ap = argparse.ArgumentParser(description="Direct MPU reconstruction of one mesh from oriented sample particles (paper Alg. 2, Tables 1 / 6 / 7).")
    adv = ap.add_argument_group("advanced / experimental (not used by any recipe)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", default=None, help="output directory (default: runs/recon/<mesh stem>)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=json")
    ap.add_argument("--mc_res", type=int, default=512)
    ap.add_argument("--iou_res", type=int, default=256)
    ap.add_argument("--chamfer_n", type=int, default=1_000_000)
    adv.add_argument("--curv_smooth", type=int, default=0)
    ap.add_argument("--curv_mode", default="robust", choices=["robust", "mesh", "pca"], help="pca = neighbourhood PCA curvature of the sample particles (as in the differentiable pipeline)")
    adv.add_argument("--pca_K", type=int, default=16)
    adv.add_argument("--curvature_max", type=float, default=400.0)
    adv.add_argument("--curv_scale", type=float, default=0.004)
    adv.add_argument("--normal_mode", default="vertex", choices=["vertex", "face"])
    adv.add_argument("--normal_smooth", type=int, default=1)
    ap.add_argument("--thin_boost", type=float, default=0.0, help=">0: curvature = max(curv, thin_boost / local thickness) (Table 6 recipe)")
    adv.add_argument("--thin_level_offset", type=int, default=3)
    adv.add_argument("--scale_factor", type=float, default=1.2)
    adv.add_argument("--seed", type=int, default=0)
    adv.add_argument("--save_mesh", type=int, default=1)
    ap.add_argument("--no_metrics", action="store_true")
    adv.add_argument("--query_timing", action="store_true", help="time 128/512/1024^3 queries (Table 7)")
    adv.add_argument("--debug", action="store_true")
    ap.add_argument("--time_rebuild", action="store_true", help="rebuild once more to measure warm timings (Table 6 recipe)")
    ap.add_argument("--ti_mem", "--device_memory_GB", dest="ti_mem", type=float, default=14.0,
                    help="Taichi up-front GPU allocation in GB (grid + particle fields; 1-2 GB suffice for Table 1, PyTorch memory comes on top); lower it if Taichi fails to allocate")
    ap.add_argument("--world_coords", action="store_true", help="export recon.ply in the input mesh frame, x * scale + center (fp_pos.npy / fp_radius.npy stay in the normalised [-1,1]^3 frame)")
    return ap


def main(args):
    if args.out is None:
        args.out = os.path.join("runs", "recon", os.path.splitext(os.path.basename(args.mesh))[0])

    ti.init(arch=ti.cuda, debug=args.debug, device_memory_GB=args.ti_mem, random_seed=args.seed,
            log_level=ti.WARN)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from mpu.recon import Reconstructor, default_config
    from mpu.data import load_mesh, normalize, make_sample_particles
    from mpu import metrics as M

    cfg = default_config()
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    os.makedirs(args.out, exist_ok=True)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print("[cfg]", json.dumps(cfg), flush=True)

    t0 = time.time()
    V, F = load_mesh(args.mesh)
    V, center, scale = normalize(V, scale_factor=args.scale_factor)
    print(f"[data] {args.mesh}: {V.shape[0]} verts, {F.shape[0]} faces, load {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    sp_pos, sp_nrm, sp_cur = make_sample_particles(V, F, cfg["n_sp"], curv_smooth=args.curv_smooth, seed=args.seed,
                                                   curv_mode="robust" if args.curv_mode == "pca" else args.curv_mode, normal_mode=args.normal_mode,
                                                   normal_smooth=args.normal_smooth, curv_scale=args.curv_scale)
    cur_np = sp_cur.cpu().numpy()
    print(f"[data] sampled {cfg['n_sp']} particles in {time.time()-t0:.1f}s; curvature quantiles "
          f"50/90/99/99.9%: {np.percentile(cur_np, [50, 90, 99, 99.9]).round(2).tolist()}", flush=True)

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cfg["surface_area"] = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
    R = Reconstructor(cfg)
    if args.curv_mode == "pca":
        R.mpu.bin_only(sp_pos, sp_nrm, sp_cur, cfg["thresholds"])
        sp_cur = R.mpu.sp_curvature_pca(K=args.pca_K, device="cuda").clamp(0, args.curvature_max)
        print(f"[data] PCA curvature (K={args.pca_K}): mean {float(sp_cur.mean()):.2f} median {float(sp_cur.median()):.2f} p90 {float(torch.quantile(sp_cur, 0.9)):.2f}", flush=True)
    if args.thin_boost > 0:
        m = R.mpu
        m.bin_only(sp_pos, sp_nrm, sp_cur, cfg["thresholds"])
        lvl = max(m.num_levels - 1 - args.thin_level_offset, 0)
        thick = m.sp_thickness(lvl)
        boost = args.thin_boost / thick.clamp_min(1e-4)
        boost = torch.where(thick > 1e8, torch.zeros_like(boost), boost)
        sp_cur = torch.maximum(sp_cur, boost)
        print(f"[data] thin boost at level {lvl}: thickness quantiles 1/10/50%: "
              f"{torch.quantile(thick[thick < 1e8][:1000000], torch.tensor([0.01, 0.1, 0.5], device=thick.device)).tolist() if (thick < 1e8).any() else 'none'}; "
              f"frac thin<2cells: {(thick < 2 * 2.0 / (cfg['base_res'] * 2 ** lvl)).float().mean().item():.3f}", flush=True)
    for attempt in range(4):
        try:
            stats = R.build(sp_pos, sp_nrm, sp_cur)
            break
        except RuntimeError as e:
            if "too many feature particles" not in str(e) or attempt == 3:
                raise
            cfg["thresholds"] = [t * 2.0 for t in cfg["thresholds"]]
            print(f"[recon] {e}; retrying with thresholds x2 -> {cfg['thresholds']}", flush=True)
    result = {"mesh": args.mesh, "stats": stats, "cfg": cfg, "timings": {}, "threshold_retries": attempt,
              "normalize": {"center": [float(c) for c in np.asarray(center).reshape(-1)], "scale": float(scale), "scale_factor": args.scale_factor}}

    # marching cubes
    field = R.field(args.mc_res)
    reV, reF = R.marching_cubes(args.mc_res, field=field)
    print(f"[mc] {reV.shape[0]} verts {reF.shape[0]} faces; query {R.timings.get(f'query_{args.mc_res}', 0):.2f}s "
          f"mc {R.timings.get(f'mc_{args.mc_res}', 0):.2f}s", flush=True)
    if args.save_mesh:
        import trimesh
        V_out = reV * scale + center if args.world_coords else reV   # back to the input mesh frame if requested
        trimesh.Trimesh(V_out, reF, process=False).export(os.path.join(args.out, "recon.ply"))
        np.save(os.path.join(args.out, "fp_pos.npy"), R.mpu.fp_pos.to_numpy()[:stats["n_fp_alloc"]])
        np.save(os.path.join(args.out, "fp_radius.npy"), R.mpu.fp_radius.to_numpy()[:stats["n_fp_alloc"]])

    if not args.no_metrics:
        t0 = time.time()
        # IoU on a grid: reuse the field if mc_res == iou_res else re-evaluate
        if args.iou_res == args.mc_res:
            f_iou = field.reshape(-1)
        else:
            f_iou = R.field(args.iou_res).reshape(-1)
        Q = M.grid_points(args.iou_res, device="cuda").double().cpu().numpy()
        occ_gt = M.winding_occupancy(V, F, Q)
        occ_re = (f_iou < 0).cpu().numpy()
        result["iou"] = M.iou(occ_gt, occ_re)
        result["iou_mc_mesh"] = M.iou(occ_gt, M.winding_occupancy(reV, reF, Q)) if reV.shape[0] > 0 else 0.0
        result["timings"]["iou"] = time.time() - t0
        print(f"[metric] IoU(field) {result['iou']:.5f}  IoU(mc mesh) {result['iou_mc_mesh']:.5f} ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()

        def grad_fn(P):
            _, g = R.eval(P, return_grad=True)
            return g

        cm = M.chamfer_and_nae(V, F, reV, reF, n_samples=args.chamfer_n, grad_fn=grad_fn)
        result.update(cm)
        result["timings"]["chamfer"] = time.time() - t0
        print(f"[metric] CL2(p2m) {cm['cl2_p2m']*1e5:.3f}e-5  CL2(p2p 100k) {cm['cl2_p2p_100k']*1e5:.3f}e-5  "
              f"CL2(p2p 1m) {cm['cl2_p2p_1m']*1e5:.3f}e-5  NAE(mesh) {cm['nae_mesh']:.2f}  NAE(grad) {cm['nae_grad']:.2f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    if args.query_timing:
        for r in (128, 512, 1024):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = R.field(r)
            torch.cuda.synchronize()
            result["timings"][f"query_{r}"] = time.time() - t0
            print(f"[timing] query {r}^3: {time.time()-t0:.3f}s", flush=True)

    if args.time_rebuild:
        R2 = Reconstructor(cfg)
        R2.mpu = R.mpu  # reuse allocated taichi fields
        R2.timings = {}
        torch.cuda.synchronize()
        R2.build(sp_pos, sp_nrm, sp_cur, verbose=False)
        torch.cuda.synchronize()
        result["timings_warm"] = R2.timings
        print("[timing] warm rebuild:", json.dumps({k: round(v, 3) for k, v in R2.timings.items()}), flush=True)

    result["timings"].update(R.timings)
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    print("[done]", json.dumps({k: v for k, v in result.items() if k not in ("cfg",)}), flush=True)
    return result


if __name__ == "__main__":
    main(build_parser().parse_args())
