"""Explicit velocity evolution tests (paper Table 5): 3D deformation field (Enright) with reversal,
and rigid body rotation. Sample particles are advected with RK4 (positions + normals), the MPU
surface is rebuilt every step, and every `resample_every` steps the sample particles are
regenerated from the reconstructed surface (marching cubes at the finest level).

Metrics: IoU (grid) and volume loss % between the final frame and the initial frame.

python scripts/run_deform.py --test deform --steps 500 --reverse_at 250 --fine_res 512 [--out runs/x]

--out defaults to runs/deform/<test>_<fine_res>. The output directory receives config.json, mesh_<step>.ply snapshots,
mesh_final.ply and result.json. The Table 5 commands are in recipes/table5_deform.jsonl (the rotation test loads the
Stanford armadillo through --set 'init_mesh="data/Armadillo.ply"' [init_scale=0.68]).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import taichi as ti

dev = "cuda"


def build_parser():
    ap = argparse.ArgumentParser(description="Explicit velocity evolution of MPU sample particles (paper Table 5): --test deform (Enright field with reversal) or rotate (rigid rotation).")
    ap.add_argument("--test", choices=["deform", "rotate"], default="deform")
    ap.add_argument("--out", default=None, help="output directory (default: runs/deform/<test>_<fine_res>)")
    ap.add_argument("--n_sp", type=int, default=5_000_000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--reverse_at", type=int, default=250, help="step index at which the deform flow is reversed")
    ap.add_argument("--resample_every", type=int, default=20, help="0 = never resample (pure Lagrangian)")
    ap.add_argument("--resample_res", type=int, default=512)
    ap.add_argument("--eval_res", type=int, default=512)
    ap.add_argument("--fine_res", type=int, default=512)
    ap.add_argument("--num_levels", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curvature_every", type=int, default=5, help="0 = only at init/resample")
    ap.add_argument("--curvature_max", type=float, default=400.0)
    ap.add_argument("--curvature_source", choices=["mpu", "mesh", "pca"], default="pca")
    ap.add_argument("--resample_mode", choices=["mc", "poisson", "none"], default="poisson")
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--poisson_points", type=int, default=2000000)
    ap.add_argument("--min_comp_frac", type=float, default=0.005)
    ap.add_argument("--project_sp", type=int, default=0, help="Newton-project sps onto the MPU surface at resample time")
    ap.add_argument("--carry_curvature", type=int, default=1, help="sample particles carry curvature computed from the MPU")
    ap.add_argument("--ti_mem", type=float, default=16.0, help="Taichi up-front GPU allocation in GB (grid + particle fields; 5M sample particles need the default 16, PyTorch memory comes on top); lower it if Taichi fails to allocate")
    return ap


# ------------------------------------------------------------------ velocity fields (unit cube coords)
def vel_enright(p, sign):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    pi = math.pi
    u = 2.0 * torch.sin(pi * x) ** 2 * torch.sin(2 * pi * y) * torch.sin(2 * pi * z)
    v = -torch.sin(2 * pi * x) * torch.sin(pi * y) ** 2 * torch.sin(2 * pi * z)
    w = -torch.sin(2 * pi * x) * torch.sin(2 * pi * y) * torch.sin(pi * z) ** 2
    return sign * torch.stack([u, v, w], dim=1)


def grad_vel_enright(p, sign):
    """Jacobian J[i,j] = d u_i / d x_j, shape (N,3,3)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    pi = math.pi
    sx, cx = torch.sin(pi * x), torch.cos(pi * x)
    sy, cy = torch.sin(pi * y), torch.cos(pi * y)
    sz, cz = torch.sin(pi * z), torch.cos(pi * z)
    s2x, c2x = torch.sin(2 * pi * x), torch.cos(2 * pi * x)
    s2y, c2y = torch.sin(2 * pi * y), torch.cos(2 * pi * y)
    s2z, c2z = torch.sin(2 * pi * z), torch.cos(2 * pi * z)
    J = torch.empty((p.shape[0], 3, 3), device=p.device)
    # u = 2 sx^2 s2y s2z
    J[:, 0, 0] = 4 * pi * sx * cx * s2y * s2z
    J[:, 0, 1] = 4 * pi * sx ** 2 * c2y * s2z
    J[:, 0, 2] = 4 * pi * sx ** 2 * s2y * c2z
    # v = -s2x sy^2 s2z
    J[:, 1, 0] = -2 * pi * c2x * sy ** 2 * s2z
    J[:, 1, 1] = -2 * pi * s2x * sy * cy * s2z
    J[:, 1, 2] = -2 * pi * s2x * sy ** 2 * c2z
    # w = -s2x s2y sz^2
    J[:, 2, 0] = -2 * pi * c2x * s2y * sz ** 2
    J[:, 2, 1] = -2 * pi * s2x * c2y * sz ** 2
    J[:, 2, 2] = -2 * pi * s2x * s2y * sz * cz
    return sign * J


def vel_rotate(p, sign):
    x, y = p[:, 0], p[:, 1]
    return sign * torch.stack([0.5 - y, x - 0.5, torch.zeros_like(x)], dim=1)


def grad_vel_rotate(p, sign):
    J = torch.zeros((p.shape[0], 3, 3), device=p.device)
    J[:, 0, 1] = -1.0
    J[:, 1, 0] = 1.0
    return sign * J


def normal_rate(J, n):
    """dn/dt = -(grad V)^T n + (n^T (grad V)^T n) n  (Ianniello & Di Mascio); for divergence-free flows
    the second term keeps |n| = 1 to first order."""
    JTn = torch.einsum("nji,nj->ni", J, n)  # (J^T n)_i = sum_j J[j,i] n_j
    return -JTn + (n * JTn).sum(1, keepdim=True) * n


def rk4_step(p, n, dt, vel_fn, grad_fn, sign):
    """Positions in [-1,1] coords; velocity defined in unit cube coords u = (p+1)/2, dp/dt = 2 * V(u)."""
    def f(pp, nn):
        u = pp * 0.5 + 0.5
        return 2.0 * vel_fn(u, sign), normal_rate(grad_fn(u, sign), nn)
    k1, s1 = f(p, n)
    k2, s2 = f(p + 0.5 * dt * k1, n + 0.5 * dt * s1)
    k3, s3 = f(p + 0.5 * dt * k2, n + 0.5 * dt * s2)
    k4, s4 = f(p + dt * k3, n + dt * s3)
    p = p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    n = n + dt * (s1 + 2 * s2 + 2 * s3 + s4) / 6.0
    return p, n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------ initial shape
def init_sphere(n, center, radius, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    v = torch.randn((n, 3), device=dev, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    c = torch.tensor(center, device=dev)
    return c + radius * v, v.clone()


def mpu_curvature(R, P, h, batch=2 ** 22):
    """|div(grad f / |grad f|)| by central differences (= 2|H| on the zero set)."""
    out = torch.empty(P.shape[0], device=P.device)
    for s in range(0, P.shape[0], batch):
        Pb = P[s:s + batch]
        div = torch.zeros(Pb.shape[0], device=P.device)
        for ax in range(3):
            e = torch.zeros(3, device=P.device)
            e[ax] = h
            _, gp = R.eval(Pb + e, return_grad=True)
            _, gm = R.eval(Pb - e, return_grad=True)
            npl = gp / gp.norm(dim=1, keepdim=True).clamp_min(1e-8)
            nmi = gm / gm.norm(dim=1, keepdim=True).clamp_min(1e-8)
            div += (npl[:, ax] - nmi[:, ax]) / (2 * h)
        out[s:s + batch] = div.abs()
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def mesh_volume(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.abs((np.cross(v1, v2) * v0).sum(1).sum()) / 6.0)


def occupancy(R, res):
    f = R.field(res).reshape(-1)
    return (f < 0).cpu().numpy()


def main(args):
    if args.out is None:
        args.out = os.path.join("runs", "deform", f"{args.test}_{args.fine_res}")

    ti.init(arch=ti.cuda, device_memory_GB=args.ti_mem, log_level=ti.WARN, random_seed=args.seed)
    torch.manual_seed(args.seed)
    from mpu.recon import Reconstructor, default_config, run_marching_cubes
    from mpu.data import sample_surface, vertex_normals_smooth, robust_curvature
    from mpu.diff import largest_components
    from mpu import metrics as M

    cfg = default_config()
    cfg["n_sp"] = args.n_sp
    cfg["num_levels"] = args.num_levels
    cfg["base_res"] = args.fine_res // (2 ** (args.num_levels - 1))
    cfg["thresholds"] = [0.0] + [85.0 * 2.0 ** (-(args.num_levels - 1 - l)) for l in range(1, args.num_levels)]
    cfg["aux_shells"] = [0.25, 0.5, 0.75]
    cfg["n_proj"] = 2
    cfg["far_winding_area"] = 4.0 * 3.141592653589793 * 0.3 ** 2  # sphere area; updated below for meshes
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except Exception:
            raise SystemExit(f"--set {k}: value must be JSON (no spaces inside lists, true/false), got {v!r}; valid keys: {sorted(default_config())}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args) | {"cfg": cfg}, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print("[cfg]", json.dumps(cfg), flush=True)

    if args.test == "deform":
        vel_fn, grad_fn = vel_enright, grad_vel_enright
        center_unit = (0.35, 0.35, 0.35)
        radius_unit = 0.15
    else:
        vel_fn, grad_fn = vel_rotate, grad_vel_rotate
        center_unit = (0.35, 0.5, 0.5)
        radius_unit = 0.15

    center = [2 * c - 1 for c in center_unit]
    radius = 2 * radius_unit
    if args.test == "rotate" and cfg.get("init_mesh"):
        from mpu.data import load_mesh, normalize
        V0, F0 = load_mesh(cfg["init_mesh"])
        V0, _, _ = normalize(V0, scale_factor=1.0 / float(cfg.get("init_scale", 0.6)))  # shape spans +-init_scale, centred on the axis
        sp_pos, sp_nrm, _, _ = sample_surface(V0, F0, args.n_sp, device=dev, seed=args.seed)
        _a, _b, _c = V0[F0[:, 0]], V0[F0[:, 1]], V0[F0[:, 2]]
        cfg["far_winding_area"] = float(0.5 * np.linalg.norm(np.cross(_b - _a, _c - _a), axis=1).sum())
    else:
        sp_pos, sp_nrm = init_sphere(args.n_sp, center, radius, args.seed)
    sp_cur = torch.zeros(args.n_sp, device=dev)
    h_fd = 0.5 * (2.0 / args.fine_res)

    R = Reconstructor(cfg)
    t0 = time.time()
    R.build(sp_pos, sp_nrm, sp_cur, verbose=False)
    if args.carry_curvature:
        sp_cur = R.mpu.sp_curvature_pca(K=16, device=dev).clamp(0, args.curvature_max) if args.curvature_source == "pca" else mpu_curvature(R, sp_pos, h_fd)
        R.build(sp_pos, sp_nrm, sp_cur, verbose=False)
        print(f"[init] curvature quantiles 50/90/99: {torch.quantile(sp_cur[:100000], torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist()}", flush=True)
    print(f"[init] built in {time.time()-t0:.1f}s: {R.mpu.fp_stats()}", flush=True)
    occ0 = occupancy(R, args.eval_res)
    V_init, F_init = R.marching_cubes(args.eval_res)
    vol0 = mesh_volume(V_init, F_init)
    vol_exact = 4.0 / 3.0 * math.pi * radius ** 3
    print(f"[init] volume {vol0:.6f} (exact sphere {vol_exact:.6f}, rel err {(vol0-vol_exact)/vol_exact*100:.3f}%)", flush=True)
    import trimesh
    trimesh.Trimesh(V_init, F_init, process=False).export(os.path.join(args.out, "mesh_0000.ply"))

    log = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        sign = 1.0
        if args.test == "deform" and step > args.reverse_at:
            sign = -1.0
        sp_pos, sp_nrm = rk4_step(sp_pos, sp_nrm, args.dt, vel_fn, grad_fn, sign)
        tb = time.time()
        R.build(sp_pos, sp_nrm, sp_cur, verbose=False)
        if args.carry_curvature and args.curvature_every > 0 and step % args.curvature_every == 0:
            if args.curvature_source == "pca":
                sp_cur = R.mpu.sp_curvature_pca(K=16, device=dev).clamp(0, args.curvature_max)
            else:
                sp_cur = mpu_curvature(R, sp_pos, h_fd).clamp(0, args.curvature_max)
        tb = time.time() - tb
        if args.resample_every > 0 and step % args.resample_every == 0 and step < args.steps and args.resample_mode != "none":
            # project current sps onto the MPU surface and reset normals from the gradient (paper 7.3)
            if args.project_sp:
                for _ in range(3):
                    f, g = R.eval(sp_pos, return_grad=True)
                    gn = g.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    step_v = (f / gn.squeeze(1)).clamp(-0.01, 0.01)
                    sp_pos = sp_pos - step_v[:, None] * g / gn
                f, g = R.eval(sp_pos, return_grad=True)
                sp_nrm = g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)
                R.build(sp_pos, sp_nrm, sp_cur, verbose=False)
            # resample uniformly from a clean reconstructed surface
            if args.resample_mode == "poisson":
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                sub = torch.randperm(sp_pos.shape[0], device=dev)[:args.poisson_points]
                pcd.points = o3d.utility.Vector3dVector(sp_pos[sub].double().cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(sp_nrm[sub].double().cpu().numpy())
                pm, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth, linear_fit=True)
                dens = np.asarray(dens)
                pm.remove_vertices_by_mask(dens < np.quantile(dens, 0.005))
                Vm, Fm = np.asarray(pm.vertices, dtype=np.float64), np.asarray(pm.triangles, dtype=np.int64)
            else:
                Vm, Fm = R.marching_cubes(args.resample_res)
            Vm, Fm = largest_components(Vm, Fm, args.min_comp_frac)
            _a, _b, _c = Vm[Fm[:, 0]], Vm[Fm[:, 1]], Vm[Fm[:, 2]]
            R.cfg["far_winding_area"] = float(0.5 * np.linalg.norm(np.cross(_b - _a, _c - _a), axis=1).sum())
            vn = vertex_normals_smooth(Vm, Fm, iters=1)
            P, N, _, _ = sample_surface(Vm, Fm, args.n_sp, device=dev, seed=args.seed + step, vertex_normals=vn)
            sp_pos, sp_nrm = P, N
            if args.carry_curvature:
                if args.curvature_source == "mesh":
                    h, _ = robust_curvature(Vm, Fm, target_scale=2.0 / args.fine_res)
                    _, _, S, _ = sample_surface(Vm, Fm, args.n_sp, device=dev, seed=args.seed + step, vertex_scalar=h)
                    sp_cur = S.clamp(0, args.curvature_max)
                else:
                    sp_cur = mpu_curvature(R, sp_pos, h_fd).clamp(0, args.curvature_max)
            R.build(sp_pos, sp_nrm, sp_cur, verbose=False)
        if step % 10 == 0 or step == args.steps:
            st = R.mpu.fp_stats()
            print(f"[step {step:4d}] t={step*args.dt:.2f} sign={sign:+.0f} fps={st['n_fp']} per_level={st['per_level']} "
                  f"build {tb:.2f}s elapsed {time.time()-t_start:.0f}s", flush=True)
        if args.save_every and step % args.save_every == 0:
            Vm, Fm = R.marching_cubes(args.eval_res)
            trimesh.Trimesh(Vm, Fm, process=False).export(os.path.join(args.out, f"mesh_{step:04d}.ply"))
            log.append({"step": step, "volume": mesh_volume(Vm, Fm), "n_fp": R.mpu.fp_stats()["n_fp"]})
            print(f"[step {step:4d}] volume {log[-1]['volume']:.6f} ({(log[-1]['volume']-vol0)/vol0*100:+.3f}%)", flush=True)

    # ------------------------------------------------------------------ final metrics
    Vf, Ff = R.marching_cubes(args.eval_res)
    volf = mesh_volume(Vf, Ff)
    if args.test == "rotate":
        # rotate the final surface back by -t about the unit-cube axis (0.5,0.5) -> [-1,1] coords (0,0)
        theta = -args.steps * args.dt
        c, s = math.cos(theta), math.sin(theta)
        Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Vf_back = Vf @ Rm.T
        occ_f = M.winding_occupancy(Vf_back, Ff, M.grid_points(args.eval_res).double().cpu().numpy())
        iou = M.iou(occ0, occ_f)
        # also IoU of the field (unrotated) vs a rotated initial mesh is equivalent; report mesh-based
    else:
        occ_f = occupancy(R, args.eval_res)
        iou = M.iou(occ0, occ_f)
    result = {"test": args.test, "steps": args.steps, "dt": args.dt, "fine_res": args.fine_res, "num_levels": args.num_levels,
              "n_sp": args.n_sp, "resample_every": args.resample_every, "iou": iou, "volume_init": vol0, "volume_final": volf,
              "volume_loss_pct": (vol0 - volf) / vol0 * 100.0, "volume_exact": vol_exact, "log": log,
              "n_fp_final": R.mpu.fp_stats()["n_fp"], "time_total": time.time() - t_start}
    trimesh.Trimesh(Vf, Ff, process=False).export(os.path.join(args.out, "mesh_final.ply"))
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    print("[done]", json.dumps({k: v for k, v in result.items() if k != "log"}), flush=True)
    return result


if __name__ == "__main__":
    main(build_parser().parse_args())
