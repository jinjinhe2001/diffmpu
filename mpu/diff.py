"""Differentiable moving-particle optimisation (paper Algorithm 3): Chamfer-loss SDF optimisation
and inverse rendering. Sample particles are advected by task gradients, normals re-estimated by
weighted PCA, colliding opposite particles deleted, the MPU rebuilt every step, and every
`resample_every` steps the sample set is regenerated from a reconstruction (MC / Poisson / DPSR)."""
import json
import math
import os
import time

import numpy as np
import torch
import trimesh

from .recon import Reconstructor, run_marching_cubes
from .data import sample_surface, vertex_normals_smooth
from . import metrics as M


def mpu_curvature(R, P, h, batch=2 ** 22):
    """|div(grad f / |grad f|)| by central differences."""
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


def init_sphere(n, radius, center=(0.0, 0.0, 0.0), device="cuda", seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    v = torch.randn((n, 3), device=device, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    return torch.tensor(center, device=device) + radius * v, v.clone()


_TREE_CACHE = {}


def knn_idx(query, ref, K=1, chunk=2 ** 18, grid=True, static_ref=False):
    if K == 1 and grid and query.is_cuda and ref.shape[0] >= 4096 and query.shape[0] >= 4096:
        from .core import grid_nn
        return grid_nn(query, ref, cache=static_ref)   # exact; brute force only for queries farther than 6 grid cells from ref
    if K == 1 and query.shape[0] * ref.shape[0] > 3e11:
        # brute force is O(N*M): switch to a (cached) KD-tree for millions x millions
        from scipy.spatial import cKDTree
        key = (ref.data_ptr(), tuple(ref.shape))
        tree = _TREE_CACHE.get(key)
        if tree is None:
            tree = cKDTree(ref.detach().cpu().numpy())
            _TREE_CACHE.clear(); _TREE_CACHE[key] = tree     # single entry: a freed tensor's address can be reused
        d, i = tree.query(query.detach().cpu().numpy(), k=1, workers=-1)
        idx = torch.as_tensor(i, device=query.device).long()[:, None]
        d2 = torch.as_tensor(d.astype(np.float32) ** 2, device=query.device)[:, None]
        return idx, d2
    try:
        from pytorch3d.ops import knn_points
    except ImportError:                         # pytorch3d is optional: exact torch fallback (slower, same result)
        return _knn_torch(query, ref, K)
    idx = torch.empty((query.shape[0], K), dtype=torch.long, device=query.device)
    d2 = torch.empty((query.shape[0], K), dtype=torch.float32, device=query.device)
    for s in range(0, query.shape[0], chunk):
        r = knn_points(query[s:s + chunk][None], ref[None], K=K, return_sorted=True)
        idx[s:s + chunk] = r.idx[0]
        d2[s:s + chunk] = r.dists[0]
    return idx, d2


def _knn_torch(query, ref, K, max_elems=2 ** 27):
    """K nearest neighbours without pytorch3d: chunked torch.cdist + topk, squared distances recomputed exactly for the
    selected neighbours (cdist uses the matmul expansion). Sorted ascending like knn_points; 300k x 300k, K=9 in ~2 s."""
    N, M = query.shape[0], ref.shape[0]
    chunk = max(64, min(N, max_elems // max(M, 1)))
    idx = torch.empty((N, K), dtype=torch.long, device=query.device)
    d2 = torch.empty((N, K), dtype=torch.float32, device=query.device)
    q = query.float(); r = ref.float()
    for s in range(0, N, chunk):
        d = torch.cdist(q[s:s + chunk], r)
        _, i = d.topk(min(K, M), dim=1, largest=False)
        ex = ((q[s:s + chunk][:, None, :] - r[i]) ** 2).sum(-1)
        ex, order = ex.sort(dim=1)
        idx[s:s + chunk] = i.gather(1, order)
        d2[s:s + chunk] = ex
    return idx, d2


def orient_thin(P, N, cell, max_cells=8, min_cells=3, r_cells=1.2):
    """Thin-structure normal orientation: probe along +n and -n at 3..max_cells finest cells; if another sheet of
    particles is found on the +n side only, the solid lies on +n and the outward normal must be -n -> flip.
    Solids (no interior particles) are untouched except in their thin parts."""
    r2 = (r_cells * cell) ** 2
    hit_p = torch.zeros(P.shape[0], dtype=torch.bool, device=P.device)
    hit_m = torch.zeros_like(hit_p)
    for d in range(int(min_cells), int(max_cells) + 1):
        _, dp = knn_idx(P + d * cell * N, P, 1)
        _, dm = knn_idx(P - d * cell * N, P, 1)
        hit_p |= dp[:, 0] < r2
        hit_m |= dm[:, 0] < r2
    flip = hit_p & ~hit_m
    return torch.where(flip[:, None], -N, N), flip


def orient_neighbors(P, N, K=16, iters=3):
    """Flip normals that disagree with the (unweighted) mean normal of their K nearest neighbours; a few sweeps
    propagate a locally consistent orientation across a sheet (does not fix a globally inverted sheet)."""
    idx, _ = knn_idx(P, P, K + 1)
    idx = idx[:, 1:]
    total = torch.zeros(P.shape[0], dtype=torch.bool, device=P.device)
    for _ in range(iters):
        m = N[idx].mean(1)
        flip = (N * m).sum(1) < 0
        N = torch.where(flip[:, None], -N, N)
        total |= flip
    return N, total


def orient_mst(P, N, K=8, signed=False):
    """Globally consistent normal orientation by propagation along a minimum spanning tree of the K-NN graph
    (Hoppe et al. 1992). The global sign is chosen so that the majority of the input normals is kept."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
    n = P.shape[0]
    idx, _ = knn_idx(P, P, K + 1)
    idx = idx[:, 1:].cpu().numpy()
    Nn = N.cpu().numpy().astype(np.float64)
    rows = np.repeat(np.arange(n), K); cols = idx.reshape(-1)
    # signed affinity: anti-parallel neighbours get weight ~2, so the tree stays inside consistently oriented
    # regions and does not jump between the two sides of a thin sheet (which would flip a correct sheet)
    dots = (Nn[rows] * Nn[cols]).sum(1)
    w = (1.0 - dots if signed else 1.0 - np.abs(dots)) + 1e-6   # unsigned (Hoppe) is the validated default
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    T = minimum_spanning_tree(G).tocsr()
    T = T.maximum(T.T)
    order, pred = breadth_first_order(T, 0, directed=False, return_predecessors=True)
    out = Nn.copy()
    for v in order[1:]:
        u = pred[v]
        if u >= 0 and (out[v] * out[u]).sum() < 0:
            out[v] = -out[v]
    # disconnected components (other trees) are left as they are; global sign: keep the majority orientation
    if ((out * Nn).sum(1) < 0).mean() > 0.5:
        out = -out
    outN = torch.as_tensor(out, dtype=N.dtype, device=N.device)
    flipped = (outN * N).sum(1) < 0
    return outN, flipped


def cubic_interp_grid(vol, P, g, a_keys=-0.5):
    """Tricubic convolution (Keys kernel) of a (g,g,g) grid on [-1,1]^3 (node i at -1 + 2i/(g-1)) at P (n,3).
    Returns values (n,) and analytic gradients (n,3) w.r.t. world coordinates."""
    h = 2.0 / (g - 1)
    u = (P + 1.0) / h                                   # continuous index
    i0 = torch.floor(u).long()
    t = u - i0.float()                                 # (n,3) in [0,1)
    def w_and_dw(t):                                   # Keys cubic convolution weights for offsets -1,0,1,2
        a = a_keys
        t2 = t * t; t3 = t2 * t
        w0 = a * (-t3 + 2 * t2 - t)
        w1 = (a + 2) * t3 - (a + 3) * t2 + 1
        w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
        w3 = -a * t3 + a * t2
        d0 = a * (-3 * t2 + 4 * t - 1)
        d1 = 3 * (a + 2) * t2 - 2 * (a + 3) * t
        d2 = -3 * (a + 2) * t2 + 2 * (2 * a + 3) * t - a
        d3 = -3 * a * t2 + 2 * a * t
        return torch.stack([w0, w1, w2, w3], -1), torch.stack([d0, d1, d2, d3], -1)
    W, dW = w_and_dw(t)                                # (n,3,4)
    val = torch.zeros(P.shape[0], device=P.device)
    grad = torch.zeros(P.shape[0], 3, device=P.device)
    for dx in range(4):
        ix = (i0[:, 0] + dx - 1).clamp(0, g - 1)
        for dy in range(4):
            iy = (i0[:, 1] + dy - 1).clamp(0, g - 1)
            for dz in range(4):
                iz = (i0[:, 2] + dz - 1).clamp(0, g - 1)
                v = vol[ix, iy, iz]
                wx, wy, wz = W[:, 0, dx], W[:, 1, dy], W[:, 2, dz]
                val += v * wx * wy * wz
                grad[:, 0] += v * dW[:, 0, dx] * wy * wz
                grad[:, 1] += v * wx * dW[:, 1, dy] * wz
                grad[:, 2] += v * wx * wy * dW[:, 2, dz]
    return val, grad / h


def largesteps_solve(g, faces, nV, lam, iters=50):
    """Solve (I + lam * L) u = g with the uniform graph Laplacian L = D - A by conjugate gradients
    (Nicolet et al. 2021 'Large Steps' preconditioning)."""
    from .render_utils import compute_edges
    E = compute_edges(faces)
    e0, e1 = E[:, 0], E[:, 1]
    ones = torch.ones(E.shape[0], device=g.device)
    deg = torch.zeros(nV, device=g.device).index_add_(0, e0, ones).index_add_(0, e1, ones)

    def A(x):
        s = torch.zeros_like(x).index_add_(0, e0, x[e1]).index_add_(0, e1, x[e0])
        return x + lam * (deg[:, None] * x - s)

    x = torch.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rs = (r * r).sum()
    for _ in range(iters):
        Ap = A(p)
        alpha = rs / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new < 1e-24:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def largest_components(V, F, min_frac=0.001):
    m = trimesh.Trimesh(V, F, process=True)   # merge duplicate vertices, drop degenerate faces
    m.remove_unreferenced_vertices()
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        nf = np.array([len(c.faces) for c in comps])
        keep = [c for c, n in zip(comps, nf) if n >= min_frac * nf.sum()]
        m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


class DiffMPU:
    def __init__(self, args, cfg, gt_V, gt_F, device="cuda"):
        self.args = args
        self.cfg = cfg
        self.dev = device
        self.gt_V, self.gt_F = gt_V, gt_F
        self.R = Reconstructor(cfg)
        self.L = cfg["num_levels"]
        self.cell_fine = 2.0 / (cfg["base_res"] * 2 ** (self.L - 1))
        self.h_fd = 0.5 * self.cell_fine
        self.gt_pts, self.gt_nrm, _, _ = sample_surface(gt_V, gt_F, args.n_gt, device=device, seed=1)
        n0 = args.n_sp_start if getattr(args, "n_sp_start", 0) > 0 else args.n_sp
        self.sp_pos, self.sp_nrm = init_sphere(n0, args.init_radius, device=device, seed=args.seed)
        self.sp_cur = torch.zeros(n0, device=device)
        self.sp_vel = torch.zeros((n0, 3), device=device)
        self.cfg["far_winding_area"] = 4 * math.pi * args.init_radius ** 2
        self.log = []
        self.renderer = None
        self.t0 = time.time()
        if args.task == "sdfgrid":
            import igl
            g = args.sdf_grid_res
            bm = float(getattr(args, "sdf_bbox", 0.0))
            if bm > 0:
                lo, hi = gt_V.min(0), gt_V.max(0)
                cb = 0.5 * (lo + hi); hb = 0.5 * (hi - lo) * bm
                self.sdf_center = torch.as_tensor(cb, dtype=torch.float32, device=device)
                self.sdf_half = torch.as_tensor(hb, dtype=torch.float32, device=device)
                axes = [torch.linspace(float(cb[i] - hb[i]), float(cb[i] + hb[i]), g, device=device) for i in range(3)]
                print(f"[sdfgrid] tight bounding-box grid x{bm}: center {cb.round(3).tolist()} half {hb.round(3).tolist()}", flush=True)
            else:
                self.sdf_center = torch.zeros(3, device=device); self.sdf_half = torch.ones(3, device=device)
                axes = [torch.linspace(-1.0, 1.0, g, device=device)] * 3
            X, Y, Z = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
            Q = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
            Qn = Q.double().cpu().numpy()
            try:
                res_sd = igl.signed_distance(np.ascontiguousarray(Qn), np.ascontiguousarray(gt_V), np.ascontiguousarray(gt_F.astype(np.int64)),
                                             igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER if hasattr(igl, "SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER") else 4)
                sd = np.asarray(res_sd[0]).ravel()
            except Exception as ex:
                print("[sdfgrid] igl.signed_distance failed, using unsigned distance + winding sign:", repr(ex)[:120], flush=True)
                from .metrics import point_mesh_sqdist, winding_occupancy
                sq, _, _ = point_mesh_sqdist(Qn, gt_V, gt_F)
                inside = winding_occupancy(gt_V, gt_F, Qn)
                sd = np.sqrt(sq) * np.where(inside, -1.0, 1.0)
            keep = np.abs(sd) < args.sdf_keep_band if args.sdf_keep_band > 0 else np.ones_like(sd, dtype=bool)
            self.grid_pts = Q[torch.as_tensor(keep, device=device)].float().contiguous()
            self.grid_sdf = torch.as_tensor(sd[keep], dtype=torch.float32, device=device)
            # full coarse SDF volume (+ finite-difference gradient volumes) for the interpolated-SDF flow
            self.sdf_res = g
            self.sdf_vol = torch.as_tensor(sd.reshape(g, g, g), dtype=torch.float32, device=device)
            sp = [float(2.0 * self.sdf_half[i] / (g - 1)) for i in range(3)]
            gx, gy, gz = torch.gradient(self.sdf_vol, spacing=sp)
            self.sdf_grad_vol = torch.stack([gx, gy, gz], 0)          # (3,g,g,g)
            print(f"[sdfgrid] {g}^3 grid, {int(keep.sum())} samples kept, |sdf| median {np.median(np.abs(sd[keep])):.3f}", flush=True)
        if args.task == "render":
            from .render_utils import Renderer
            self.renderer = Renderer(args.views, args.res, seed=42, device=device)
            self.target = self.renderer.render_mesh_np(gt_V, gt_F)
            self.target_mask = self.renderer.render_mask_np(gt_V, gt_F)
            from .render_utils import build_pyramid
            self.target_pyr = build_pyramid(self.target, self.renderer.kernel)          # constant -> computed once
            self.target_mask_pyr = build_pyramid(self.target_mask, self.renderer.kernel)
            self.renderer_test = Renderer(args.test_views, args.res, seed=4242, device=device)
            self.target_test = self.renderer_test.render_mesh_np(gt_V, gt_F)
            if getattr(args, "hull_w", 0.0) > 0 or getattr(args, "hull_delete_px", 0.0) > 0 or getattr(args, "hull_carve", 0):
                self._init_hull(gt_V)
        self.build()

    # ------------------------------------------------------------------
    def build(self):
        a = self.args
        # the chamfer / sdfgrid loops never read the far field between builds (velocities use the GT points / SDF grid,
        # normals use the gradient only); it is computed lazily by mesh() / evaluate(). The render loop meshes every iteration.
        need_far = (a.task == "render" or a.resample_mode == "mc" or getattr(a, "resample_mode2", "same") == "mc"
                    or a.curv_source == "mpu" or getattr(a, "hull_carve", 0) or getattr(a, "lazy_far", 1) == 0)
        self.R.build(self.sp_pos, self.sp_nrm, self.sp_cur, verbose=False, far=bool(need_far))
        if self.args.carry_curvature and self.args.curv_source == "mpu":
            self.sp_cur = mpu_curvature(self.R, self.sp_pos, self.h_fd).clamp(0, self.args.curvature_max)
        elif self.args.carry_curvature and self.args.curv_source == "pca":
            self.sp_cur = self.R.mpu.sp_curvature_pca(K=self.args.pca_K, device=self.dev).clamp(0, self.args.curvature_max)

    def mesh(self, res):
        if getattr(self.args, "hull_carve", 0):
            field = self.carve(self.R.field(res), res)
            return run_marching_cubes(field, res, device=self.dev)
        V, F = self.R.marching_cubes(res)
        return V, F

    # ------------------------------------------------------------------ velocities
    def velocity_chamfer(self):
        P = self.sp_pos
        G = self.gt_pts
        i1, d1 = knn_idx(P, G, 1, static_ref=True)
        g1 = 2.0 * (P - G[i1[:, 0]])
        i2, d2 = knn_idx(G, P, 1)
        g2 = torch.zeros_like(P).index_add_(0, i2[:, 0], 2.0 * (P[i2[:, 0]] - G))
        # per-particle normalisation of the pull term so that a particle nearest to many GT points is not exploded
        cnt = torch.zeros(P.shape[0], device=P.device).index_add_(0, i2[:, 0], torch.ones(G.shape[0], device=P.device))
        g2 = g2 / cnt.clamp_min(1.0)[:, None] * float(self.args.pull_weight)
        loss = float(d1.mean() + d2.mean())
        return -(g1 + g2), loss

    def sdf_interp(self, P):
        """Interpolation of the coarse SDF grid and of its gradient at P (n,3) -> s (n,), grad (n,3).
        --sdf_interp cubic uses tricubic (Keys / Catmull-Rom) convolution, else trilinear."""
        if getattr(self.args, "sdf_interp", "linear") == "cubic":
            return cubic_interp_grid(self.sdf_vol, P, self.sdf_res)
        Fn = torch.nn.functional
        Pn = (P - self.sdf_center) / self.sdf_half                                      # bbox grid -> [-1,1]^3
        coords = torch.stack([Pn[:, 2], Pn[:, 1], Pn[:, 0]], -1)[None, None, None]    # grid_sample wants (z,y,x)
        s = Fn.grid_sample(self.sdf_vol[None, None], coords, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0, 0]
        gvol = self.sdf_grad_vol[None]                                                  # (1,3,g,g,g)
        gr = Fn.grid_sample(gvol, coords, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0, :].t()
        return s, gr

    def velocity_sdfgrid(self):
        a = self.args
        mode = a.sdf_mode
        if getattr(a, "sdf_mode2", "same") != "same" and getattr(self, "it", 0) >= a.lr_switch:
            mode = a.sdf_mode2
        if mode == "interp":
            # Newton projection onto the zero level set of the trilinearly interpolated coarse SDF
            P = self.sp_pos
            s, gr = self.sdf_interp(P)
            gn2 = (gr * gr).sum(1).clamp_min(1e-8)
            V = -(s / gn2)[:, None] * gr
            return V, float((s ** 2).mean())
        return self._velocity_sdfgrid_samples()

    def _velocity_sdfgrid_samples(self):
        """Tangency-aware flow (Reach-for-the-Spheres energy).
        mode 'rfs': (1) every sparse sample q with |s| pulls its *closest* surface particle onto the
        sphere of radius |s| around q (tangency); (2) particles found inside any nearby sample's ball
        are pushed out to the ball surface (emptiness).  mode 'normal': legacy per-particle normal
        residual with the K nearest samples."""
        P, N = self.sp_pos, self.sp_nrm
        a = self.args
        if a.sdf_mode == "rfs":
            Q, S = self.grid_pts, self.grid_sdf
            R = S.abs()
            # (1) tangency: nearest particle of each sample
            ip, d2 = knn_idx(Q, P, K=1)
            i = ip[:, 0]
            d = d2[:, 0].clamp_min(1e-18).sqrt()
            u = (P[i] - Q) / d[:, None]
            move = (R - d)[:, None] * u                      # bring the closest particle onto the sphere
            V1 = torch.zeros_like(P).index_add_(0, i, move)
            cnt = torch.zeros(P.shape[0], device=P.device).index_add_(0, i, torch.ones(Q.shape[0], device=P.device))
            V1 = V1 / cnt.clamp_min(1.0)[:, None]
            # (2) emptiness: push particles out of the balls of their K nearest samples
            iq, dq2 = knn_idx(P, Q, K=a.sdf_K)
            dq = dq2.clamp_min(1e-18).sqrt()                 # (n,K)
            rq = R[iq]
            pen = (rq - dq).clamp_min(0.0)                   # penetration depth
            dirv = (P[:, None, :] - Q[iq]) / dq[:, :, None]
            V2 = (pen[:, :, None] * dirv).sum(1)
            V = V1 + a.sdf_empty_w * V2
            loss = float(((R - d) ** 2).mean())
            return V, loss
        idx, _ = knn_idx(P, self.grid_pts, K=self.args.sdf_K)
        q = self.grid_pts[idx]                       # (n,K,3)
        s = self.grid_sdf[idx]                       # (n,K)
        if a.sdf_mode == "sphere":
            # exact sphere tangency: the surface must touch the sphere of radius |s_k| around each sample q_k.
            # residual = |p - q_k| - |s_k|, moved radially; the sign of s selects inside/outside samples alike.
            d = P[:, None, :] - q                    # (n,K,3)
            dn = d.norm(dim=-1).clamp_min(1e-9)      # (n,K)
            r = dn - s.abs()                         # >0: particle outside the sphere -> move toward q
            u = d / dn[..., None]
            w = 1.0 / (s.abs() + a.sdf_sphere_eps)   # closer (smaller-radius) spheres are more reliable
            w = w / w.sum(1, keepdim=True)
            V = -((w * r)[..., None] * u).sum(1)
            if a.sdf_sphere_normal:
                V = (V * N).sum(1, keepdim=True) * N  # keep only the normal component (tangency-aware)
            return V, float((r ** 2).mean())
        r = s - ((q - P[:, None, :]) * N[:, None, :]).sum(-1)   # (n,K)
        rm = r.mean(1)
        V = -2.0 * rm[:, None] * N
        return V, float((r ** 2).mean())

    def velocity_render(self, it):
        from .render_utils import vertex_normals_t, compute_edges, laplacian_uniform, img_loss
        a = self.args
        jr = int(getattr(a, "mc_jitter_range", 3))
        res = a.mc_res + int(np.random.randint(-jr, jr + 1)) if a.mc_jitter else a.mc_res
        tp = time.time()
        V, F = self.mesh(res)
        tp = self._tick("vel/marching_cubes", tp)
        if V.shape[0] == 0:
            return torch.zeros_like(self.sp_pos), float("nan"), None
        Vt = torch.as_tensor(V, dtype=torch.float32, device=self.dev).requires_grad_(True)
        Ft = torch.as_tensor(F, dtype=torch.long, device=self.dev)
        use_field_normals = a.normals_from_field and not (a.mesh_normals_from > 0 and it >= a.mesh_normals_from)
        if use_field_normals:
            with torch.no_grad():
                _, gfield = self.R.eval(Vt.detach(), return_grad=True)
                vn = gfield / gfield.norm(dim=1, keepdim=True).clamp_min(1e-8)
                bad = ~torch.isfinite(vn).all(dim=1) | (gfield.norm(dim=1) < 1e-6)
                vn_mesh = vertex_normals_t(Vt.detach(), Ft)
                vn = torch.where(bad[:, None], vn_mesh, vn)
        else:
            vn = vertex_normals_t(Vt, Ft)
        tp = self._tick("vel/normals", tp)
        photo_w = a.photo_w if it >= a.photo_start else 0.0
        lam = a.laplace_lam if it < a.lr_switch else a.laplace_lam2
        kernel = self.renderer.kernel
        n_views = self.renderer.mvps.shape[0]
        chunk = int(getattr(a, "view_chunk", 0) or 0)
        if chunk <= 0 or chunk >= n_views:
            if a.mask_w > 0:
                imgs, masks = self.renderer.render_with_mask(Vt, Ft, vn)       # one rasterisation for shading and silhouette
            else:
                imgs, masks = self.renderer.render(Vt, Ft, vn), None
            tp = self._tick("vel/render", tp)
            loss = photo_w * img_loss(imgs, self.target, kernel, multi_scale=True, target_pyr=getattr(self, "target_pyr", None))
            if a.mask_w > 0:
                loss = loss + a.mask_w * img_loss(masks, self.target_mask, kernel, multi_scale=True, target_pyr=getattr(self, "target_mask_pyr", None))
            if lam > 0:
                E = compute_edges(Ft)
                Lm = laplacian_uniform(Vt.shape[0], E)
                lap = torch.sparse.mm(Lm, Vt)
                loss = loss + lam * (lap * Vt).sum()
            tp = self._tick("vel/loss", tp)
            loss.backward()
            loss_val = float(loss.item())
            mse = (imgs.detach() - self.target).square().mean()
        else:
            # views in chunks, gradients accumulated in Vt.grad: identical to the one-batch gradient (the loss is a mean over
            # views), peak memory divided by n_views / chunk. Mesh normals with gradients are recomputed per chunk so that no
            # chunk graph has to be retained.
            loss_val = 0.0
            se_sum = torch.zeros((), device=self.dev)
            for s in range(0, n_views, chunk):
                ids = torch.arange(s, min(n_views, s + chunk), device=self.dev)
                w = float(ids.numel()) / n_views
                vn_c = vn if use_field_normals else vertex_normals_t(Vt, Ft)
                if a.mask_w > 0:
                    imgs_c, masks_c = self.renderer.render_with_mask(Vt, Ft, vn_c, view_ids=ids)
                else:
                    imgs_c, masks_c = self.renderer.render(Vt, Ft, vn_c, view_ids=ids), None
                lc = photo_w * img_loss(imgs_c, self.target[ids], kernel, multi_scale=True,
                                        target_pyr=[p[ids] for p in self.target_pyr] if hasattr(self, "target_pyr") else None)
                if a.mask_w > 0:
                    lc = lc + a.mask_w * img_loss(masks_c, self.target_mask[ids], kernel, multi_scale=True,
                                                  target_pyr=[p[ids] for p in self.target_mask_pyr] if hasattr(self, "target_mask_pyr") else None)
                (w * lc).backward()
                loss_val += w * float(lc.detach())
                se_sum += (imgs_c.detach() - self.target[ids]).square().sum()
                del imgs_c, masks_c, lc
            if lam > 0:
                E = compute_edges(Ft)
                Lm = laplacian_uniform(Vt.shape[0], E)
                lap = torch.sparse.mm(Lm, Vt)
                ll = lam * (lap * Vt).sum()
                ll.backward()
                loss_val += float(ll.detach())
            mse = se_sum / float(self.target.numel())
            tp = self._tick("vel/render+loss+backward", tp)
        gV = torch.nan_to_num(Vt.grad.detach())
        gV = gV.clamp(-1e3, 1e3)
        tp = self._tick("vel/backward", tp)
        lam_ls = a.ls_lambda if it < a.lr_switch else a.ls_lambda2
        if lam_ls > 0:
            gV = largesteps_solve(gV, Ft, Vt.shape[0], lam_ls, iters=a.ls_iters)
        tp = self._tick("vel/largesteps", tp)
        # transfer vertex velocities to sample particles: mean over K nearest vertices
        idx, _ = knn_idx(self.sp_pos, Vt.detach(), K=a.vel_K)
        Vsp = -gV[idx].mean(1)
        tp = self._tick("vel/knn_transfer", tp)
        with torch.no_grad():
            ps = float(-10 * torch.log10(mse.clamp_min(1e-12)))
        return Vsp, loss_val, ps

    # ------------------------------------------------------------------ visual-hull (dense silhouette) term
    def _init_hull(self, gt_V):
        """Euclidean distance transform (pixels) of every target mask: 0 on the foreground, distance to the
        nearest foreground pixel on the background.  Used as a dense silhouette energy for the particles."""
        from scipy.ndimage import distance_transform_edt
        Mk = (self.target_mask[..., 0] > 0.5).cpu().numpy()         # (V,H,W)
        D = np.stack([distance_transform_edt(~m) for m in Mk]).astype(np.float32)
        Din = np.stack([distance_transform_edt(m) for m in Mk]).astype(np.float32)
        self.hull_D = torch.as_tensor(D, device=self.dev)[:, None]  # (V,1,H,W) pixels to the silhouette, 0 inside
        self.hull_Din = torch.as_tensor(Din, device=self.dev)[:, None]  # pixels to the background, 0 outside
        # verify the projection convention with the GT vertices (they must project onto the foreground)
        Vt = torch.as_tensor(gt_V[np.random.RandomState(0).choice(len(gt_V), min(20000, len(gt_V)), replace=False)],
                             dtype=torch.float32, device=self.dev)
        best = None
        for flip in (1.0, -1.0):
            self.hull_flip_y = flip
            d, _ = self._hull_sample(Vt)
            frac = float((d < 1.0).float().mean())
            print(f"[hull] flip_y {flip:+.0f}: GT vertices inside the hull {frac:.4f}", flush=True)
            if best is None or frac > best[0]:
                best = (frac, flip)
        self.hull_flip_y = best[1]

    def _hull_sample(self, P, batch=25, signed=False):
        """Pixel distance to the target silhouette of every particle in every view: (V,n) tensor; also
        returns the world size of one pixel at the particle depth (V,n).  Differentiable w.r.t. P."""
        F = torch.nn.functional
        v_hom = F.pad(P, (0, 1), value=1.0)                            # (n,4)
        Ds, Ps = [], []
        nviews = self.renderer.mvps.shape[0]
        for s in range(0, nviews, batch):
            mvps = self.renderer.mvps[s:s + batch]                     # (b,4,4)
            clip = torch.einsum("bij,nj->bni", mvps, v_hom)            # (b,n,4)
            w = clip[..., 3].clamp_min(1e-6)
            ndc = clip[..., :2] / w[..., None]
            ndc = torch.stack([ndc[..., 0], self.hull_flip_y * ndc[..., 1]], -1)
            grid = ndc[:, None, :, :]                                  # (b,1,n,2)
            d = F.grid_sample(self.hull_D[s:s + batch], grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, 0, :]
            pix = 2.0 * w / (3.0 * self.renderer.res)                    # world units per pixel (proj n/x = 3)
            if signed:
                din = F.grid_sample(self.hull_Din[s:s + batch], grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, 0, :]
                d = (d - din) * pix                                        # world-unit signed distance to the silhouette (+ outside)
            Ds.append(d); Ps.append(pix)
        return torch.cat(Ds), torch.cat(Ps)

    # ------------------------------------------------------------------ visual-hull carving of the implicit field
    def hull_signed_at(self, Q, batch_pts=400_000):
        """max over views of the signed silhouette distance (world units, + outside the visual hull)."""
        out = torch.empty(Q.shape[0], device=self.dev)
        with torch.no_grad():
            for s in range(0, Q.shape[0], batch_pts):
                d, _ = self._hull_sample(Q[s:s + batch_pts], signed=True)
                out[s:s + batch_pts] = d.max(0).values
        return out

    def hull_grid(self, res):
        if not hasattr(self, "_hull_grid"):
            self._hull_grid = {}
        if res not in self._hull_grid:
            Q = M.grid_points(res, device=self.dev)
            self._hull_grid[res] = self.hull_signed_at(Q).reshape(res, res, res)
            print(f"[hull] carve grid {res}^3 built: outside fraction {float((self._hull_grid[res] > 0).float().mean()):.3f}", flush=True)
        return self._hull_grid[res]

    def carve(self, field, res):
        """Intersect the MPU field with the visual hull (space carving): max(field, hull_sdf)."""
        if not getattr(self.args, "hull_carve", 0):
            return field
        return torch.maximum(field, self.hull_grid(res).to(field.dtype).reshape(field.shape))

    def hull_displacement(self, P):
        """World-space displacement that moves every particle lying outside the visual hull toward the
        silhouette boundary, averaged over the views in which it is outside.  Zero inside the hull."""
        P = P.detach().requires_grad_(True)
        d, pix = self._hull_sample(P)                                  # (V,n)
        d = (d - self.args.hull_margin).clamp_min(0.0)                 # dead zone: rim particles get no push
        dw = d * pix.detach()
        E = 0.5 * (dw ** 2).sum()
        g, = torch.autograd.grad(E, P)
        cnt = (d.detach() > 0.0).float().sum(0)
        disp = -g / cnt.clamp_min(1.0)[:, None]
        return torch.nan_to_num(disp), cnt

    # ------------------------------------------------------------------ one optimisation step
    def _tick(self, key, t0):
        """--profile: accumulate the (synchronised) wall time since t0 under key; returns the current time."""
        if not getattr(self, "profile", False):
            return t0
        torch.cuda.synchronize()
        t = time.time()
        if not hasattr(self, "prof"):
            self.prof = {}
        self.prof[key] = self.prof.get(key, 0.0) + (t - t0)
        return t

    def step(self, it):
        a = self.args
        m = self.R.mpu
        n = self.sp_pos.shape[0]
        t0 = time.time()
        if a.task == "chamfer":
            Vel, loss = self.velocity_chamfer()
            psnr = None
        elif a.task == "sdfgrid":
            Vel, loss = self.velocity_sdfgrid()
            psnr = None
        else:
            Vel, loss, psnr = self.velocity_render(it)
        t_vel = time.time() - t0
        tp = self._tick("velocity", t0)
        # smooth velocities over neighbours (partition-of-unity like), using the current binning
        Vel = Vel.contiguous().float()
        dummy_n = torch.zeros((n, 3), device=self.dev)
        out_v = torch.zeros((n, 3), device=self.dev)
        flag = torch.zeros(n, dtype=torch.int32, device=self.dev)
        lvl = max(self.L - 1 - a.smooth_level_offset, 0)
        rad = a.smooth_radius_cells * (2.0 / (self.cfg["base_res"] * 2 ** lvl))
        off = a.smooth_level_offset if it < a.lr_switch else a.smooth_level_offset2
        if off >= 0:
            lvl = max(self.L - 1 - off, 0)
            rad = a.smooth_radius_cells * (2.0 / (self.cfg["base_res"] * 2 ** lvl))
        if a.smooth_K > 0 and off >= 0:
            dcur = torch.zeros(1, device=self.dev)
            m.sp_neighborhood(lvl, int(a.smooth_K), 0.0, -2.0, Vel, dummy_n, out_v, flag, 0, int(a.smooth_mode), float(rad), dcur, 0)
            Vel = out_v
        # robust scaling: clip outlier magnitudes, then either a fixed lr or a normalised speed
        mag = Vel.norm(dim=1)
        nz = mag > 0
        if nz.any() and a.clip_pct < 100:
            clip = torch.quantile(mag[nz][:1_000_000], a.clip_pct / 100.0)
            Vel = Vel * torch.clamp(clip / mag.clamp_min(1e-30), max=1.0)[:, None]
            mag = Vel.norm(dim=1)
        if a.speed_cells > 0 and nz.any():
            ref = torch.quantile(mag[nz][:1_000_000], a.speed_pct / 100.0).clamp_min(1e-30)
            target = (a.speed_cells if it < a.lr_switch else a.speed_cells2) * self.cell_fine
            dp = Vel * (target / ref)
        else:
            lr = a.lr if it < a.lr_switch else a.lr2
            dp = lr * Vel
        nrm = dp.norm(dim=1, keepdim=True)
        max_step = (a.max_step_cells if it < a.lr_switch else a.max_step_cells2) * self.cell_fine
        dp = torch.where(nrm > max_step, dp * (max_step / nrm.clamp_min(1e-12)), dp)
        if a.task == "render" and getattr(a, "hull_w", 0.0) > 0 and it >= a.hull_start:
            hd, hcnt = self.hull_displacement(self.sp_pos)
            hn = hd.norm(dim=1, keepdim=True)
            hmax = a.hull_max_cells * self.cell_fine
            hd = torch.where(hn > hmax, hd * (hmax / hn.clamp_min(1e-12)), hd)
            dp = dp + a.hull_w * hd
            if it % 100 == 0:
                print(f"[hull] it {it}: {int((hcnt > 0).sum())} particles outside the hull, mean |disp| {float(hn[hcnt > 0].mean()) if (hcnt > 0).any() else 0.0:.2e}", flush=True)
        if a.kick_every > 0 and it % a.kick_every == 0 and it < a.lr_switch:
            # Shape-as-Points style restart: random directional perturbation of every sample particle
            rnd = torch.randn_like(self.sp_pos)
            rnd = rnd / rnd.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dp = dp + a.kick_cells * self.cell_fine * rnd
            print(f"[kick] it {it}: random perturbation of {a.kick_cells} cells", flush=True)
        self.it = it
        self.sp_pos = (self.sp_pos + dp).clamp(-0.999, 0.999)
        self.sp_vel = dp
        if a.densify_every > 0 and it % a.densify_every == 0 and it < a.lr_switch:
            mag = dp.norm(dim=1)
            k = int(a.densify_frac * mag.shape[0])
            cap = int(a.n_sp * a.densify_cap) - mag.shape[0]
            k = min(k, max(cap, 0))
            if k > 0:
                idx = torch.topk(mag, k).indices
                dirv = dp[idx] / mag[idx, None].clamp_min(1e-12)
                newp = (self.sp_pos[idx] + a.densify_step * self.cell_fine * dirv).clamp(-0.999, 0.999)
                if a.densify_remove:
                    # move the fastest particles ahead instead of cloning them
                    self.sp_pos[idx] = newp
                else:
                    self.sp_pos = torch.cat([self.sp_pos, newp])
                    self.sp_nrm = torch.cat([self.sp_nrm, self.sp_nrm[idx]])
                    self.sp_cur = torch.cat([self.sp_cur, self.sp_cur[idx]])
                    self.sp_vel = torch.cat([self.sp_vel, dp[idx]])
                    dp = torch.cat([dp, dp[idx]])
                    Vel = torch.cat([Vel, Vel[idx]])
                n = self.sp_pos.shape[0]
        if a.normal_mode == "field":
            # orientation-consistent normals from the previous field's gradient at the new positions
            f_old, g_old, cov = self.R.mpu.eval(self.sp_pos, return_grad=True, device=self.dev)
            gn = g_old.norm(dim=1, keepdim=True)
            ok = cov & (gn.squeeze(1) > 1e-6)
            n_field = g_old / gn.clamp_min(1e-12)
            self.sp_nrm = torch.where(ok[:, None], n_field, self.sp_nrm)
        # re-bin at new positions; PCA normals + collisions
        m.bin_only(self.sp_pos, self.sp_nrm, self.sp_cur, self.cfg["thresholds"])
        out_n = torch.zeros((n, 3), device=self.dev)
        flag = torch.zeros(n, dtype=torch.int32, device=self.dev)
        cur_out = torch.zeros(n, device=self.dev)
        m.sp_neighborhood(self.L - 1, int(a.pca_K), float(a.collide_cells * self.cell_fine), float(a.collide_dot),
                          Vel, out_n, out_v, flag, int(a.pca_normals), 0, 1.0, cur_out, 1 if a.curv_source == "pca" else 0)
        if a.pca_normals and a.normal_mode == "pca":
            self.sp_nrm = out_n
        elif a.pca_normals and a.normal_mode == "field":
            # PCA direction, orientation from the field normal
            flip = (out_n * self.sp_nrm).sum(1, keepdim=True) < 0
            self.sp_nrm = torch.where(flip, -out_n, out_n)
        if a.curv_source == "pca":
            self.sp_cur = torch.nan_to_num(cur_out).clamp(0, a.curvature_max)
        if getattr(a, "orient_thin", 0) > 0:
            self.sp_nrm, fl = orient_thin(self.sp_pos, self.sp_nrm, self.cell_fine, max_cells=a.orient_thin)
            if it % 100 == 0:
                print(f"[orient] it {it}: flipped {int(fl.sum())} thin-sheet normals", flush=True)
        keep = flag == 0
        if a.task == "render" and getattr(a, "hull_delete_px", 0.0) > 0 and it >= a.hull_start:
            # space carving: delete particles that project onto the background (beyond margin + delete_px)
            # in at least one view -> the membranes closing lattice holes vanish instead of being smoothed over
            with torch.no_grad():
                d, _ = self._hull_sample(self.sp_pos)
                outside = ((d - a.hull_margin) > a.hull_delete_px).any(0)
            keep = keep & ~outside
            if it % 100 == 0:
                print(f"[hull] it {it}: carving {int(outside.sum())} particles outside the visual hull", flush=True)
        if a.task == "chamfer" and a.far_delete_cells > 0 and it >= a.lr_switch:
            # paper 7.4: remove sample particles with large gradients = far from every target point
            _, dfar = knn_idx(self.sp_pos, self.gt_pts, 1)
            keep = keep & (dfar[:, 0].sqrt() < a.far_delete_cells * self.cell_fine)
        n_del = int((~keep).sum().item())
        if n_del > 0 and keep.sum() > 1000:
            self.sp_pos, self.sp_nrm, self.sp_cur, self.sp_vel = self.sp_pos[keep], self.sp_nrm[keep], self.sp_cur[keep], self.sp_vel[keep]
        tp = self._tick("update", tp)
        if getattr(self, "profile", False):
            self.R.timings = {}                 # per-build stage timings (the Reconstructor accumulates them otherwise)
        t1 = time.time()
        self.build()
        t_build = time.time() - t1
        tp = self._tick("build", t1)
        if getattr(self, "profile", False):
            for k, v in self.R.timings.items():
                if k in ("grid", "generate", "neighbors", "solve", "project", "fix", "far"):
                    self.prof["build/" + k] = self.prof.get("build/" + k, 0.0) + v
        info = {"it": it, "loss": loss, "psnr": psnr, "n_sp": int(self.sp_pos.shape[0]), "n_fp": self.R.mpu.fp_stats()["n_fp"],
                "n_del": n_del, "mean_step": float(nrm.mean()), "t_vel": t_vel, "t_build": t_build}
        if a.resample_every > 0 and it % a.resample_every == 0 and it > 0 and not getattr(self, "no_resample", False):
            t2 = time.time()
            mode = a.resample_mode
            if getattr(a, "resample_mode2", "same") != "same" and it >= a.lr_switch:
                mode = a.resample_mode2          # e.g. Poisson/DPSR while growing, MPU marching cubes once converged
            if mode != "none":
                self.resample(mode)
            info["t_resample"] = time.time() - t2
            self._tick("resample", t2)
        self.log.append(info)
        return info

    # ------------------------------------------------------------------ resampling
    def resample(self, mode):
        a = self.args
        n_sp = a.n_sp
        if a.n_sp_start > 0 and a.grow_until > 0:
            it = getattr(self, "it", 0)
            frac = min(1.0, it / float(a.grow_until))
            n_sp = int(a.n_sp_start + (a.n_sp - a.n_sp_start) * frac)
        if mode == "mc":
            V, F = self.mesh(a.resample_res)
            V, F = largest_components(V, F, a.min_comp_frac)
        elif mode == "poisson":
            import open3d as o3d
            if getattr(a, "orient_mst", 0) > 0 and getattr(a, "orient_mst_resample", 1):
                self.sp_nrm, fl = orient_mst(self.sp_pos, self.sp_nrm, K=a.orient_mst, signed=bool(getattr(a, 'orient_mst_signed', 0)))
                print(f"[orient] resample MST orientation flipped {int(fl.sum())} normals", flush=True)
            if getattr(a, "poisson_safe", 1):
                from .poisson_worker import poisson_safe
                out = poisson_safe(self.sp_pos.double().cpu().numpy(), self.sp_nrm.double().cpu().numpy(), a.poisson_depth,
                                   seed=int(getattr(self, "it", 0)))
                if out is None:
                    print("[resample] Poisson failed in every attempt, skipping this resample", flush=True)
                    return
                Vp, Fp, dens = out
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vp), o3d.utility.Vector3iVector(Fp.astype(np.int32)))
            else:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.sp_pos.double().cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(self.sp_nrm.double().cpu().numpy())
                mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=a.poisson_depth, linear_fit=True)
                dens = np.asarray(dens)
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, a.poisson_trim))
            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.triangles, dtype=np.int64)
            V, F = largest_components(V, F, a.min_comp_frac)
        elif mode == "dpsr":
            from .dpsr import DPSR
            if not hasattr(self, "dpsr"):
                self.dpsr = DPSR((a.dpsr_res,) * 3, sig=a.dpsr_sig).to(self.dev)
            P = ((self.sp_pos + 1.0) * 0.5).clamp(0, 0.999999)
            with torch.no_grad():
                phi = self.dpsr(P[None], self.sp_nrm[None])[0]
            if phi[0, 0, 0] < 0:   # make outside positive
                phi = -phi
            V, F = run_marching_cubes(phi.contiguous(), a.dpsr_res, device=self.dev, corner_aligned=True)
            V, F = largest_components(V, F, a.min_comp_frac)
        else:
            return
        if V.shape[0] == 0:
            print("[resample] empty mesh, skipping", flush=True)
            return
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        self.cfg["far_winding_area"] = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        vn = vertex_normals_smooth(V, F, iters=1)
        P, N, _, _ = sample_surface(V, F, n_sp, device=self.dev, seed=int(time.time()) % 100000, vertex_normals=vn)
        cur = torch.zeros(n_sp, device=self.dev)
        if self.args.carry_curvature:
            # neighbourhood-based curvature of the new samples (avoids igl on possibly degenerate meshes)
            m = self.R.mpu
            m.bin_only(P, N, cur, self.cfg["thresholds"])
            cur = m.sp_curvature_pca(K=self.args.pca_K, device=self.dev).clamp(0, self.args.curvature_max)
        self.sp_pos, self.sp_nrm, self.sp_cur = P, N, cur
        self.sp_vel = torch.zeros((n_sp, 3), device=self.dev)
        a = self.args
        if a.task == "chamfer" and a.far_delete_cells > 0 and getattr(self, "it", 0) >= a.lr_switch:
            # drop resampled particles that Poisson placed on membranes far from every target point
            _, dfar = knn_idx(self.sp_pos, self.gt_pts, 1)
            keep = dfar[:, 0].sqrt() < a.far_delete_cells * self.cell_fine
            if keep.sum() > 1000:
                self.sp_pos, self.sp_nrm, self.sp_cur, self.sp_vel = self.sp_pos[keep], self.sp_nrm[keep], self.sp_cur[keep], self.sp_vel[keep]
                print(f"[resample] far-deletion removed {int((~keep).sum())} of {n_sp} resampled particles", flush=True)
        self.build()

    # ------------------------------------------------------------------ final dense flow
    def final_flow(self):
        """After the final (dense) resample: K more flow iterations with late-phase parameters (no resample, no
        densification) and optionally a denser GT point set, so that the resampled particles sit on the target again."""
        a = self.args
        if getattr(a, "final_n_gt", 0) > 0 and a.task == "chamfer":
            self.gt_pts, self.gt_nrm, _, _ = sample_surface(self.gt_V, self.gt_F, int(a.final_n_gt), device=self.dev, seed=2)
        it0 = max(int(getattr(self, "it", 0)) + 1, int(a.lr_switch))
        self.no_resample = True
        t0 = time.time()
        K = int(a.final_flow_iters)
        s0, l0 = float(a.speed_cells2), float(a.lr2)
        anneal = float(getattr(a, "final_flow_anneal", 0.0))
        for k in range(K):
            if anneal > 0:
                f = anneal ** (k / max(K - 1, 1))          # geometric decay 1 -> anneal
                a.speed_cells2, a.lr2 = s0 * f, l0 * f
            info = self.step(it0 + k)
            if k % 10 == 0 or k == int(a.final_flow_iters) - 1:
                print(f"[final] flow {k}: loss {info['loss']:.3e} psnr {info['psnr'] if info['psnr'] is None else round(info['psnr'], 2)} n_sp {info['n_sp']} n_fp {info['n_fp']} del {info['n_del']} "
                      f"step {info['mean_step']:.2e} t_vel {info['t_vel']:.1f} t_build {info['t_build']:.1f} elapsed {time.time() - t0:.0f}s", flush=True)
        self.no_resample = False
        a.speed_cells2, a.lr2 = s0, l0
        if getattr(a, "final_flow_mst", 1) and getattr(a, "orient_mst", 0) > 0:
            self.sp_nrm, fl = orient_mst(self.sp_pos, self.sp_nrm, K=a.orient_mst, signed=bool(getattr(a, 'orient_mst_signed', 0)))
            print(f"[final] MST orientation after the flow flipped {int(fl.sum())} normals", flush=True)
            self.build()

    # ------------------------------------------------------------------ final fine re-fit
    def final_refit(self):
        """Re-fit the final particles with a finer MPU configuration (Table-1 style: more levels, graded
        curvature thresholds, thin-structure boost). The flow itself keeps the coarse configuration."""
        a = self.args
        L = int(a.final_levels)
        cfg2 = dict(self.cfg)
        cfg2["num_levels"] = L
        if getattr(a, "final_fine_res", 0) > 0:
            cfg2["base_res"] = int(a.final_fine_res) // (2 ** (L - 1))
        if getattr(a, "final_thresholds", ""):
            cfg2["thresholds"] = [float(x) for x in a.final_thresholds.split(",")]
        else:
            cfg2["thresholds"] = [0.0] + [85.0 * 2.0 ** (-(L - 1 - l)) for l in range(1, L)]
        assert len(cfg2["thresholds"]) == L, "final_thresholds must have final_levels entries"
        cfg2["max_fp"] = int(getattr(a, "final_max_fp", 2 ** 21))
        cfg2["n_sp"] = int(self.sp_pos.shape[0]) + 16
        print(f"[final] re-fit with {L} levels (fine {cfg2['base_res'] * 2 ** (L - 1)}), thresholds {cfg2['thresholds']}, "
              f"{self.sp_pos.shape[0]} particles, far area {cfg2['far_winding_area']:.3f}", flush=True)
        R2 = Reconstructor(cfg2, device=self.dev)
        cur = self.sp_cur
        if getattr(a, "final_recompute_curv", 1) or getattr(a, "final_thin_boost", 0.0) > 0:
            R2.mpu.bin_only(self.sp_pos, self.sp_nrm, cur, cfg2["thresholds"])
        if getattr(a, "final_recompute_curv", 1):
            cur = R2.mpu.sp_curvature_pca(K=a.pca_K, device=self.dev).clamp(0, a.curvature_max)
        if getattr(a, "final_thin_boost", 0.0) > 0:
            lvl = max(L - 1 - int(getattr(a, "final_thin_level_offset", 3)), 0)
            thick = R2.mpu.sp_thickness(lvl)
            boost = a.final_thin_boost / thick.clamp_min(1e-4)
            boost = torch.where(thick > 1e8, torch.zeros_like(boost), boost)
            cur = torch.maximum(cur, boost)
            cell = 2.0 / (cfg2["base_res"] * 2 ** lvl)
            print(f"[final] thin boost at level {lvl}: frac thin<2cells {(thick < 2 * cell).float().mean().item():.3f}, "
                  f"curvature mean {float(cur.mean()):.1f} p90 {float(torch.quantile(cur, 0.9)):.1f}", flush=True)
        self.sp_cur = cur
        self.R, self.cfg = R2, cfg2
        self.R.build(self.sp_pos, self.sp_nrm, self.sp_cur, verbose=False)
        print(f"[final] re-fit done: {self.R.mpu.fp_stats()['n_fp']} fps, per level {self.R.mpu.fp_stats()['per_level']}", flush=True)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, out_dir, mc_res=512, iou_res=256, chamfer_n=1_000_000):
        res = {}
        if getattr(self.args, "orient_mst", 0) > 0:
            self.sp_nrm, fl = orient_mst(self.sp_pos, self.sp_nrm, K=self.args.orient_mst, signed=bool(getattr(self.args, 'orient_mst_signed', 0)))
            print(f"[orient] final MST orientation flipped {int(fl.sum())} normals", flush=True)
            self.build()
        if getattr(self.args, "orient_thin", 0) > 0:
            self.sp_nrm, fl = orient_thin(self.sp_pos, self.sp_nrm, self.cell_fine, max_cells=self.args.orient_thin)
            print(f"[orient] final: flipped {int(fl.sum())} thin-sheet normals", flush=True)
            self.build()
        if getattr(self.args, "final_resample", "none") != "none":
            n_sp_save, pd_save, pt_save = self.args.n_sp, self.args.poisson_depth, self.args.poisson_trim
            if getattr(self.args, "final_n_sp", 0) > 0:
                self.args.n_sp = int(self.args.final_n_sp)
            if getattr(self.args, "final_poisson_depth", 0) > 0:
                self.args.poisson_depth = int(self.args.final_poisson_depth)
            if getattr(self.args, "final_poisson_trim", -1.0) >= 0:
                self.args.poisson_trim = float(self.args.final_poisson_trim)
            self.resample(self.args.final_resample)
            self.args.n_sp, self.args.poisson_depth, self.args.poisson_trim = n_sp_save, pd_save, pt_save
        if getattr(self.args, "final_flow_iters", 0) > 0:
            self.final_flow()
        if getattr(self.args, "final_levels", 0) > 0:
            self.final_refit()
        field = self.carve(self.R.field(mc_res), mc_res)
        V, F = run_marching_cubes(field, mc_res, device=self.dev)
        trimesh.Trimesh(V, F, process=False).export(os.path.join(out_dir, "final.ply"))
        Q = M.grid_points(iou_res, device=self.dev)
        occ_gt = M.winding_occupancy(self.gt_V, self.gt_F, Q.double().cpu().numpy())
        f_iou, _ = self.R.eval(Q)
        if getattr(self.args, "hull_carve", 0):
            f_iou = torch.maximum(f_iou, self.hull_grid(iou_res).reshape(-1).to(f_iou.dtype))
        res["iou"] = M.iou(occ_gt, (f_iou < 0).cpu().numpy())
        if V.shape[0] > 0:
            res.update(M.chamfer_and_nae(self.gt_V, self.gt_F, V, F, n_samples=chamfer_n, device=self.dev))
        if self.renderer is not None:
            imgs = self.renderer.render_mesh_np(V, F)
            res["psnr_train"] = float(-10 * torch.log10((imgs - self.target).square().mean().clamp_min(1e-12)))
            imgs_t = self.renderer_test.render_mesh_np(V, F)
            res["psnr_test"] = float(-10 * torch.log10((imgs_t - self.target_test).square().mean().clamp_min(1e-12)))
            # PSNR averaged per view (common convention)
            per = -10 * torch.log10((imgs - self.target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12))
            res["psnr_train_perview"] = float(per.mean())
        res["n_fp"] = self.R.mpu.fp_stats()["n_fp"]
        res["n_sp"] = int(self.sp_pos.shape[0])
        return res
