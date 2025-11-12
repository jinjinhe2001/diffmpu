"""Reconstruction quality metrics: IoU, Chamfer-L2 / L1, normal angular error (NAE)."""
import numpy as np
import torch
from scipy.spatial import cKDTree

from .data import sample_surface


def winding_occupancy(V, F, Q, batch=4_000_000):
    """Inside test by generalised winding number (robust to small holes)."""
    import igl
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    out = np.zeros(Q.shape[0], dtype=np.float64)
    fn = getattr(igl, "fast_winding_number_for_meshes", None)
    if fn is None:
        fn = getattr(igl, "fast_winding_number")
    for s in range(0, Q.shape[0], batch):
        out[s:s + batch] = np.asarray(fn(V, F, Q[s:s + batch])).ravel()
    return out > 0.5


def grid_points(res, device="cuda"):
    """Cell-centred grid over [-1,1]^3 (res^3 points), ordered (x, y, z) with z fastest."""
    ax = (torch.arange(res, device=device, dtype=torch.float32) + 0.5) * (2.0 / res) - 1.0
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)


def iou(occ_a, occ_b):
    inter = np.logical_and(occ_a, occ_b).sum()
    union = np.logical_or(occ_a, occ_b).sum()
    return float(inter) / float(max(union, 1))


def point_mesh_sqdist(P, V, F):
    import igl
    sqd, I, C = igl.point_mesh_squared_distance(np.ascontiguousarray(P, dtype=np.float64),
                                                np.ascontiguousarray(V, dtype=np.float64),
                                                np.ascontiguousarray(F, dtype=np.int64))
    return np.asarray(sqd).ravel(), np.asarray(I).ravel(), np.asarray(C)


def face_normals(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(l, 1e-20)


def chamfer_and_nae(gt_V, gt_F, re_V, re_F, n_samples=1_000_000, device="cuda", seed=1,
                    grad_fn=None):
    """Point-to-mesh (exact) and point-to-point Chamfer distances plus normal angular error.

    Returns dict with:
      cl2_p2m : mean sq dist gt->rec + mean sq dist rec->gt   (exact point-to-mesh)
      cl1_p2m : mean |d| both ways summed
      cl2_p2p_100k / cl2_p2p_1m : DeepSDF/NIE style KD-tree point-to-point (sum of means of sq dist)
      nae_mesh : mean angle (deg) between GT sample normal and closest rec face normal
      nae_grad : mean angle (deg) between GT sample normal and analytic field gradient at GT point
    """
    out = {}
    from .data import vertex_normals_smooth
    vn_gt = vertex_normals_smooth(gt_V, gt_F, iters=0)
    Pg, Ng_face, _, _ = sample_surface(gt_V, gt_F, n_samples, device=device, seed=seed)
    _, Ng, _, _ = sample_surface(gt_V, gt_F, n_samples, device=device, seed=seed, vertex_normals=vn_gt)
    vn_re = vertex_normals_smooth(re_V, re_F, iters=0) if re_V.shape[0] > 0 else None
    Pr, Nr, _, _ = sample_surface(re_V, re_F, n_samples, device=device, seed=seed + 1)
    Pg_np = Pg.double().cpu().numpy()
    Pr_np = Pr.double().cpu().numpy()
    Ng_np = Ng.double().cpu().numpy()

    sq_g2r, I_g2r, _ = point_mesh_sqdist(Pg_np, re_V, re_F)
    sq_r2g, _, _ = point_mesh_sqdist(Pr_np, gt_V, gt_F)
    out["cl2_p2m"] = float(sq_g2r.mean() + sq_r2g.mean())
    out["cl2_p2m_g2r"] = float(sq_g2r.mean())
    out["cl2_p2m_r2g"] = float(sq_r2g.mean())
    out["cl1_p2m"] = float(np.sqrt(sq_g2r).mean() + np.sqrt(sq_r2g).mean())
    out["hausdorff_g2r"] = float(np.sqrt(sq_g2r.max()))
    out["hausdorff_r2g"] = float(np.sqrt(sq_r2g.max()))

    # point-to-point (NIE / DeepSDF compute_trimesh_chamfer style)
    for n_pp, tag in ((100_000, "100k"), (min(n_samples, 500_000), "500k"), (min(n_samples, 1_000_000), "1m")):
        a = Pg_np[:n_pp]
        b = Pr_np[:n_pp]
        d_ab, _ = cKDTree(b).query(a, workers=-1)
        d_ba, _ = cKDTree(a).query(b, workers=-1)
        out[f"cl2_p2p_{tag}"] = float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))
        out[f"cl1_p2p_{tag}"] = float(np.mean(d_ab) + np.mean(d_ba))

    # NAE from reconstructed mesh normals at the closest face (face normals / vertex normals)
    fn_r = face_normals(re_V, re_F)
    nr = fn_r[I_g2r]
    Ng_face_np = Ng_face.double().cpu().numpy()
    for tag, gtn in (("", Ng_np), ("_gtface", Ng_face_np)):
        cos = np.clip(np.abs((nr * gtn).sum(1)), 0, 1)   # orientation-agnostic
        out["nae_mesh" + tag] = float(np.degrees(np.arccos(cos)).mean())
    cos_o = np.clip((nr * Ng_np).sum(1), -1, 1)
    out["nae_mesh_oriented"] = float(np.degrees(np.arccos(cos_o)).mean())
    if vn_re is not None:
        # vertex-normal of the reconstructed mesh at the closest point (barycentric via face vertices avg)
        f = re_F[I_g2r]
        nrv = vn_re[f].mean(1)
        nrv = nrv / np.maximum(np.linalg.norm(nrv, axis=1, keepdims=True), 1e-12)
        cos = np.clip(np.abs((nrv * Ng_np).sum(1)), 0, 1)
        out["nae_mesh_vn"] = float(np.degrees(np.arccos(cos)).mean())

    if grad_fn is not None:
        g = grad_fn(Pg)  # (N,3) torch
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosg = (g * Ng).sum(1).clamp(-1, 1)
        out["nae_grad_oriented"] = float(torch.rad2deg(torch.arccos(cosg)).mean().item())
        out["nae_grad"] = float(torch.rad2deg(torch.arccos(cosg.abs())).mean().item())
        cosf = (g * Ng_face).sum(1).clamp(-1, 1)
        out["nae_grad_gtface"] = float(torch.rad2deg(torch.arccos(cosf.abs())).mean().item())
    return out
