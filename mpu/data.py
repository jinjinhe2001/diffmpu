"""Mesh loading, normalisation and oriented sample particle generation with curvature."""
import numpy as np
import torch
import trimesh
import scipy.sparse as sp


def load_mesh(path, process=False):
    m = trimesh.load(path, force="mesh", process=process)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    try:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol = float((np.cross(v1, v2) * v0).sum() / 6.0)
        if vol < 0:
            print(f"[load] negative signed volume ({vol:.3g}): flipping face orientation", flush=True)
            F = F[:, [0, 2, 1]]
    except Exception:
        pass
    return V, F


def normalize(V, scale_factor=1.2, center="bbox"):
    """Center the mesh and scale so that max |coord| = 1/scale_factor (fits in [-1,1])."""
    if center == "bbox":
        c = (V.max(0) + V.min(0)) / 2.0
    else:
        c = V.mean(0)
    V = V - c
    s = np.abs(V).max() * scale_factor
    return V / s, c, s


def vertex_mean_curvature(V, F, smooth_iters=0):
    """|mean curvature normal| per vertex = 2|H| via cotangent Laplacian / Voronoi mass."""
    import igl
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    md = np.asarray(M.diagonal()).ravel()
    md = np.where(md > 1e-18, md, 1e-18)
    Minv = sp.diags(1.0 / md)
    HN = -Minv.dot(L.dot(V))
    h = np.linalg.norm(HN, axis=1)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_iters > 0:
        A = igl.adjacency_matrix(F).astype(np.float64)
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        Dinv = sp.diags(1.0 / deg)
        for _ in range(smooth_iters):
            h = 0.5 * h + 0.5 * Dinv.dot(A.dot(h))
    return h


def vertex_normals_smooth(V, F, iters=0):
    """Area-weighted vertex normals, optionally smoothed over the 1-ring graph."""
    import igl
    N = igl.per_vertex_normals(np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(F, dtype=np.int64))
    N = np.nan_to_num(N)
    if iters > 0:
        A = igl.adjacency_matrix(np.ascontiguousarray(F, dtype=np.int64)).astype(np.float64)
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        Dinv = sp.diags(1.0 / deg)
        for _ in range(iters):
            N = 0.5 * N + 0.5 * Dinv.dot(A.dot(N))
            N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return N


def robust_curvature(V, F, target_scale=0.004, radius=None, smooth_iters=0):
    """2|H| from quadric fitting over k-ring neighbourhoods (igl.principal_curvature), with the ring
    radius chosen so that the neighbourhood spans ~target_scale (in normalised units)."""
    import igl
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    if radius is None:
        e = V[F[:, 1]] - V[F[:, 0]]
        mean_edge = float(np.linalg.norm(e[:min(len(e), 200000)], axis=1).mean())
        radius = int(np.clip(round(target_scale / max(mean_edge, 1e-9)), 2, 8))
    try:
        res = igl.principal_curvature(V, F, int(radius), True)
        pd1, pd2, pv1, pv2 = res[0], res[1], res[2], res[3]
        h = np.abs(np.nan_to_num(pv1) + np.nan_to_num(pv2))
    except Exception as ex:
        print("[curv] principal_curvature failed, falling back to cotan:", repr(ex)[:200], flush=True)
        h = vertex_mean_curvature(V, F)
    if smooth_iters > 0:
        A = igl.adjacency_matrix(F).astype(np.float64)
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1
        Dinv = sp.diags(1.0 / deg)
        for _ in range(smooth_iters):
            h = 0.5 * h + 0.5 * Dinv.dot(A.dot(h))
    return h, radius


def sample_surface(V, F, n, vertex_scalar=None, device="cuda", seed=0, vertex_normals=None):
    """Area-weighted surface sampling on GPU. Returns points, normals (face normals, or interpolated
    vertex normals if given), interpolated scalar, face ids."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    Ft = torch.as_tensor(F, dtype=torch.long, device=device)
    v0, v1, v2 = Vt[Ft[:, 0]], Vt[Ft[:, 1]], Vt[Ft[:, 2]]
    cr = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = cr.norm(dim=1) * 0.5
    fn = cr / (2.0 * area[:, None]).clamp_min(1e-20)
    probs = area / area.sum()
    # multinomial has a 2^24 category limit; fall back to CDF sampling for huge meshes
    if probs.shape[0] < 2 ** 24:
        fid = torch.multinomial(probs, n, replacement=True, generator=g)
    else:
        cdf = torch.cumsum(probs.double(), 0)
        u = torch.rand(n, device=device, generator=g, dtype=torch.float64)
        fid = torch.searchsorted(cdf, u).clamp_max(probs.shape[0] - 1)
    r1 = torch.rand(n, device=device, generator=g)
    r2 = torch.rand(n, device=device, generator=g)
    s1 = torch.sqrt(r1)
    a = 1.0 - s1
    b = s1 * (1.0 - r2)
    c = s1 * r2
    P = a[:, None] * v0[fid] + b[:, None] * v1[fid] + c[:, None] * v2[fid]
    N = fn[fid]
    if vertex_normals is not None:
        vn = torch.as_tensor(vertex_normals, dtype=torch.float32, device=device)
        Ni = a[:, None] * vn[Ft[fid, 0]] + b[:, None] * vn[Ft[fid, 1]] + c[:, None] * vn[Ft[fid, 2]]
        Ni = Ni / Ni.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # keep orientation consistent with the face normal
        flip = (Ni * N).sum(1, keepdim=True) < 0
        N = torch.where(flip, -Ni, Ni)
    S = None
    if vertex_scalar is not None:
        st = torch.as_tensor(vertex_scalar, dtype=torch.float32, device=device)
        S = a * st[Ft[fid, 0]] + b * st[Ft[fid, 1]] + c * st[Ft[fid, 2]]
    return P, N, S, fid


def make_sample_particles(V, F, n, curv_smooth=0, curv_mode="robust", device="cuda", seed=0,
                          normal_mode="vertex", normal_smooth=1, curv_scale=0.004):
    """Returns (pos, nrm, cur) torch cuda tensors for the normalised mesh (V, F)."""
    h = None
    info = {}
    if curv_mode == "mesh":
        h = vertex_mean_curvature(V, F, smooth_iters=curv_smooth)
    elif curv_mode == "robust":
        h, rad = robust_curvature(V, F, target_scale=curv_scale, smooth_iters=curv_smooth)
        info["curv_radius"] = rad
    vn = vertex_normals_smooth(V, F, iters=normal_smooth) if normal_mode == "vertex" else None
    P, N, S, fid = sample_surface(V, F, n, vertex_scalar=h, device=device, seed=seed, vertex_normals=vn)
    if S is None:
        S = torch.zeros(n, device=device)
    return P, N, S
