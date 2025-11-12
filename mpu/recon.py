"""Direct implicit surface reconstruction (paper Algorithm 2) driver."""
import time
import numpy as np
import torch
import taichi as ti

from .core import MPU


def knn1(query, ref, chunk=2 ** 20):
    """Nearest neighbour index of each query in ref (exact grid search on the GPU; brute force fallback)."""
    out = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    try:
        if query.is_cuda and ref.shape[0] >= 4096 and query.shape[0] >= 4096:
            from .core import grid_nn
            # the far cells are far from the surface: a coarse grid (cell = 1/32 of the bbox) with a 10-cell ring cap
            return grid_nn(query, ref, res=32, max_ring=10, cache=False)[0][:, 0]
        from pytorch3d.ops import knn_points
        for s in range(0, query.shape[0], chunk):
            q = query[s:s + chunk]
            out[s:s + chunk] = knn_points(q[None], ref[None], K=1, return_sorted=False).idx[0, :, 0]
        return out
    except Exception:
        for s in range(0, query.shape[0], 65536):
            q = query[s:s + 65536]
            d = torch.cdist(q, ref)
            out[s:s + 65536] = d.argmin(dim=1)
        return out


class Reconstructor:
    """Direct MPU reconstruction (paper Algorithm 2) around one mpu.core.MPU instance (self.mpu); cfg is a default_config()
    dict kept by reference (self.cfg; cfg["thresholds"] edits between builds take effect). build(sp_pos, sp_nrm, sp_cur):
    bins the oriented sample particles (torch CUDA (N,3), (N,3), (N,), normalised frame) into the multi-level sparse grid,
    creates one quadric feature particle per activated cell, fits them by least squares, then runs the error-refinement,
    Newton projection (moving feature particles) and hole-fixing rounds of cfg; returns fp_stats(). build(far=False) defers
    the winding-number far field to the first off-particle query (ensure_far). eval(P) -> (f, grad): blended field (and
    gradient) at query points P (M,3) in the domain [-1,1]^3; field(res) -> (res,res,res) tensor on the cell centres;
    marching_cubes(res) -> numpy (V, F) of the zero level set. Sign convention: f < 0 inside, f > 0 outside (far-field sign
    from the winding number). Feature particles live in the Taichi fields self.mpu.fp_pos / fp_nrm / fp_radius / fp_layer:
    the first fp_stats()["n_fp_alloc"] entries are allocated, fp_radius > 0 marks the fp_stats()["n_fp"] active ones.
    """

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        c = cfg
        aux = build_aux_offsets(c)
        self.mpu = MPU(base_res=c["base_res"], num_levels=c["num_levels"], max_sp=c["n_sp"] + 16,
                       max_fp=c["max_fp"], max_sp_per_cell=c["max_sp_per_cell"],
                       max_fp_per_cell=c["max_fp_per_cell"], max_nb=c["max_nb"],
                       search_extent=c["search_extent"], weight_type=c["weight_type"],
                       level_gain=c["level_gain"], alpha=c["alpha"], aux_offsets=aux,
                       far_levels=c.get("far_levels"), block_size=c.get("block_size", 4))
        self.mpu.far_stage1 = int(c.get("far_stage1", 20_000)); self.mpu.far_stage1_band = float(c.get("far_stage1_band", 0.25))
        self.timings = {}
        self.log = []

    def _t(self, key, t0):
        self.timings[key] = self.timings.get(key, 0.0) + (time.time() - t0)

    def fit(self):
        c = self.cfg
        t0 = time.time()
        self.mpu.collect_neighbors()
        ti.sync()
        self._t("neighbors", t0)
        t0 = time.time()
        n_empty = self.mpu.fit(aux_k=c["aux_k"], aux_w=c["aux_w"], aux_sign_check=c["aux_sign_check"],
                               sp_w_scale=c["sp_w_scale"], ridge=c["ridge"], device=self.device)
        ti.sync()
        self._t("solve", t0)
        return n_empty

    def build(self, sp_pos, sp_nrm, sp_cur, verbose=True, far=True):
        """far=False skips the winding-number far field; it is then computed lazily by the first off-particle query
        (eval / field / marching_cubes) through ensure_far(), so results are identical."""
        c = self.cfg
        m = self.mpu
        t_all = time.time()
        t0 = time.time()
        m.set_sample_particles(sp_pos, sp_nrm, sp_cur)
        ti.sync()
        self._t("grid_fill", t0)
        t1 = time.time()
        m.clear_grid()
        ti.sync()
        self._t("grid_clear", t1)
        t1 = time.time()
        m.set_thresholds(c["thresholds"])
        for l in range(m.num_levels):
            t2 = time.time()
            m._bin_sp(l)
            ti.sync()
            self._t(f"grid_bin{l}", t2)
        self._t("grid_bin", t1)
        self._t("grid", t0)
        t0 = time.time()
        n_fp = m.generate_feature_particles()
        ti.sync()
        self._t("generate", t0)
        if verbose:
            print(f"[recon] generated {n_fp} fps, per level {m.fp_stats(self.device)['per_level']}", flush=True)
        if n_fp >= m.max_fp:
            raise RuntimeError(f"too many feature particles ({n_fp} >= max_fp {m.max_fp})")
        n_empty = self.fit()

        # error-based refinement rounds
        for it in range(c["n_refine"]):
            t0 = time.time()
            m.compute_fit_error(int(c.get("refine_use_aux", 1)), float(c.get("refine_aux_scale", 1.0)))
            m.clear_refine()
            n_mark = m.mark_refine(c["refine_tol"])
            if n_mark == 0:
                break
            m.apply_refine_marks()
            n_fp = m.generate_feature_particles()
            ti.sync()
            self._t("refine", t0)
            if n_fp >= m.max_fp:
                raise RuntimeError(f"too many feature particles ({n_fp} >= max_fp {m.max_fp})")
            n_empty = self.fit()
            if verbose:
                print(f"[recon] refine {it}: marked {n_mark} cells -> {n_fp} fps {m.fp_stats(self.device)['per_level']}", flush=True)

        # Newton projection rounds (moving feature particles)
        for it in range(c["n_proj"]):
            t0 = time.time()
            step = m.project(c["proj_max_frac"])
            m.allocate_fp()
            ti.sync()
            self._t("project", t0)
            n_empty = self.fit()
            if verbose:
                print(f"[recon] proj {it}: mean |step|/r = {step:.4f}, empty fps {n_empty}", flush=True)
            if step < c["proj_tol"]:
                break

        # hole fixing rounds
        for it in range(c["n_fix"]):
            t0 = time.time()
            n_unc, n_add = m.fix_holes(radius_scale=c["fix_radius_scale"], device=self.device)
            ti.sync()
            self._t("fix", t0)
            if verbose:
                print(f"[recon] fix {it}: uncovered sps {n_unc}, added {n_add} fps", flush=True)
            if n_add == 0:
                break
            m.allocate_fp()
            n_empty = self.fit()
            if c["fix_project"]:
                m.project(c["proj_max_frac"])
                m.allocate_fp()
                n_empty = self.fit()

        # far field (winding-number sign for the empty cells); optional here, computed lazily on demand otherwise
        self.far_valid = False
        if far:
            self.compute_far()
        self.timings["total_build"] = time.time() - t_all
        stats = m.fp_stats(self.device)
        if verbose:
            print(f"[recon] done: {stats} timings {self.timings}", flush=True)
        return stats

    def compute_far(self):
        c = self.cfg
        m = self.mpu
        t0 = time.time()
        n = m.sp_n[None]
        sub = min(n, c["far_sub"])
        idx = torch.randperm(n, device=self.device)[:sub]
        sp_pos_t = m.sp_pos.to_torch(device=self.device)[:n]
        sp_nrm_t = m.sp_nrm.to_torch(device=self.device)[:n]
        wpos = wnrm = warea = None
        if c.get("far_winding", 0) > 0:
            k = min(n, int(c["far_winding"]))
            widx = torch.randperm(n, device=self.device)[:k]
            wpos = sp_pos_t[widx].contiguous()
            wnrm = sp_nrm_t[widx].contiguous()
            warea = float(c.get("surface_area", 0.0)) / k if c.get("surface_area", 0.0) > 0 else None
            if warea is None:
                # estimate area from sample density: mean nn distance^2 * N (rough), fallback constant
                warea = float(c.get("far_winding_area", 4.0)) / k
        m.compute_far_field(sp_pos_t[idx].contiguous(), sp_nrm_t[idx].contiguous(), knn1, device=self.device,
                            winding_pos=wpos, winding_nrm=wnrm, winding_area=warea)
        ti.sync()
        self._t("far", t0)
        self.far_valid = True

    def ensure_far(self):
        if not getattr(self, "far_valid", True):
            self.compute_far()

    def eval(self, P, batch=2 ** 24, return_grad=False):
        self.ensure_far()
        fs, gs = [], []
        for s in range(0, P.shape[0], batch):
            f, g, _ = self.mpu.eval(P[s:s + batch], return_grad=return_grad, device=self.device)
            fs.append(f)
            if return_grad:
                gs.append(g)
        f = torch.cat(fs)
        g = torch.cat(gs) if return_grad else None
        return f, g

    def field(self, res, batch=2 ** 24):
        self.ensure_far()
        from .metrics import grid_points
        t0 = time.time()
        P = grid_points(res, device=self.device)
        f, _ = self.eval(P, batch=batch)
        ti.sync()
        self.timings[f"query_{res}"] = time.time() - t0
        return f.reshape(res, res, res)

    def marching_cubes(self, res, field=None):
        if field is None:
            field = self.field(res)
        t0 = time.time()
        V, Fc = run_marching_cubes(field, res, device=self.device)
        self.timings[f"mc_{res}"] = time.time() - t0
        return V, Fc


def run_marching_cubes(field, res, device="cuda", corner_aligned=False):
    """field: torch (res,res,res) on cell centres of [-1,1]^3 (or on corners i/res if corner_aligned).
    Returns numpy V, F."""
    spacing = 2.0 / res
    origin = -1.0 + (0.0 if corner_aligned else 0.5 * spacing)
    try:
        from pytorch3d.ops import marching_cubes as p3d_mc
        verts, faces = p3d_mc.marching_cubes(field[None].float(), isolevel=0.0, return_local_coords=False)
        V = verts[0].double().cpu().numpy()
        Fc = faces[0].long().cpu().numpy()
        # pytorch3d returns (W,H,D)-ordered coordinates: swap x<->z and flip winding to keep handedness
        V = V[:, [2, 1, 0]]
        Fc = Fc[:, [0, 2, 1]]
        V = origin + V * spacing
        if V.shape[0] > 0:
            return V, Fc
    except Exception as e:
        print(f"[mc] pytorch3d marching cubes failed ({e}); falling back to skimage", flush=True)
    import skimage.measure
    f_np = field.cpu().numpy()
    V, Fc, _, _ = skimage.measure.marching_cubes(f_np, level=0.0, spacing=(spacing,) * 3, method="lewiner")
    V = V + origin
    return V.astype(np.float64), Fc.astype(np.int64)


def build_aux_offsets(c):
    offs = []
    if c.get("aux_center27", True):
        s = c.get("aux_center_scale", 0.2)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    offs.append((s * i, s * j, s * k))
    for s in c.get("aux_shells", []):
        for ax in range(3):
            for sgn in (-1, 1):
                o = [0.0, 0.0, 0.0]
                o[ax] = sgn * s
                offs.append(tuple(o))
    n_rand = int(c.get("aux_random", 0))
    if n_rand > 0:
        rng = np.random.RandomState(12345)
        rad = float(c.get("aux_random_radius", 0.9))
        pts = []
        while len(pts) < n_rand:
            v = rng.uniform(-1, 1, 3)
            if np.dot(v, v) <= 1.0:
                pts.append(tuple((v * rad).tolist()))
        offs.extend(pts)
    return offs


def default_config():
    return dict(
        base_res=16, num_levels=6, n_sp=3_000_000, max_fp=2 ** 18,
        max_sp_per_cell=32, max_fp_per_cell=16, max_nb=512, search_extent=1,
        weight_type=0, level_gain=10.0, alpha=1.15,
        thresholds=[0.0] + [85.0] * 5,
        aux_center27=True, aux_center_scale=0.2, aux_shells=[], aux_k=6, aux_w=1.0 / 9.0,
        aux_sign_check=False, sp_w_scale=1.0, ridge=1e-7, aux_random=0, aux_random_radius=0.9,
        refine_use_aux=1, refine_aux_scale=1.0,
        n_refine=0, refine_tol=0.05,
        n_proj=3, proj_max_frac=1.0, proj_tol=1e-4,
        n_fix=3, fix_radius_scale=1.0, fix_project=False,
        far_sub=200_000, far_levels=None, far_winding=100_000, far_stage1=20_000, far_stage1_band=0.25, surface_area=0.0, far_winding_area=4.0,
    )
