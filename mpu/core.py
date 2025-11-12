"""
Core Taichi implementation of the multi-level partition-of-unity (MPU) moving-particle
representation: sparse multi-level grid, sample particles, feature particles with
quadratic patches, weighted least squares assembly, Newton projection and field evaluation.

Coordinates: the domain is [-1, 1]^3.  Level l has resolution base_res * 2**l.
"""
import numpy as np
import taichi as ti
import torch


@ti.func
def basis10(x):
    return ti.Vector([x[0] * x[0], x[1] * x[1], x[2] * x[2],
                      x[0] * x[1], x[1] * x[2], x[2] * x[0],
                      x[0], x[1], x[2], 1.0])


@ti.func
def quad_value(c, x):
    return (c[0] * x[0] * x[0] + c[1] * x[1] * x[1] + c[2] * x[2] * x[2]
            + c[3] * x[0] * x[1] + c[4] * x[1] * x[2] + c[5] * x[2] * x[0]
            + c[6] * x[0] + c[7] * x[1] + c[8] * x[2] + c[9])


@ti.func
def quad_grad(c, x):
    return ti.Vector([2.0 * c[0] * x[0] + c[3] * x[1] + c[5] * x[2] + c[6],
                      2.0 * c[1] * x[1] + c[3] * x[0] + c[4] * x[2] + c[7],
                      2.0 * c[2] * x[2] + c[4] * x[1] + c[5] * x[0] + c[8]])


@ti.kernel
def _winding_kernel(q: ti.types.ndarray(dtype=ti.f32, ndim=2), p: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    an: ti.types.ndarray(dtype=ti.f32, ndim=2), out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    """Brute-force generalised winding number, one thread per query, f64 accumulation. Compute-bound, whereas the
    torch version below is memory-bound (~8 (chunk x K) temporaries per chunk)."""
    for i in range(q.shape[0]):
        qx, qy, qz = q[i, 0], q[i, 1], q[i, 2]
        acc = ti.cast(0.0, ti.f64)
        for j in range(p.shape[0]):
            dx, dy, dz = p[j, 0] - qx, p[j, 1] - qy, p[j, 2] - qz
            r2 = ti.max(dx * dx + dy * dy + dz * dz, 1e-12)
            num = dx * an[j, 0] + dy * an[j, 1] + dz * an[j, 2]
            acc += ti.cast(num / (r2 * ti.sqrt(r2)), ti.f64)
        out[i] = ti.cast(acc / (4.0 * 3.141592653589793), ti.f32)


def point_winding_number(Q, P, N, area, chunk=1024, use_kernel=True):
    """Generalised winding number of an oriented point set (Barill et al. 2018), brute force.
    Q (M,3) queries, P (K,3) points, N (K,3) unit normals, area: scalar or (K,) per-point area."""
    if use_kernel and Q.is_cuda and Q.shape[0] > 0 and P.shape[0] > 0:
        a = area if torch.is_tensor(area) else torch.full((P.shape[0],), float(area), device=Q.device)
        an = (N * a.reshape(-1, 1)).float().contiguous()
        out = torch.empty(Q.shape[0], dtype=torch.float32, device=Q.device)
        _winding_kernel(Q.float().contiguous(), P.float().contiguous(), an, out)
        return out
    out = torch.empty(Q.shape[0], dtype=torch.float32, device=Q.device)
    pn = (P * N).sum(1)                       # p.n
    pp = (P * P).sum(1)                       # |p|^2
    if not torch.is_tensor(area):
        area = torch.full((P.shape[0],), float(area), device=Q.device)
    aN = N * area[:, None]
    apn = pn * area
    for s in range(0, Q.shape[0], chunk):
        q = Q[s:s + chunk]
        qn = q @ aN.t()                       # a * (q.n)
        num = apn[None, :] - qn               # a * (p - q).n
        d2 = pp[None, :] + (q * q).sum(1)[:, None] - 2.0 * (q @ P.t())
        d3 = d2.clamp_min(1e-12).pow(1.5)
        out[s:s + chunk] = (num / d3).sum(1) / (4.0 * 3.141592653589793)
    return out


@ti.kernel
def _grid_nn_kernel(q: ti.types.ndarray(dtype=ti.f32, ndim=2), r: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    cell_start: ti.types.ndarray(dtype=ti.i32, ndim=1), ox: ti.f32, oy: ti.f32, oz: ti.f32,
                    h: ti.f32, res: ti.i32, max_ring: ti.i32,
                    out_idx: ti.types.ndarray(dtype=ti.i32, ndim=1), out_d2: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    """Exact nearest neighbour in a uniform grid: scan the Chebyshev rings around the query cell until the best distance
    is below the distance to every unvisited cell. Unresolved queries (ring cap reached) get idx -1."""
    for i in range(q.shape[0]):
        qx, qy, qz = q[i, 0], q[i, 1], q[i, 2]
        gx, gy, gz = (qx - ox) / h, (qy - oy) / h, (qz - oz) / h
        cx = ti.min(ti.max(ti.cast(ti.floor(gx), ti.i32), 0), res - 1)
        cy = ti.min(ti.max(ti.cast(ti.floor(gy), ti.i32), 0), res - 1)
        cz = ti.min(ti.max(ti.cast(ti.floor(gz), ti.i32), 0), res - 1)
        fx, fy, fz = gx - cx, gy - cy, gz - cz            # offset in cell units (outside the grid: <0 or >=1)
        best_d2 = 3.4e38
        best = -1
        ring = 0
        done = False
        while ring <= max_ring and not done:
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    for dz in range(-ring, ring + 1):
                        if ti.max(ti.abs(dx), ti.max(ti.abs(dy), ti.abs(dz))) == ring:
                            ix, iy, iz = cx + dx, cy + dy, cz + dz
                            if ix >= 0 and iy >= 0 and iz >= 0 and ix < res and iy < res and iz < res:
                                cid = (ix * res + iy) * res + iz
                                for j in range(cell_start[cid], cell_start[cid + 1]):
                                    ex, ey, ez = r[j, 0] - qx, r[j, 1] - qy, r[j, 2] - qz
                                    d2 = ex * ex + ey * ey + ez * ez
                                    if d2 < best_d2:
                                        best_d2 = d2
                                        best = j
            # lower bound on the distance to any point in a cell of ring+1 or beyond (only directions where cells exist)
            lb = 3.4e38
            if cx + ring + 1 < res:
                lb = ti.min(lb, ring + 1 - fx)
            if cx - ring - 1 >= 0:
                lb = ti.min(lb, ring + fx)
            if cy + ring + 1 < res:
                lb = ti.min(lb, ring + 1 - fy)
            if cy - ring - 1 >= 0:
                lb = ti.min(lb, ring + fy)
            if cz + ring + 1 < res:
                lb = ti.min(lb, ring + 1 - fz)
            if cz - ring - 1 >= 0:
                lb = ti.min(lb, ring + fz)
            lbh = lb * h
            if best >= 0 and (lb >= 3.0e38 or best_d2 <= lbh * lbh):
                done = True
            ring += 1
        if done:
            out_idx[i] = best
            out_d2[i] = best_d2
        else:
            out_idx[i] = -1
            out_d2[i] = best_d2


@ti.kernel
def _brute_nn_kernel(q: ti.types.ndarray(dtype=ti.f32, ndim=2), r: ti.types.ndarray(dtype=ti.f32, ndim=2),
                     out_idx: ti.types.ndarray(dtype=ti.i32, ndim=1), out_d2: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    """Exact brute-force nearest neighbour, one thread per query (all threads stream the same reference points, which
    stay in L2). 260k x 200k in ~15 ms, 5x faster than pytorch3d's knn_points for K=1."""
    for i in range(q.shape[0]):
        qx, qy, qz = q[i, 0], q[i, 1], q[i, 2]
        best = 3.4e38
        bj = -1
        for j in range(r.shape[0]):
            dx, dy, dz = r[j, 0] - qx, r[j, 1] - qy, r[j, 2] - qz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best:
                best = d2
                bj = j
        out_idx[i] = bj
        out_d2[i] = best


def brute_nn(query, ref):
    """Exact nearest neighbour by brute force on the GPU: idx (N,) long, d2 (N,) float32."""
    out_idx = torch.empty(query.shape[0], dtype=torch.int32, device=query.device)
    out_d2 = torch.empty(query.shape[0], dtype=torch.float32, device=query.device)
    _brute_nn_kernel(query.float().contiguous(), ref.float().contiguous(), out_idx, out_d2)
    return out_idx.long(), out_d2


_GRID_CACHE = {}


def _fingerprint(ref):
    """Cheap identity check for a cached reference set: address, shape and three sampled rows (a new tensor can reuse the
    address of a freed one, so the address alone is not enough)."""
    M = ref.shape[0]
    rows = ref[[0, M // 2, M - 1]].detach().double().sum(1).tolist()
    return (ref.data_ptr(), tuple(ref.shape), str(ref.device), tuple(round(x, 9) for x in rows))


def grid_nn(query, ref, res=None, max_ring=6, cache=False):
    """Exact nearest neighbour of each query in ref on the GPU (uniform grid + ring search, brute force only for the few
    queries farther than max_ring cells from ref). Returns idx (N,1) long, d2 (N,1) float32 squared distances.
    res defaults to ~4 points per occupied cell for a surface sampling (capped at 256). cache=True keeps the grid of the
    reference set (for a static target; the entry is validated by a fingerprint). Building the grid costs ~1 ms anyway."""
    N, M = query.shape[0], ref.shape[0]
    if res is None:
        res = int(min(256, max(16, round((M / 7.0) ** 0.5))))
    key = (_fingerprint(ref), int(res)) if cache else None
    g = _GRID_CACHE.get(key) if cache else None
    if g is None:
        lo, hi = ref.min(0).values, ref.max(0).values
        ext = float((hi - lo).max()) * (1.0 + 2e-3) + 1e-6
        origin = lo - 1e-3 * ext
        h = ext / res
        c = ((ref - origin) / h).floor().long().clamp_(0, res - 1)
        cid = (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
        order = cid.argsort()
        counts = torch.bincount(cid, minlength=res ** 3)
        cell_start = torch.zeros(res ** 3 + 1, dtype=torch.int32, device=ref.device)
        cell_start[1:] = counts.cumsum(0).int()
        g = (origin.float(), h, order, ref[order].float().contiguous(), cell_start)
        if cache:
            if len(_GRID_CACHE) >= 4:
                _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
            _GRID_CACHE[key] = g
    origin, h, order, r_sorted, cell_start = g
    out_idx = torch.empty(N, dtype=torch.int32, device=query.device)
    out_d2 = torch.empty(N, dtype=torch.float32, device=query.device)
    _grid_nn_kernel(query.float().contiguous(), r_sorted, cell_start, float(origin[0]), float(origin[1]), float(origin[2]),
                    float(h), int(res), int(max_ring), out_idx, out_d2)
    idx = order[out_idx.clamp_min(0).long()]
    un = out_idx < 0
    if bool(un.any()):
        sel_all = un.nonzero()[:, 0]
        bi, bd = brute_nn(query[sel_all], ref)
        idx[sel_all] = bi
        out_d2[sel_all] = bd
    return idx[:, None], out_d2[:, None]


@ti.data_oriented
class MPU:
    def __init__(self, base_res=16, num_levels=6, max_sp=3_000_005, max_fp=2 ** 18,
                 max_sp_per_cell=32, max_fp_per_cell=16, max_nb=1024,
                 search_extent=1, weight_type=0, level_gain=10.0, alpha=1.15,
                 aux_offsets=None, far_levels=None, block_size=4):
        self.dim = 3
        self.num_levels = num_levels
        self.res = [base_res * 2 ** l for l in range(num_levels)]
        self.cell_size = [2.0 / r for r in self.res]
        self.max_sp = max_sp
        self.max_fp = max_fp
        self.max_sp_per_cell = max_sp_per_cell
        self.max_fp_per_cell = max_fp_per_cell
        self.max_nb = max_nb
        self.search_extent = int(search_extent)
        self.weight_type = int(weight_type)
        self.level_gain = float(level_gain)
        self.far_stage1 = 20_000          # winding points of the coarse pass (0 = single pass); set via Reconstructor cfg far_stage1
        self.far_stage1_band = 0.25
        self.alpha = float(alpha)
        self.far_levels = num_levels - 2 if far_levels is None else far_levels

        # ---- sample particles ----
        self.sp_n = ti.field(ti.i32, shape=())
        self.sp_pos = ti.Vector.field(3, ti.f32, shape=max_sp)
        self.sp_nrm = ti.Vector.field(3, ti.f32, shape=max_sp)
        self.sp_cur = ti.field(ti.f32, shape=max_sp)

        # ---- feature particles ----
        self.fp_n = ti.field(ti.i32, shape=())
        self.fp_pos = ti.Vector.field(3, ti.f32, shape=max_fp)
        self.fp_nrm = ti.Vector.field(3, ti.f32, shape=max_fp)
        self.fp_radius = ti.field(ti.f32, shape=max_fp)
        self.fp_layer = ti.field(ti.i32, shape=max_fp)
        self.fp_nb_num = ti.field(ti.i32, shape=max_fp)
        self.fp_nb_idx = ti.field(ti.i32, shape=(max_fp, max_nb))
        self.coef = ti.Vector.field(10, ti.f32, shape=max_fp)
        self.fp_err = ti.field(ti.f32, shape=max_fp)   # max |f(p)| / r over neighbours (fit error)

        # ---- multi-level sparse grid ----
        self.thresholds = ti.field(ti.f32, shape=num_levels)
        self.sp_num = [ti.field(ti.i32) for _ in range(num_levels)]
        self.sp_idx = [ti.field(ti.i32) for _ in range(num_levels)]
        self.act = [ti.field(ti.i8) for _ in range(num_levels)]      # curvature activated
        self.cov = [ti.field(ti.i8) for _ in range(num_levels)]      # covered by fp (own or children)
        self.refine = [ti.field(ti.i8) for _ in range(num_levels)]   # error-based refinement request
        self.fp_num = [ti.field(ti.i32) for _ in range(num_levels)]
        self.fp_idx = [ti.field(ti.i32) for _ in range(num_levels)]
        self.far_sdf = [ti.field(ti.f32) for _ in range(num_levels)]
        self.far_ok = [ti.field(ti.i8) for _ in range(num_levels)]
        self.block_size = [min(int(block_size), r) for r in self.res]
        self.blocks = [ti.root.pointer(ti.ijk, (r // b, r // b, r // b)) for r, b in zip(self.res, self.block_size)]
        self.pixels = [blk.dense(ti.ijk, (b, b, b)) for blk, b in zip(self.blocks, self.block_size)]
        for l in range(num_levels):
            self.pixels[l].place(self.sp_num[l], self.act[l], self.cov[l], self.refine[l],
                                 self.fp_num[l], self.far_sdf[l], self.far_ok[l])
            self.pixels[l].dense(ti.l, max_sp_per_cell).place(self.sp_idx[l])
            self.pixels[l].dense(ti.l, max_fp_per_cell).place(self.fp_idx[l])

        # ---- auxiliary points for the least squares (offsets in units of radius) ----
        if aux_offsets is None:
            aux_offsets = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        aux_offsets.append((0.2 * i, 0.2 * j, 0.2 * k))
        aux_offsets = np.asarray(aux_offsets, dtype=np.float32).reshape(-1, 3)
        self.n_aux = aux_offsets.shape[0]
        self.aux_offs = ti.Vector.field(3, ti.f32, shape=max(self.n_aux, 1))
        self.aux_offs.from_numpy(aux_offsets)
        self.aux_D = ti.field(ti.f32, shape=(max_fp, max(self.n_aux, 1)))
        self.aux_ok = ti.field(ti.i8, shape=(max_fp, max(self.n_aux, 1)))

        # far field query scratch
        self.far_count = ti.field(ti.i32, shape=())

    # ------------------------------------------------------------------ helpers
    @ti.func
    def weight(self, t):
        """Partition-of-unity kernel on normalised distance t = d / r in [0, 1)."""
        w = 0.0
        if ti.static(self.weight_type == 0):
            w = (1.0 - t) * (1.0 - t)
        elif ti.static(self.weight_type == 2):              # Wendland C2: (1-t)^4 (4t+1)
            u = 1.0 - t
            w = u * u * u * u * (4.0 * t + 1.0)
        elif ti.static(self.weight_type == 3):              # (1-t)^4: sharper, C1 at the boundary
            u = 1.0 - t
            w = u * u * u * u
        elif ti.static(self.weight_type == 4):              # smoothstep-like C1 bump: 1 - 3t^2 + 2t^3
            w = 1.0 - 3.0 * t * t + 2.0 * t * t * t
        else:                                               # 1: Ohtake et al. piecewise quadratic B-spline
            if t < 1.0 / 3.0:
                w = 1.0 - 3.0 * t * t
            else:
                w = 1.5 * (1.0 - t) * (1.0 - t)
        return w

    @ti.func
    def dweight(self, t, r):
        """d w / d d  where d = t * r."""
        dw = 0.0
        if ti.static(self.weight_type == 0):
            dw = -2.0 * (1.0 - t) / r
        elif ti.static(self.weight_type == 2):
            u = 1.0 - t
            dw = -20.0 * t * u * u * u / r
        elif ti.static(self.weight_type == 3):
            u = 1.0 - t
            dw = -4.0 * u * u * u / r
        elif ti.static(self.weight_type == 4):
            dw = (-6.0 * t + 6.0 * t * t) / r
        else:
            if t < 1.0 / 3.0:
                dw = -6.0 * t / r
            else:
                dw = -3.0 * (1.0 - t) / r
        return dw

    @ti.func
    def cell_of(self, p, l: ti.template()):
        return ti.cast(ti.floor((p + 1.0) / self.cell_size[l]), ti.i32)

    @ti.func
    def in_domain(self, p):
        return (-1.0 <= p[0] < 1.0) and (-1.0 <= p[1] < 1.0) and (-1.0 <= p[2] < 1.0)

    @ti.func
    def in_cells(self, c, l: ti.template()):
        r = self.res[l]
        return (0 <= c[0] < r) and (0 <= c[1] < r) and (0 <= c[2] < r)

    @ti.func
    def is_alloc(self, c, l: ti.template()):
        b = ti.static(self.block_size[l])
        return ti.is_active(self.blocks[l], [c[0] // b, c[1] // b, c[2] // b])

    @ti.func
    def cell_center(self, c, l: ti.template()):
        h = self.cell_size[l]
        return ti.Vector([(c[0] + 0.5) * h - 1.0, (c[1] + 0.5) * h - 1.0, (c[2] + 0.5) * h - 1.0])

    # ------------------------------------------------------------------ sample particles
    def set_sample_particles(self, pos, nrm, cur):
        """pos/nrm/cur: torch cuda tensors (N,3),(N,3),(N,)"""
        n = pos.shape[0]
        assert n <= self.max_sp, f"{n} > max_sp {self.max_sp}"
        self.sp_n[None] = n
        self._fill_sp(n, pos.contiguous().float(), nrm.contiguous().float(), cur.contiguous().float())

    @ti.kernel
    def _fill_sp(self, n: ti.i32, pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 nrm: ti.types.ndarray(dtype=ti.f32, ndim=2), cur: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            self.sp_pos[i] = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            nn = ti.Vector([nrm[i, 0], nrm[i, 1], nrm[i, 2]])
            self.sp_nrm[i] = nn / ti.max(nn.norm(), 1e-12)
            self.sp_cur[i] = cur[i]

    # ------------------------------------------------------------------ grid
    def clear_grid(self):
        for l in range(self.num_levels):
            self.blocks[l].deactivate_all()

    def set_thresholds(self, th):
        th = np.asarray(th, dtype=np.float32)
        assert th.shape[0] == self.num_levels
        self.thresholds.from_numpy(th)

    def bin_sample_particles(self):
        for l in range(self.num_levels):
            self._bin_sp(l)

    @ti.kernel
    def _bin_sp(self, l: ti.template()):
        for i in range(self.sp_n[None]):
            p = self.sp_pos[i]
            if self.in_domain(p):
                c = self.cell_of(p, l)
                k = ti.atomic_add(self.sp_num[l][c], 1)
                if k < self.max_sp_per_cell:
                    self.sp_idx[l][c, k] = i
                if l == 0 or self.sp_cur[i] > self.thresholds[l]:
                    self.act[l][c] = ti.cast(1, ti.i8)
                    for offs in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        cc = c + offs
                        if self.in_cells(cc, l):
                            self.act[l][cc] = ti.cast(1, ti.i8)

    @ti.kernel
    def _apply_refine(self, l: ti.template()):
        """Cells marked for refinement at level l-1 activate their children at level l (and the
        children's neighbours), mimicking the curvature-activation footprint."""
        for I in ti.grouped(self.refine[l - 1]):
            if self.refine[l - 1][I] == 1:
                for offs in ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2))):
                    c = I * 2 + offs
                    if self.sp_num[l][c] > 0:
                        self.act[l][c] = ti.cast(1, ti.i8)
                        for o2 in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                            cc = c + o2
                            if self.in_cells(cc, l):
                                self.act[l][cc] = ti.cast(1, ti.i8)

    def apply_refine_marks(self):
        for l in range(1, self.num_levels):
            self._apply_refine(l)

    # ------------------------------------------------------------------ feature particle generation
    def generate_feature_particles(self):
        """Coarse-to-fine coverage: fps at the finest activated cells, coarser cells only where
        children are not all covered."""
        self.fp_n[None] = 0
        L = self.num_levels
        self._gen_finest(L - 1)
        for l in range(L - 2, -1, -1):
            self._gen_level(l)
        self.allocate_fp()
        return self.fp_n[None]

    @ti.func
    def _new_fp(self, c, l: ti.template()):
        idx = ti.atomic_add(self.fp_n[None], 1)
        if idx < self.max_fp:
            self.fp_pos[idx] = self.cell_center(c, l)
            self.fp_layer[idx] = l
            self.fp_radius[idx] = self.alpha * self.cell_size[l]
            self.fp_nrm[idx] = ti.Vector([0.0, 1.0, 0.0])
            self.coef[idx] = ti.Vector([0.0] * 9 + [1.0])
            self.fp_err[idx] = 0.0
            self.fp_nb_num[idx] = 0

    @ti.kernel
    def _gen_finest(self, l: ti.template()):
        for I in ti.grouped(self.sp_num[l]):
            if self.sp_num[l][I] > 0 and self.act[l][I] == 1:
                self._new_fp(I, l)
                self.cov[l][I] = ti.cast(1, ti.i8)

    @ti.kernel
    def _gen_level(self, l: ti.template()):
        for I in ti.grouped(self.sp_num[l]):
            if self.sp_num[l][I] > 0:
                all_cov = True
                for offs in ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2))):
                    c = I * 2 + offs
                    if self.sp_num[l + 1][c] > 0 and self.cov[l + 1][c] == 0:
                        all_cov = False
                if all_cov:
                    self.cov[l][I] = ti.cast(1, ti.i8)
                elif self.act[l][I] == 1:
                    self._new_fp(I, l)
                    self.cov[l][I] = ti.cast(1, ti.i8)
                else:
                    self.cov[l][I] = ti.cast(0, ti.i8)

    def allocate_fp(self):
        for l in range(self.num_levels):
            self._clear_fp_level(l)
        for l in range(self.num_levels):
            self._alloc_fp(l)

    @ti.kernel
    def _clear_fp_level(self, l: ti.template()):
        for I in ti.grouped(self.fp_num[l]):
            self.fp_num[l][I] = 0

    @ti.kernel
    def _alloc_fp(self, l: ti.template()):
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            if self.fp_layer[i] == l and self.fp_radius[i] > 0.0 and self.in_domain(self.fp_pos[i]):
                c = self.cell_of(self.fp_pos[i], l)
                k = ti.atomic_add(self.fp_num[l][c], 1)
                if k < self.max_fp_per_cell:
                    self.fp_idx[l][c, k] = i
                else:
                    self.fp_num[l][c] = self.max_fp_per_cell

    def add_feature_particles(self, pos, layer, radius):
        """Append new feature particles (torch cuda tensors)."""
        n0 = self.fp_n[None]
        n = pos.shape[0]
        assert n0 + n <= self.max_fp
        self._append_fp(n0, n, pos.contiguous().float(), layer.contiguous().int(), radius.contiguous().float())
        self.fp_n[None] = n0 + n

    @ti.kernel
    def _append_fp(self, n0: ti.i32, n: ti.i32, pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   layer: ti.types.ndarray(dtype=ti.i32, ndim=1), radius: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            j = n0 + i
            self.fp_pos[j] = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            self.fp_layer[j] = layer[i]
            self.fp_radius[j] = radius[i]
            self.fp_nrm[j] = ti.Vector([0.0, 1.0, 0.0])
            self.coef[j] = ti.Vector([0.0] * 9 + [1.0])
            self.fp_err[j] = 0.0
            self.fp_nb_num[j] = 0

    # ------------------------------------------------------------------ neighbour search + normals
    def collect_neighbors(self):
        for l in range(self.num_levels):
            self._collect_nb(l)
        self._fp_normals()

    @ti.kernel
    def _collect_nb(self, l: ti.template()):
        ext = ti.static(self.search_extent)
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            if self.fp_layer[i] == l and self.fp_radius[i] > 0.0:
                p = self.fp_pos[i]
                r = self.fp_radius[i]
                c = self.cell_of(p, l)
                cnt = 0
                for offs in ti.grouped(ti.ndrange((-ext, ext + 1), (-ext, ext + 1), (-ext, ext + 1))):
                    cc = c + offs
                    if self.in_cells(cc, l):
                        if self.is_alloc(cc, l):
                            m = ti.min(self.sp_num[l][cc], self.max_sp_per_cell)
                            for k in range(m):
                                j = self.sp_idx[l][cc, k]
                                if (self.sp_pos[j] - p).norm() < r:
                                    if cnt < self.max_nb:
                                        self.fp_nb_idx[i, cnt] = j
                                        cnt += 1
                                    else:
                                        rid = ti.cast(ti.random(ti.f32) * self.max_nb, ti.i32)
                                        self.fp_nb_idx[i, ti.min(rid, self.max_nb - 1)] = j
                self.fp_nb_num[i] = cnt

    @ti.kernel
    def _fp_normals(self):
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            acc = ti.Vector([0.0, 0.0, 0.0])
            r = self.fp_radius[i]
            for k in range(self.fp_nb_num[i]):
                j = self.fp_nb_idx[i, k]
                d = (self.sp_pos[j] - self.fp_pos[i]).norm()
                w = ((r - d) / (r * ti.max(d, 1e-6))) ** 2
                acc += self.sp_nrm[j] * w
            if acc.norm() > 1e-12:
                self.fp_nrm[i] = acc / acc.norm()

    # ------------------------------------------------------------------ least squares assembly
    @ti.kernel
    def assemble(self, A: ti.types.ndarray(dtype=ti.f32, ndim=3), B: ti.types.ndarray(dtype=ti.f32, ndim=2),
                 aux_k: ti.i32, aux_w: ti.f32, aux_sign_check: ti.i32, sp_w_scale: ti.f32):
        """Normal equations in radius-normalised local coordinates x' = (x - c) / r.
        Sample particles: weight theta^2 / sum(theta) with f = 0.
        Auxiliary points q = c + off * r: weight aux_w^2 with f = mean plane distance to aux_k
        nearest sample neighbours."""
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            r = self.fp_radius[i]
            c = self.fp_pos[i]
            nb = self.fp_nb_num[i]
            M = ti.Matrix.zero(ti.f32, 10, 10)
            rhs = ti.Vector.zero(ti.f32, 10)
            wsum = 0.0
            for k in range(nb):
                x = (self.sp_pos[self.fp_nb_idx[i, k]] - c) / r
                th = self.weight(ti.min(x.norm(), 1.0))
                bv = basis10(x) * th
                M += bv.outer_product(bv)
                wsum += th
            if wsum > 0.0:
                M *= (sp_w_scale / wsum)
            # auxiliary points
            for a in range(self.n_aux):
                q = c + self.aux_offs[a] * r
                near_idx = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1])
                near_d = ti.Vector([1e9] * 8)
                for k in range(nb):
                    j = self.fp_nb_idx[i, k]
                    d = (q - self.sp_pos[j]).norm()
                    # insertion into sorted (ascending) list of size aux_k
                    if d < near_d[aux_k - 1]:
                        pos = aux_k - 1
                        while pos > 0 and near_d[pos - 1] > d:
                            near_d[pos] = near_d[pos - 1]
                            near_idx[pos] = near_idx[pos - 1]
                            pos -= 1
                        near_d[pos] = d
                        near_idx[pos] = j
                tot = 0.0
                cnt = 0
                npos = 0
                nneg = 0
                for k in range(aux_k):
                    j = near_idx[k]
                    if j >= 0:
                        dist = ti.math.dot(q - self.sp_pos[j], self.sp_nrm[j])
                        tot += dist
                        cnt += 1
                        if dist > 0.0:
                            npos += 1
                        else:
                            nneg += 1
                ok = cnt > 0
                if aux_sign_check == 1 and npos > 0 and nneg > 0:
                    ok = False
                self.aux_ok[i, a] = ti.cast(0, ti.i8)
                self.aux_D[i, a] = 0.0
                if ok:
                    f = tot / cnt
                    self.aux_ok[i, a] = ti.cast(1, ti.i8)
                    self.aux_D[i, a] = f
                    bv = basis10((q - c) / r) * aux_w
                    M += bv.outer_product(bv)
                    rhs += bv * (f * aux_w)
            for u in ti.static(range(10)):
                B[i, u] = rhs[u]
                for v in ti.static(range(10)):
                    A[i, u, v] = M[u, v]

    @ti.kernel
    def set_coef(self, beta: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        """beta is in normalised coordinates; convert to world units."""
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            r = self.fp_radius[i]
            if r > 0.0:
                inv_r = 1.0 / r
                cvec = ti.Vector.zero(ti.f32, 10)
                for u in ti.static(range(6)):
                    cvec[u] = beta[i, u] * inv_r * inv_r
                for u in ti.static(range(6, 9)):
                    cvec[u] = beta[i, u] * inv_r
                cvec[9] = beta[i, 9]
                self.coef[i] = cvec

    @ti.kernel
    def compute_fit_error(self, use_aux: ti.i32, aux_scale: ti.f32):
        """Max over neighbours of |f_i(p)| / |grad f_i(p)| / r, and (optionally) max over valid
        auxiliary points of |f_i(q) - D(q)| / r scaled by aux_scale."""
        n = ti.min(self.fp_n[None], self.max_fp)
        for i in range(n):
            r = self.fp_radius[i]
            c = self.fp_pos[i]
            err = 0.0
            if r > 0.0:
                for k in range(self.fp_nb_num[i]):
                    j = self.fp_nb_idx[i, k]
                    x = self.sp_pos[j] - c
                    f = quad_value(self.coef[i], x)
                    g = quad_grad(self.coef[i], x)
                    d = ti.abs(f) / ti.max(g.norm(), 1e-8)
                    err = ti.max(err, d / r)
                if use_aux == 1:
                    for a in range(self.n_aux):
                        if self.aux_ok[i, a] == 1:
                            x = self.aux_offs[a] * r
                            f = quad_value(self.coef[i], x)
                            err = ti.max(err, aux_scale * ti.abs(f - self.aux_D[i, a]) / r)
            self.fp_err[i] = err

    def fit(self, aux_k=6, aux_w=1.0 / 9.0, aux_sign_check=False, sp_w_scale=1.0, ridge=1e-7, device="cuda"):
        n = min(self.fp_n[None], self.max_fp)
        A = torch.zeros((n, 10, 10), dtype=torch.float32, device=device)
        B = torch.zeros((n, 10), dtype=torch.float32, device=device)
        self.assemble(A, B, int(aux_k), float(aux_w), int(aux_sign_check), float(sp_w_scale))
        A64 = A.double()
        B64 = B.double()
        tr = torch.diagonal(A64, dim1=1, dim2=2).sum(-1).clamp_min(1e-12) / 10.0
        eye = torch.eye(10, dtype=torch.float64, device=device)
        A64 = A64 + (ridge * tr)[:, None, None] * eye
        try:
            beta = torch.linalg.solve(A64, B64)
        except Exception:
            beta = torch.linalg.lstsq(A64.cpu(), B64.cpu(), driver="gelsd").solution.to(device)
        bad = ~torch.isfinite(beta).all(dim=1)
        if bad.any():
            sol = torch.linalg.lstsq(A64[bad].cpu(), B64[bad].cpu(), driver="gelsd").solution
            beta[bad] = sol.to(device)
            bad2 = ~torch.isfinite(beta).all(dim=1)
            beta[bad2] = 0.0
            beta[bad2, 9] = 1.0
        # particles with no neighbours: disable
        nbn = self.fp_nb_num.to_torch(device=device)[:n]
        empty = nbn == 0
        beta[empty] = 0.0
        beta[empty, 9] = 1.0
        self.set_coef(beta.float().contiguous())
        if empty.any():
            self._disable(empty.int().contiguous(), n)
        return int(empty.sum().item())

    @ti.kernel
    def _disable(self, mask: ti.types.ndarray(dtype=ti.i32, ndim=1), n: ti.i32):
        for i in range(n):
            if mask[i] == 1:
                self.fp_radius[i] = 0.0

    # ------------------------------------------------------------------ Newton projection
    @ti.kernel
    def project(self, max_frac: ti.f32) -> ti.f32:
        """One Newton step c <- c - f(c) grad/|grad| using each particle's own quadric.
        Returns the mean |step| / radius."""
        n = ti.min(self.fp_n[None], self.max_fp)
        tot = 0.0
        cnt = 0
        for i in range(n):
            r = self.fp_radius[i]
            if r > 0.0 and self.fp_nb_num[i] > 0:
                c = self.coef[i]
                f0 = c[9]
                g = ti.Vector([c[6], c[7], c[8]])
                gn = g.norm()
                if gn > 1e-8:
                    step = f0 / gn
                    lim = max_frac * r
                    step = ti.min(ti.max(step, -lim), lim)
                    newp = self.fp_pos[i] - step * g / gn
                    if self.in_domain(newp):
                        self.fp_pos[i] = newp
                    tot += ti.abs(step) / r
                    cnt += 1
        return tot / ti.max(cnt, 1)

    # ------------------------------------------------------------------ evaluation
    @ti.kernel
    def _eval_level(self, l: ti.template(), n: ti.i32, q: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    F: ti.types.ndarray(dtype=ti.f32, ndim=1), W: ti.types.ndarray(dtype=ti.f32, ndim=1),
                    G: ti.types.ndarray(dtype=ti.f32, ndim=2), GW: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    cnt: ti.types.ndarray(dtype=ti.i32, ndim=1), gain: ti.f32):
        ext = ti.static(self.search_extent)
        for i in range(n):
            p = ti.Vector([q[i, 0], q[i, 1], q[i, 2]])
            c = self.cell_of(p, l)
            Fi = 0.0
            Wi = 0.0
            Gi = ti.Vector([0.0, 0.0, 0.0])
            GWi = ti.Vector([0.0, 0.0, 0.0])
            ci = 0
            for offs in ti.grouped(ti.ndrange((-ext, ext + 1), (-ext, ext + 1), (-ext, ext + 1))):
                cc = c + offs
                if self.in_cells(cc, l):
                    if self.is_alloc(cc, l):
                        m = ti.min(self.fp_num[l][cc], self.max_fp_per_cell)
                        for k in range(m):
                            j = self.fp_idx[l][cc, k]
                            r = self.fp_radius[j]
                            dx = p - self.fp_pos[j]
                            d = dx.norm()
                            if d < r and r > 0.0:
                                t = d / r
                                g_l = gain ** self.fp_layer[j]
                                w = self.weight(t) * g_l
                                dwdd = self.dweight(t, r) * g_l
                                gw = dwdd * dx / ti.max(d, 1e-12)
                                f = quad_value(self.coef[j], dx)
                                gf = quad_grad(self.coef[j], dx)
                                Fi += w * f
                                Wi += w
                                Gi += w * gf + f * gw
                                GWi += gw
                                ci += 1
            if ci > 0:
                F[i] += Fi
                W[i] += Wi
                G[i, 0] += Gi[0]
                G[i, 1] += Gi[1]
                G[i, 2] += Gi[2]
                GW[i, 0] += GWi[0]
                GW[i, 1] += GWi[1]
                GW[i, 2] += GWi[2]
                cnt[i] += ci

    # ---- far field: constant sdf in empty cells (sign far from the surface)
    def compute_far_field(self, sp_sub_pos, sp_sub_nrm, knn_fn, max_cells=8_000_000, device="cuda",
                          winding_pos=None, winding_nrm=None, winding_area=None):
        """Fill far_sdf for empty cells at levels < far_levels whose parent contains particles
        (level 0: all empty cells). Distance = plane distance to the nearest of the sub-sampled
        sample particles (knn_fn(query, ref) -> idx (Q,))."""
        for l in range(self.num_levels):
            self._clear_far(l)
        centers = torch.zeros((max_cells, 3), dtype=torch.float32, device=device)
        levels = torch.zeros((max_cells,), dtype=torch.int32, device=device)
        self.far_count[None] = 0
        for l in range(self.far_levels):
            self._collect_far(l, centers, levels, max_cells)
        m = min(self.far_count[None], max_cells)
        if m == 0:
            return 0
        idx = knn_fn(centers[:m], sp_sub_pos)
        pp = sp_sub_pos[idx]
        nn = sp_sub_nrm[idx]
        d = ((centers[:m] - pp) * nn).sum(-1)
        if winding_pos is not None:
            K = winding_pos.shape[0]
            if K > 2 * self.far_stage1 and self.far_stage1 > 0:
                # two-stage: a coarse subset decides the sign of the unambiguous cells, the full set is only evaluated where
                # the coarse winding number is close to 0.5 (identical result at ~1/4 of the cost)
                sel = torch.randperm(K, device=centers.device)[:self.far_stage1]
                w = point_winding_number(centers[:m], winding_pos[sel], winding_nrm[sel], winding_area * K / self.far_stage1)
                amb = (w - 0.5).abs() < self.far_stage1_band
                if bool(amb.any()):
                    w[amb] = point_winding_number(centers[:m][amb], winding_pos, winding_nrm, winding_area)
            else:
                w = point_winding_number(centers[:m], winding_pos, winding_nrm, winding_area)
            sign = torch.where(w > 0.5, -1.0, 1.0)
            d = sign * d.abs().clamp_min(1e-4)
        for l in range(self.far_levels):
            self._write_far(l, m, centers, levels, d.contiguous().float())
        return m

    @ti.kernel
    def _clear_far(self, l: ti.template()):
        for I in ti.grouped(self.far_ok[l]):
            self.far_ok[l][I] = ti.cast(0, ti.i8)
            self.far_sdf[l][I] = 0.0

    @ti.kernel
    def _collect_far(self, l: ti.template(), centers: ti.types.ndarray(dtype=ti.f32, ndim=2),
                     levels: ti.types.ndarray(dtype=ti.i32, ndim=1), max_cells: ti.i32):
        r = self.res[l]
        for i, j, k in ti.ndrange(r, r, r):
            c = ti.Vector([i, j, k])
            empty = self.sp_num[l][c] == 0
            take = False
            if empty:
                if ti.static(l == 0):
                    take = True
                else:
                    pc = ti.Vector([i // 2, j // 2, k // 2])
                    if self.sp_num[l - 1][pc] > 0:
                        take = True
            if take:
                idx = ti.atomic_add(self.far_count[None], 1)
                if idx < max_cells:
                    cen = self.cell_center(c, l)
                    centers[idx, 0] = cen[0]
                    centers[idx, 1] = cen[1]
                    centers[idx, 2] = cen[2]
                    levels[idx] = l

    @ti.kernel
    def _write_far(self, l: ti.template(), m: ti.i32, centers: ti.types.ndarray(dtype=ti.f32, ndim=2),
                   levels: ti.types.ndarray(dtype=ti.i32, ndim=1), d: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(m):
            if levels[i] == l:
                p = ti.Vector([centers[i, 0], centers[i, 1], centers[i, 2]])
                c = self.cell_of(p, l)
                self.far_sdf[l][c] = d[i]
                self.far_ok[l][c] = ti.cast(1, ti.i8)

    @ti.kernel
    def _far_lookup(self, n: ti.i32, q: ti.types.ndarray(dtype=ti.f32, ndim=2),
                    cnt: ti.types.ndarray(dtype=ti.i32, ndim=1), out: ti.types.ndarray(dtype=ti.f32, ndim=1),
                    ok: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(n):
            if cnt[i] == 0:
                p = ti.Vector([q[i, 0], q[i, 1], q[i, 2]])
                found = 0
                val = 0.0
                if self.in_domain(p):
                    for l in ti.static(range(self.num_levels)):
                        if found == 0:
                            c = self.cell_of(p, l)
                            if self.is_alloc(c, l):
                                if self.far_ok[l][c] == 1:
                                    val = self.far_sdf[l][c]
                                    found = 1
                out[i] = val
                ok[i] = found

    @ti.kernel
    def _fallback(self, n: ti.i32, q: ti.types.ndarray(dtype=ti.f32, ndim=2),
                  need: ti.types.ndarray(dtype=ti.i32, ndim=1), out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        """Plane distance to the nearest stored sample particle, searching the 27-cell neighbourhood
        from the finest level to the coarsest (first level with any stored particle wins)."""
        for i in range(n):
            if need[i] == 1:
                p = ti.Vector([q[i, 0], q[i, 1], q[i, 2]])
                best = 1e9
                val = 1.0
                if self.in_domain(p):
                    for l in ti.static(range(self.num_levels - 1, -1, -1)):
                        if best > 1e8:
                            c = self.cell_of(p, l)
                            for offs in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                                cc = c + offs
                                if self.in_cells(cc, l):
                                    if self.is_alloc(cc, l):
                                        m = ti.min(self.sp_num[l][cc], self.max_sp_per_cell)
                                        for k in range(m):
                                            j = self.sp_idx[l][cc, k]
                                            d = (p - self.sp_pos[j]).norm()
                                            if d < best:
                                                best = d
                                                val = ti.math.dot(p - self.sp_pos[j], self.sp_nrm[j])
                out[i] = val

    def eval(self, P, gain=None, return_grad=True, use_far=True, device="cuda"):
        """P: torch cuda (N,3). Returns f (N,), grad (N,3), covered (N,) bool."""
        gain = self.level_gain if gain is None else gain
        P = P.contiguous().float()
        n = P.shape[0]
        F = torch.zeros(n, dtype=torch.float32, device=device)
        W = torch.zeros(n, dtype=torch.float32, device=device)
        G = torch.zeros((n, 3), dtype=torch.float32, device=device)
        GW = torch.zeros((n, 3), dtype=torch.float32, device=device)
        cnt = torch.zeros(n, dtype=torch.int32, device=device)
        for l in range(self.num_levels):
            self._eval_level(l, n, P, F, W, G, GW, cnt, float(gain))
        covered = cnt > 0
        Wc = W.clamp_min(1e-30)
        f = F / Wc
        grad = None
        if return_grad:
            grad = (G * W[:, None] - F[:, None] * GW) / (Wc * Wc)[:, None]
            grad[~covered] = 0.0
        if use_far:
            far = torch.zeros(n, dtype=torch.float32, device=device)
            ok = torch.zeros(n, dtype=torch.int32, device=device)
            self._far_lookup(n, P, cnt, far, ok)
            f = torch.where(covered, f, far)
            need = ((~covered) & (ok == 0)).int().contiguous()
            if need.any():
                fb = torch.zeros(n, dtype=torch.float32, device=device)
                self._fallback(n, P, need, fb)
                f = torch.where(need.bool(), fb, f)
        return f, grad, covered

    # ------------------------------------------------------------------ hole fixing support
    @ti.kernel
    def _finest_fp_cell(self, n: ti.i32, q: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        out_level: ti.types.ndarray(dtype=ti.i32, ndim=1), out_cell: ti.types.ndarray(dtype=ti.i64, ndim=1)):
        """For each point, the finest level whose containing cell holds a feature particle
        (fallback: level 0 cell)."""
        for i in range(n):
            p = ti.Vector([q[i, 0], q[i, 1], q[i, 2]])
            lev = 0
            cid = ti.cast(0, ti.i64)
            found = 0
            for l in ti.static(range(self.num_levels - 1, -1, -1)):
                if found == 0:
                    c = self.cell_of(p, l)
                    if self.in_cells(c, l):
                        take = False
                        if ti.static(l == 0):
                            take = True
                        else:
                            if self.is_alloc(c, l):
                                if self.fp_num[l][c] > 0:
                                    take = True
                        if take:
                            lev = l
                            r = ti.cast(self.res[l], ti.i64)
                            cid = (ti.cast(c[0], ti.i64) * r + ti.cast(c[1], ti.i64)) * r + ti.cast(c[2], ti.i64)
                            found = 1
            out_level[i] = lev
            out_cell[i] = cid

    def fix_holes(self, radius_scale=1.0, device="cuda", batch=2 ** 22):
        """Add feature particles at the mean position of uncovered sample particles, grouped by
        the finest fp-holding cell. Returns number of uncovered sps before fixing and number added."""
        n = self.sp_n[None]
        sp = self.sp_pos.to_torch(device=device)[:n]
        unc = []
        for s in range(0, n, batch):
            _, _, cov = self.eval(sp[s:s + batch], return_grad=False, use_far=False)
            unc.append(~cov)
        unc = torch.cat(unc)
        n_unc = int(unc.sum().item())
        if n_unc == 0:
            return 0, 0
        q = sp[unc].contiguous()
        m = q.shape[0]
        lev = torch.zeros(m, dtype=torch.int32, device=device)
        cid = torch.zeros(m, dtype=torch.int64, device=device)
        self._finest_fp_cell(m, q, lev, cid)
        key = lev.long() * (self.res[-1] ** 3 + 1) + cid
        uniq, inv = torch.unique(key, return_inverse=True)
        k = uniq.shape[0]
        sums = torch.zeros((k, 3), dtype=torch.float32, device=device).index_add_(0, inv, q)
        counts = torch.zeros(k, dtype=torch.float32, device=device).index_add_(0, inv, torch.ones(m, device=device))
        mean = sums / counts[:, None]
        lev_u = torch.zeros(k, dtype=torch.int32, device=device)
        lev_u.scatter_(0, inv, lev)
        cs = torch.tensor(self.cell_size, dtype=torch.float32, device=device)
        rad = radius_scale * self.alpha * cs[lev_u.long()]
        k_add = min(k, self.max_fp - self.fp_n[None])
        self.add_feature_particles(mean[:k_add], lev_u[:k_add], rad[:k_add])
        return n_unc, k_add

    # ------------------------------------------------------------------ error-based refinement
    @ti.kernel
    def _mark_refine(self, l: ti.template(), tol: ti.f32) -> ti.i32:
        """Mark cells at level l whose fp has fit error > tol (so that children get activated)."""
        n = ti.min(self.fp_n[None], self.max_fp)
        cnt = 0
        for i in range(n):
            if self.fp_layer[i] == l and self.fp_radius[i] > 0.0 and self.fp_err[i] > tol:
                p = self.fp_pos[i]
                if self.in_domain(p):
                    c = self.cell_of(p, l)
                    self.refine[l][c] = ti.cast(1, ti.i8)
                    cnt += 1
        return cnt

    def mark_refine(self, tol):
        total = 0
        for l in range(self.num_levels - 1):
            total += self._mark_refine(l, float(tol))
        return total

    @ti.kernel
    def _clear_refine(self, l: ti.template()):
        for I in ti.grouped(self.refine[l]):
            self.refine[l][I] = ti.cast(0, ti.i8)

    def clear_refine(self):
        for l in range(self.num_levels):
            self._clear_refine(l)

    # ------------------------------------------------------------------ sample particle neighbourhood ops
    def bin_only(self, pos, nrm, cur, thresholds):
        """Re-bin sample particles (positions may have changed) without touching feature particles."""
        self.set_sample_particles(pos, nrm, cur)
        self.clear_grid()
        self.set_thresholds(thresholds)
        self.bin_sample_particles()

    @ti.kernel
    def sp_neighborhood(self, l: ti.template(), K: ti.i32, coll_dist: ti.f32, coll_dot: ti.f32,
                        vel: ti.types.ndarray(dtype=ti.f32, ndim=2), out_nrm: ti.types.ndarray(dtype=ti.f32, ndim=2),
                        out_vel: ti.types.ndarray(dtype=ti.f32, ndim=2), flag: ti.types.ndarray(dtype=ti.i32, ndim=1),
                        do_normals: ti.i32, do_vel: ti.i32, vel_radius: ti.f32,
                        out_cur: ti.types.ndarray(dtype=ti.f32, ndim=1), do_cur: ti.i32):
        """For each sample particle: K nearest stored neighbours at level l (27 cells); PCA normal
        (oriented with the current normal), neighbour-averaged velocity, collision flag (a neighbour
        closer than coll_dist with normal dot < coll_dot)."""
        for i in range(self.sp_n[None]):
            p = self.sp_pos[i]
            c = self.cell_of(p, l)
            nd = ti.Vector([1e9] * 32)
            ni = ti.Vector([-1] * 32)
            kk = ti.min(K, 32)
            for offs in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                cc = c + offs
                if self.in_cells(cc, l):
                    if self.is_alloc(cc, l):
                        m = ti.min(self.sp_num[l][cc], self.max_sp_per_cell)
                        for k in range(m):
                            j = self.sp_idx[l][cc, k]
                            if j != i:
                                d = (self.sp_pos[j] - p).norm()
                                if d < nd[kk - 1]:
                                    pos = kk - 1
                                    while pos > 0 and nd[pos - 1] > d:
                                        nd[pos] = nd[pos - 1]
                                        ni[pos] = ni[pos - 1]
                                        pos -= 1
                                    nd[pos] = d
                                    ni[pos] = j
            n_old = self.sp_nrm[i]
            if do_normals == 1:
                mean = p
                cnt = 1.0
                for k in range(kk):
                    if ni[k] >= 0:
                        mean += self.sp_pos[ni[k]]
                        cnt += 1.0
                mean /= cnt
                H = ti.Matrix.zero(ti.f32, 3, 3)
                x = p - mean
                H += x.outer_product(x)
                for k in range(kk):
                    if ni[k] >= 0:
                        x = self.sp_pos[ni[k]] - mean
                        H += x.outer_product(x)
                nn = n_old
                if cnt >= 4.0:
                    ev, evec = ti.sym_eig(H)
                    idx = 0
                    if ev[1] < ev[idx]:
                        idx = 1
                    if ev[2] < ev[idx]:
                        idx = 2
                    nn = ti.Vector([evec[0, idx], evec[1, idx], evec[2, idx]])
                    if nn.norm() > 1e-12:
                        nn = nn / nn.norm()
                        if ti.math.dot(nn, n_old) < 0.0:
                            nn = -nn
                    else:
                        nn = n_old
                out_nrm[i, 0] = nn[0]
                out_nrm[i, 1] = nn[1]
                out_nrm[i, 2] = nn[2]
            if do_vel == 1:
                v = ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]])
                w = 1.0
                for k in range(kk):
                    j = ni[k]
                    if j >= 0 and nd[k] < vel_radius:
                        wk = (1.0 - nd[k] / vel_radius) ** 2
                        v += ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) * wk
                        w += wk
                v /= w
                out_vel[i, 0] = v[0]
                out_vel[i, 1] = v[1]
                out_vel[i, 2] = v[2]
            elif do_vel == 2:
                # partition-of-unity style: all stored neighbours within vel_radius (27 cells at level l)
                v = ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]])
                w = 1.0
                for offs in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    cc = c + offs
                    if self.in_cells(cc, l):
                        if self.is_alloc(cc, l):
                            m = ti.min(self.sp_num[l][cc], self.max_sp_per_cell)
                            for k in range(m):
                                j = self.sp_idx[l][cc, k]
                                d = (self.sp_pos[j] - p).norm()
                                if j != i and d < vel_radius:
                                    wk = (1.0 - d / vel_radius) ** 2
                                    v += ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) * wk
                                    w += wk
                v /= w
                out_vel[i, 0] = v[0]
                out_vel[i, 1] = v[1]
                out_vel[i, 2] = v[2]
            f = 0
            for k in range(kk):
                j = ni[k]
                if j >= 0 and nd[k] < coll_dist:
                    if ti.math.dot(n_old, self.sp_nrm[j]) < coll_dot:
                        f = 1
            flag[i] = f
            if do_cur == 1:
                # curvature ~ 2 * mean |n_j - n_i| / |p_j - p_i| over neighbours (2|H| scale for a sphere)
                acc = 0.0
                cnt = 0.0
                for k in range(kk):
                    j = ni[k]
                    if j >= 0 and nd[k] > 1e-7:
                        acc += (self.sp_nrm[j] - n_old).norm() / nd[k]
                        cnt += 1.0
                out_cur[i] = 2.0 * acc / ti.max(cnt, 1.0)

    @ti.kernel
    def _sp_thickness(self, l: ti.template(), dot_th: ti.f32, out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        """Distance from each sample particle to the nearest stored particle with an opposing normal
        (dot < dot_th) within the 27-cell neighbourhood at level l (1e9 if none)."""
        for i in range(self.sp_n[None]):
            p = self.sp_pos[i]
            n = self.sp_nrm[i]
            c = self.cell_of(p, l)
            best = 1e9
            for offs in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                cc = c + offs
                if self.in_cells(cc, l):
                    if self.is_alloc(cc, l):
                        m = ti.min(self.sp_num[l][cc], self.max_sp_per_cell)
                        for k in range(m):
                            j = self.sp_idx[l][cc, k]
                            if ti.math.dot(n, self.sp_nrm[j]) < dot_th:
                                d = (self.sp_pos[j] - p).norm()
                                # only count particles roughly across the normal direction
                                if d < best and ti.abs(ti.math.dot(self.sp_pos[j] - p, n)) > 0.5 * d:
                                    best = d
            out[i] = best

    def sp_thickness(self, level, dot_th=-0.5, device="cuda"):
        n = self.sp_n[None]
        out = torch.full((n,), 1e9, device=device)
        self._sp_thickness(level, float(dot_th), out)
        return out

    def sp_curvature_pca(self, K=16, level=None, device="cuda"):
        """Neighbour-based curvature estimate for the currently binned sample particles."""
        n = self.sp_n[None]
        l = self.num_levels - 1 if level is None else level
        out_n = torch.zeros((n, 3), device=device)
        out_v = torch.zeros((1, 3), device=device)
        flag = torch.zeros(n, dtype=torch.int32, device=device)
        cur = torch.zeros(n, device=device)
        vel = torch.zeros((n, 3), device=device)
        self.sp_neighborhood(l, int(K), 0.0, -2.0, vel, out_n, out_v, flag, 0, 0, 1.0, cur, 1)
        return torch.nan_to_num(cur, nan=0.0, posinf=0.0)

    # ------------------------------------------------------------------ stats
    def fp_stats(self, device="cuda"):
        n = min(self.fp_n[None], self.max_fp)
        layer = self.fp_layer.to_torch(device=device)[:n]
        rad = self.fp_radius.to_torch(device=device)[:n]
        active = rad > 0
        per_level = [int(((layer == l) & active).sum().item()) for l in range(self.num_levels)]
        return {"n_fp": int(active.sum().item()), "n_fp_alloc": n, "per_level": per_level,
                "params": 14 * int(active.sum().item())}
