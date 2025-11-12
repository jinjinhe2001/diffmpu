"""Make scanned meshes watertight: drop tiny components, close holes (pymeshlab), verify.
Fallback: generalised-winding-number marching cubes at --wn_res if hole closing fails (or with --force_wn).

python scripts/data/make_watertight.py in.ply out.ply [--min_comp_frac 0.002] [--max_hole 20000] [--wn_res 768] [--force_wn]

Requires: pymeshlab (GPL-3), libigl, scikit-image, open3d.
(trimesh, numpy and scipy from the core requirements; open3d is only the fallback loader, pymeshlab only the
hole closing, libigl / scikit-image / torch only the winding-number fallback.)
"""
import argparse
import sys
import time
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--min_comp_frac", type=float, default=0.002)
ap.add_argument("--max_hole", type=int, default=20000)
ap.add_argument("--wn_res", type=int, default=768)
ap.add_argument("--force_wn", action="store_true")
args = ap.parse_args()

t0 = time.time()
try:
    m = trimesh.load(args.inp, process=False, force="mesh")
except Exception:
    import open3d as o3d
    om = o3d.io.read_triangle_mesh(args.inp)
    m = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
print(f"[in] V {len(m.vertices)} F {len(m.faces)} watertight={m.is_watertight} load {time.time()-t0:.1f}s", flush=True)

# drop small components
comps = m.split(only_watertight=False)
if len(comps) > 1:
    nf = np.array([len(c.faces) for c in comps])
    keep = [c for c, n in zip(comps, nf) if n >= args.min_comp_frac * nf.sum()]
    print(f"[comp] {len(comps)} components, keeping {len(keep)} (dropped {len(comps)-len(keep)} small ones, "
          f"{nf.sum()-sum(len(c.faces) for c in keep)} faces)", flush=True)
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]

ok = False
if not args.force_wn:
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)))
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        ms.meshing_repair_non_manifold_vertices()
        for it in range(3):
            ms.meshing_close_holes(maxholesize=args.max_hole, newfaceselected=False, selfintersection=False)
            mm = ms.current_mesh()
            t = trimesh.Trimesh(mm.vertex_matrix(), mm.face_matrix(), process=False)
            print(f"[close] iter {it}: F {len(t.faces)} watertight={t.is_watertight} consistent={t.is_winding_consistent}", flush=True)
            if t.is_watertight:
                break
        if t.is_watertight:
            if not t.is_winding_consistent:
                trimesh.repair.fix_normals(t)
            if t.volume < 0:
                t.invert()
            m = t
            ok = True
    except Exception as e:
        print("[close] pymeshlab failed:", repr(e)[:300], flush=True)

if not ok:
    import igl
    import torch
    print(f"[wn] winding-number marching cubes at {args.wn_res}^3", flush=True)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    c = (V.max(0) + V.min(0)) / 2
    s = np.abs(V - c).max() * 1.05
    Vn = (V - c) / s
    res = args.wn_res
    ax = (np.arange(res) + 0.5) * (2.0 / res) - 1.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    Q = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    # only evaluate the winding number in a band near the surface; far points by nearest-sample sign
    from scipy.spatial import cKDTree
    sub = Vn[np.random.RandomState(0).choice(len(Vn), min(len(Vn), 2_000_000), replace=False)]
    d, _ = cKDTree(sub).query(Q, workers=-1)
    band = d < 4.0 * (2.0 / res)
    print(f"[wn] band points {band.sum()} / {len(Q)}", flush=True)
    w = np.zeros(len(Q))
    t1 = time.time()
    w[band] = igl.fast_winding_number(Vn, F, np.ascontiguousarray(Q[band]))
    print(f"[wn] winding numbers in {time.time()-t1:.0f}s", flush=True)
    # far points: propagate sign by flood fill from band (inside if w>0.5), using scipy labeling
    from scipy import ndimage
    occ = np.zeros(len(Q), dtype=bool)
    occ[band] = w[band] > 0.5
    far = ~band
    # label connected far regions; a region is inside if its band-neighbours are mostly inside
    lab, nl = ndimage.label(far.reshape(res, res, res))
    lab = lab.ravel()
    print(f"[wn] {nl} far regions", flush=True)
    field = np.where(band, 0.5 - w, 0.0)
    inside_band = (band & (w > 0.5)).reshape(res, res, res)
    outside_band = (band & (w <= 0.5)).reshape(res, res, res)
    dil_in = ndimage.binary_dilation(inside_band).ravel()
    dil_out = ndimage.binary_dilation(outside_band).ravel()
    ni = np.bincount(lab[dil_in], minlength=nl + 1)
    no = np.bincount(lab[dil_out], minlength=nl + 1)
    region_val = np.where(ni > no, -1.0, 1.0)
    region_val[0] = 0.0
    field[far] = region_val[lab[far]]
    field = field.reshape(res, res, res)
    import skimage.measure
    sp = 2.0 / res
    Vm, Fm, _, _ = skimage.measure.marching_cubes(field, level=0.0, spacing=(sp,) * 3, method="lewiner")
    Vm = (Vm + (-1.0 + 0.5 * sp)) * s + c
    m = trimesh.Trimesh(Vm, Fm, process=False)
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        nf = np.array([len(cc.faces) for cc in comps])
        m = trimesh.util.concatenate([cc for cc, n in zip(comps, nf) if n >= 0.001 * nf.sum()])
    if m.volume < 0:
        m.invert()
    print(f"[wn] mesh F {len(m.faces)} watertight={m.is_watertight}", flush=True)

m.export(args.out)
print(f"[out] {args.out}: V {len(m.vertices)} F {len(m.faces)} watertight={m.is_watertight} volume {m.volume:.6g} total {time.time()-t0:.0f}s", flush=True)
