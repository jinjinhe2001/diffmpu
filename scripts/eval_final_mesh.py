"""Re-evaluate a differentiable run from a saved mesh (final.ply, or the latest mesh_XXXXX.ply if the
run is still going) after removing interior / floating fragments (keep connected components with
>= min_frac of the faces). Writes result_clean.json (and result.json if the run has none yet).

python scripts/eval_final_mesh.py runs/render/armadillo [--min_frac 0.05] [--mesh final.ply|latest]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("run_dirs", nargs="+")
ap.add_argument("--min_frac", type=float, default=0.05)
ap.add_argument("--iou_res", type=int, default=256)
ap.add_argument("--chamfer_n", type=int, default=500000)
ap.add_argument("--mesh", default="final.ply", help="final.ply or latest (newest mesh_XXXXX.ply)")
ap.add_argument("--no_raw", action="store_true")
ap.add_argument("--out_name", default="result_clean.json")
args = ap.parse_args()

from mpu.data import load_mesh, normalize
from mpu import metrics as M

for rd in args.run_dirs:
    cfgp = os.path.join(rd, "config.json")
    if not os.path.exists(cfgp):
        print("[skip]", rd, flush=True)
        continue
    mesh_path = os.path.join(rd, args.mesh)
    iters = None
    if args.mesh == "latest" or not os.path.exists(mesh_path):
        cands = sorted(glob.glob(os.path.join(rd, "mesh_*.ply")))
        if os.path.exists(os.path.join(rd, "final.ply")):
            mesh_path = os.path.join(rd, "final.ply")
        elif cands:
            mesh_path = cands[-1]
            iters = int(re.findall(r"mesh_(\d+)", mesh_path)[0])
        else:
            print("[skip] no mesh", rd, flush=True)
            continue
    cfg = json.load(open(cfgp))
    a = cfg["args"]
    V, F = load_mesh(a["mesh"])
    V, _, _ = normalize(V, scale_factor=a.get("scale_factor", 1.2))
    m = trimesh.load(mesh_path, process=False)
    comps = m.split(only_watertight=False)
    nf = np.array([len(c.faces) for c in comps])
    keep = [c for c, n in zip(comps, nf) if n >= args.min_frac * nf.sum()]
    mm = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    reV, reF = np.asarray(mm.vertices, dtype=np.float64), np.asarray(mm.faces, dtype=np.int64)
    rawV, rawF = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)
    res = {"mesh_used": os.path.basename(mesh_path), "n_components_total": int(len(comps)), "n_components_kept": int(len(keep)),
           "faces_total": int(nf.sum()), "faces_kept": int(len(reF))}
    Q = M.grid_points(args.iou_res, device="cuda").double().cpu().numpy()
    occ_gt = M.winding_occupancy(V, F, Q)
    res["iou_clean"] = M.iou(occ_gt, M.winding_occupancy(reV, reF, Q))
    cm = M.chamfer_and_nae(V, F, reV, reF, n_samples=args.chamfer_n, device="cuda")
    res.update({k + "_clean": v for k, v in cm.items()})
    if args.no_raw:
        res["iou_raw_mesh"] = float("nan")
        res["cl2_p2p_100k_raw_mesh"] = float("nan")
    else:
        res["iou_raw_mesh"] = M.iou(occ_gt, M.winding_occupancy(rawV, rawF, Q))
        cm_raw = M.chamfer_and_nae(V, F, rawV, rawF, n_samples=min(args.chamfer_n, 200000), device="cuda")
        res["cl2_p2p_100k_raw_mesh"] = cm_raw["cl2_p2p_100k"]
    if a.get("task") == "render":
        from mpu.render_utils import Renderer
        R = Renderer(a["views"], a["res"], seed=42, device="cuda")
        tgt = R.render_mesh_np(V, F)
        img = R.render_mesh_np(reV, reF)
        res["psnr_train_clean"] = float(-10 * torch.log10((img - tgt).square().mean().clamp_min(1e-12)))
        Rt = Renderer(a["test_views"], a["res"], seed=4242, device="cuda")
        res["psnr_test_clean"] = float(-10 * torch.log10((Rt.render_mesh_np(reV, reF) - Rt.render_mesh_np(V, F)).square().mean().clamp_min(1e-12)))
    json.dump(res, open(os.path.join(rd, args.out_name), "w"), indent=1)
    rp = os.path.join(rd, "result.json")
    if not os.path.exists(rp):
        stub = {"result": {"iou": res["iou_raw_mesh"], "cl2_p2p_100k": res["cl2_p2p_100k_raw_mesh"], "iters": iters,
                           "task": a.get("task"), "mesh": a["mesh"], "n_fp": None, "note": f"evaluated from {os.path.basename(mesh_path)} while the run was still going"}}
        json.dump(stub, open(rp, "w"), indent=1)
    print(rd, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()
                          if k in ("mesh_used", "n_components_total", "n_components_kept", "iou_clean", "iou_raw_mesh", "cl2_p2p_100k_clean", "cl2_p2p_500k_clean", "cl2_p2m_clean", "psnr_train_clean", "psnr_test_clean")}), flush=True)
