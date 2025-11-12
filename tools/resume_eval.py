"""Run only the final stage of a differentiable run, from its saved pre-final particles (particles.pt):
[MST orientation] -> final resample (optionally denser) -> [final dense flow] -> [final fine re-fit] -> evaluation.
Any run_diff flag can be overridden (the run's config.json provides the defaults), e.g.
  python tools/resume_eval.py runs/table2/vbunny --out_name result_final3m.json \
      --final_n_sp 3000000 --final_flow_iters 40 --final_n_gt 3000000 --set max_fp=1048576 --ti_mem 24
Table 4 stage 2 (recipes/table4_final_flow.jsonl) is this script with the corrected late-flow flags.

Usage: python tools/resume_eval.py <run_dir> [--out_name result_resume.json] [--particles file.pt] [run_diff flags ...]
Outputs: <run_dir>/<out_name> (the metrics) and <run_dir>/resume_<out_name stem>/ with result.json, final.ply and the
particles after the stage (particles.pt; pass it to --particles to evaluate the same particles again, e.g. with another
--eval_mc_res). The script also works as a flat copy (tools_resume_eval.py) placed anywhere inside the repository: the
repository root is the closest ancestor directory that contains the mpu/ package.
"""
import os, sys, json, time, argparse


def find_root(start):
    """Closest ancestor of start (inclusive) that contains the mpu/ package and scripts/run_diff.py."""
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "mpu")) and os.path.isfile(os.path.join(d, "scripts", "run_diff.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(f"[resume] cannot find the repository root (a directory containing mpu/ and scripts/run_diff.py) above {start}")
        d = parent


root = find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts"))
import numpy as np, torch, taichi as ti, trimesh
from run_diff import build_parser

# the run_diff parser plus this script's own options parses the overrides, with the saved args of the run as defaults
ap = build_parser()
ap.add_argument("run_dir", help="finished run_diff.py output directory (config.json + particles.pt)")
ap.add_argument("--out_name", default="result_resume.json", help="metrics file written into run_dir (its stem also names the resume_<stem>/ directory)")
ap.add_argument("--particles", default=None, help="particles .pt to start from (default: <run_dir>/particles.pt)")
for act in ap._actions:
    if act.option_strings:        # every flag becomes optional (defaults come from the run); the positional run_dir stays required
        act.required = False
run_dir = ap.parse_known_args()[0].run_dir
cfgj = json.load(open(os.path.join(run_dir, "config.json")))
cfg, args_d = cfgj["cfg"], cfgj["args"]
dests = {a.dest for a in ap._actions}
for k, v in args_d.items():
    if k in dests and k not in ("run_dir", "out_name", "particles"):
        ap.set_defaults(**{k: v})
args = ap.parse_args()
args.out = run_dir
out_name, particles_path = args.out_name, args.particles
rest = [a for a in sys.argv[1:] if a != run_dir]      # the overrides (recorded in the result)
for kv in args.set:
    k, v = kv.split("=", 1)
    cfg[k] = json.loads(v)
cfg["n_sp"] = max(int(args.n_sp * (args.densify_cap if args.densify_every > 0 else 1.0)), int(getattr(args, "final_n_sp", 0))) + 16
print("[resume] overrides:", rest, flush=True)

ti.init(arch=ti.cuda, device_memory_GB=float(getattr(args, "ti_mem", 16.0)), log_level=ti.WARN, random_seed=args.seed)
from mpu.data import load_mesh, normalize
from mpu.diff import DiffMPU

V, F = load_mesh(args.mesh)
V, _, _ = normalize(V, scale_factor=args.scale_factor)
D = DiffMPU(args, cfg, V, F)
st = torch.load(particles_path or os.path.join(run_dir, "particles.pt"))
D.sp_pos, D.sp_nrm, D.sp_cur = st["sp_pos"].cuda(), st["sp_nrm"].cuda(), st["sp_cur"].cuda()
D.sp_vel = torch.zeros_like(D.sp_pos)
D.it = int(st.get("it", args.iters))
# far-field normalisation: the saved config only holds the initial sphere area -> estimate from the particles' Poisson mesh
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(D.sp_pos.double().cpu().numpy()); pcd.normals = o3d.utility.Vector3dVector(D.sp_nrm.double().cpu().numpy())
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, linear_fit=True, n_threads=16)
dens = np.asarray(dens); mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.03))
D.cfg["far_winding_area"] = float(mesh.get_surface_area())
print(f"[resume] {D.sp_pos.shape[0]} particles from it {D.it}; far area {D.cfg['far_winding_area']:.3f}", flush=True)
D.build()
t0 = time.time()
out_dir = os.path.join(run_dir, "resume_" + out_name.replace(".json", ""))
os.makedirs(out_dir, exist_ok=True)
res = D.evaluate(out_dir, mc_res=args.eval_mc_res, iou_res=args.iou_res, chamfer_n=args.chamfer_n)
res["time_final_stage"] = time.time() - t0
res["overrides"] = rest
json.dump(res, open(os.path.join(run_dir, out_name), "w"), indent=1)
json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
torch.save({"sp_pos": D.sp_pos.cpu(), "sp_nrm": D.sp_nrm.cpu(), "sp_cur": D.sp_cur.cpu(), "it": int(getattr(D, "it", 0))}, os.path.join(out_dir, "particles.pt"))
print("[resume]", json.dumps(res), flush=True)
