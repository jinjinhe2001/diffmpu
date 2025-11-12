"""Paper-style feature-particle figures (paper Fig. 9 / Fig. 2): feature particles drawn as spheres of their own support
radius, coloured by level (finest = blue ... coarsest = red), next to the GT and the reconstructed surface, all from one
upright canonical camera.

usage:
  python tools/render_fps.py <out.png> [--res 900] [--az 35 --el 18] [--sphere_scale 1.0] [--strip]
        "<label>=<run_dir>[:particles.pt]" ...
Default sheet (Fig. 9 style): one column per item, rows GT / Ours / Feature particles.
--strip (Fig. 2 style): one row of feature-particle panels (one per item, e.g. particle snapshots of one run) followed by
the target surface, with the particle count under each panel.
run_recon runs are rebuilt from GT samples exactly like scripts/run_recon.py; differentiable runs from the given / newest saved
particles, re-applying the --set overrides recorded in the matching resume_*.json (so reduced-budget thresholds are honoured)."""
import os, sys, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, taichi as ti, trimesh
from PIL import Image, ImageDraw, ImageFont
ti.init(arch=ti.cuda, device_memory_GB=20, log_level=ti.WARN)
from mpu.data import load_mesh, normalize, make_sample_particles
from mpu.recon import Reconstructor, run_marching_cubes
from mpu.render_utils import Renderer, vertex_normals_t, projection, translate

argv = sys.argv[1:]
out_png = argv[0]; rest = argv[1:]
RES, AZ, EL, SPH, STRIP = 900, 35.0, 18.0, 1.0, False
items = []
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--res": RES = int(rest[i + 1]); i += 2; continue
    if a == "--az": AZ = float(rest[i + 1]); i += 2; continue
    if a == "--el": EL = float(rest[i + 1]); i += 2; continue
    if a == "--sphere_scale": SPH = float(rest[i + 1]); i += 2; continue
    if a == "--strip": STRIP = True; i += 1; continue
    label, spec = a.split("=", 1)
    spec, _, cam = spec.partition("@")
    run_dir, _, forced = spec.partition(":")
    items.append((label, run_dir, forced or None, cam)); i += 1

R = Renderer(1, RES, seed=42, device="cuda")                      # rasteriser context only
dr = R.dr


def rot_y(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32)


def rot_x(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32)


PROJ = projection(x=0.5, n=1.5, f=100.0)
MVP = LIGHT = None


def set_camera(cam=""):
    """canonical camera: azimuth about +y, elevation, same projection/distance as the metric cameras, head light.
    cam = 'az,el' or 'az,el,upz' (upz: the mesh is z-up -> rotate it so that z becomes y first)."""
    global MVP, LIGHT
    az, el, upz = AZ, EL, False
    if cam:
        parts = cam.split(",")
        az, el = float(parts[0]), float(parts[1]); upz = len(parts) > 2 and parts[2] == "upz"
    model = rot_x(math.radians(-90.0)) if upz else np.eye(4, dtype=np.float32)
    MV = translate(0, 0, -4.0) @ rot_x(math.radians(el)) @ rot_y(math.radians(az)) @ model
    MVP = torch.as_tensor(PROJ @ MV, dtype=torch.float32, device="cuda")[None]
    campos = np.linalg.inv(MV)[:3, 3]
    LIGHT = torch.as_tensor(-campos / np.linalg.norm(campos), dtype=torch.float32, device="cuda").view(1, 1, 1, 3)


set_camera()

ico = trimesh.creation.icosphere(subdivisions=2)                    # 162 vertices / 320 faces per particle
ICO_V = torch.as_tensor(np.asarray(ico.vertices), dtype=torch.float32, device="cuda")
ICO_F = torch.as_tensor(np.asarray(ico.faces), dtype=torch.long, device="cuda")

# fixed level colours indexed from the finest level: deep blue, light blue, light red, red, dark red (paper Fig. 9 palette)
LEVEL_RGB = torch.tensor([[0.16, 0.32, 0.86], [0.55, 0.70, 0.95], [0.95, 0.55, 0.50], [0.85, 0.15, 0.15], [0.60, 0.05, 0.05], [0.40, 0.0, 0.0]], device="cuda")


def level_colors(layer, num_levels):
    idx = (num_levels - 1 - layer.long()).clamp(0, LEVEL_RGB.shape[0] - 1)         # 0 = finest level
    return LEVEL_RGB[idx]


def build(run_dir, forced=None):
    cfgj = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = dict(cfgj.get("cfg", cfgj)); a = cfgj.get("args", {})
    pts = [forced] if forced else sorted(glob.glob(os.path.join(run_dir, "resume_*", "particles.pt")), key=os.path.getmtime)
    if not pts and os.path.exists(os.path.join(run_dir, "particles.pt")):
        pts = [os.path.join(run_dir, "particles.pt")]
    mesh_path = a.get("mesh") or json.load(open(os.path.join(run_dir, "result.json"))).get("mesh")
    V, F = load_mesh(mesh_path); V, _, _ = normalize(V, scale_factor=a.get("scale_factor", 1.2))
    if pts:
        st = torch.load(pts[-1]); P, N, C = st["sp_pos"].cuda(), st["sp_nrm"].cuda(), st["sp_cur"].cuda()
        src = os.path.relpath(pts[-1], run_dir)
        cfg["far_winding_area"] = float(st.get("far_winding_area") or trimesh.Trimesh(V, F, process=False).area)
        rd = os.path.basename(os.path.dirname(pts[-1]))
        if rd.startswith("resume_"):
            rj = os.path.join(run_dir, rd[len("resume_"):] + ".json")
            if os.path.exists(rj):
                ov = json.load(open(rj)).get("overrides", [])
                if "--set" in ov:
                    k = ov.index("--set") + 1
                    while k < len(ov) and not ov[k].startswith("--"):
                        key, val = ov[k].split("=", 1); cfg[key] = json.loads(val); k += 1
    else:
        P, N, C = make_sample_particles(V, F, int(cfg["n_sp"]), curv_smooth=0, seed=0, curv_mode="robust", normal_mode="vertex", normal_smooth=1, curv_scale=0.004)
        src = "GT samples"; cfg["far_winding_area"] = float(trimesh.Trimesh(V, F, process=False).area)
    cfg["n_sp"] = P.shape[0] + 16; cfg["max_fp"] = max(int(cfg.get("max_fp", 2 ** 19)), 2 ** 20)
    Rc = Reconstructor(cfg)
    Rc.mpu.bin_only(P, N, C, cfg["thresholds"])
    C = Rc.mpu.sp_curvature_pca(K=int(a.get("pca_K", 16)), device="cuda").clamp(0, float(a.get("curvature_max", 400.0)))
    Rc.build(P, N, C, verbose=False)
    m = Rc.mpu; n = min(m.fp_n[None], m.max_fp)
    pos = m.fp_pos.to_torch(device="cuda")[:n]; rad = m.fp_radius.to_torch(device="cuda")[:n]; layer = m.fp_layer.to_torch(device="cuda")[:n]
    act = rad > 0
    field = Rc.field(256); Vm, Fm = run_marching_cubes(field, 256, device="cuda")
    return pos[act], rad[act], layer[act], int(cfg["num_levels"]), m.fp_stats(), P.shape[0], src, (V, F), (Vm, Fm)


def rasterize(Vs, Fs):
    v_hom = torch.nn.functional.pad(Vs, (0, 1), "constant", 1.0)
    v_ndc = torch.matmul(v_hom, MVP.transpose(1, 2)).contiguous()
    rast, _ = dr.rasterize(R.glctx, v_ndc, Fs, [RES * 2, RES * 2])
    return rast, v_ndc


def render_spheres(pos, rad, layer, num_levels, chunk=25000):
    n = pos.shape[0]; S2 = RES * 2
    colors = level_colors(layer, num_levels)
    img = torch.ones((S2, S2, 3), device="cuda"); depth = torch.full((S2, S2), float("inf"), device="cuda")
    with torch.no_grad():
        for s in range(0, n, chunk):
            p, r, c = pos[s:s + chunk], rad[s:s + chunk], colors[s:s + chunk]; m = p.shape[0]
            Vs = (p[:, None, :] + SPH * r[:, None, None] * ICO_V[None]).reshape(-1, 3)
            Fs = (ICO_F[None] + (torch.arange(m, device="cuda") * ICO_V.shape[0])[:, None, None]).reshape(-1, 3).int().contiguous()
            col = c[:, None, :].expand(m, ICO_V.shape[0], 3).reshape(-1, 3).contiguous()
            nrm = ICO_V[None].expand(m, -1, -1).reshape(-1, 3).contiguous()
            rast, _ = rasterize(Vs, Fs)
            pn, _ = dr.interpolate(nrm[None].contiguous(), rast, Fs)
            pc, _ = dr.interpolate(col[None].contiguous(), rast, Fs)
            ndl = torch.sum(-LIGHT * pn, -1, keepdim=True).clamp(0, 1)
            cimg = (pc * (0.35 + 0.65 * ndl) + 0.25 * ndl.pow(32))[0].clamp(0, 1)
            hit = rast[0, ..., 3] > 0; z = rast[0, ..., 2]
            closer = hit & (z < depth)
            img[closer] = cimg[closer]; depth[closer] = z[closer]
    return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).resize((RES, RES), Image.LANCZOS)


def render_surface(Vm, Fm, tint):
    Vt = torch.as_tensor(Vm, dtype=torch.float32, device="cuda"); Fl = torch.as_tensor(Fm, dtype=torch.long, device="cuda")
    Ft = Fl.int().contiguous()
    with torch.no_grad():
        rast, v_ndc = rasterize(Vt, Ft)
        pn, _ = dr.interpolate(vertex_normals_t(Vt, Fl)[None].contiguous(), rast, Ft)
        ndl = torch.sum(-LIGHT * pn, -1, keepdim=True).clamp(0, 1)
        col = (torch.tensor(tint, device="cuda").view(1, 1, 1, 3) * (0.30 + 0.70 * ndl) + 0.08 * ndl.pow(32)).clamp(0, 1)
        img = torch.where(rast[..., 3:4] > 0, col, torch.ones_like(col))
        img = dr.antialias(img.contiguous(), rast, v_ndc, Ft)[0].clamp(0, 1)
    return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).resize((RES, RES), Image.LANCZOS)


def font(px):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", px)
    except Exception:
        return ImageFont.load_default()


F_BIG, F_MID, F_SMALL = font(int(RES * 0.075)), font(int(RES * 0.05)), font(int(RES * 0.036))
GT_TINT, OURS_TINT = (0.80, 0.80, 0.80), (0.86, 0.72, 0.50)

built = []
for label, run_dir, forced, cam in items:
    pos, rad, layer, L, st, n_sp, src, gt, rec = build(run_dir, forced)
    built.append((label, pos, rad, layer, L, st, gt, rec, cam))
    print(f"[render_fps] {label}: {st['n_fp']} fps, per level {st['per_level']}, radius {float(rad.min()):.4f}-{float(rad.max()):.4f}, {n_sp} samples, from {src}", flush=True)

LABEL_W = int(RES * 0.55)
if STRIP:
    # paper Fig. 2: Initialization (surface of the first snapshot) | particles of the snapshots | Result (surface of the last)
    panels = []
    set_camera(built[0][8]); panels.append((render_surface(*built[0][7], OURS_TINT), ""))
    for (_, pos, rad, layer, L, st, gt, rec, cam) in built:
        set_camera(cam); panels.append((render_spheres(pos, rad, layer, L), f"{st['n_fp']:,}"))
    set_camera(built[-1][8]); panels.append((render_surface(*built[-1][7], OURS_TINT), ""))
    head = int(RES * 0.12); foot = int(RES * 0.14)
    sheet = Image.new("RGB", (LABEL_W + len(panels) * RES, head + RES + foot), "white"); d = ImageDraw.Draw(sheet)
    d.text((LABEL_W + 10, int(head * 0.25)), "Initialization", fill="black", font=F_MID)
    d.text((LABEL_W + RES + 10, int(head * 0.25)), "Optimization with differentiable moving particles", fill="black", font=F_MID)
    d.text((LABEL_W + (len(panels) - 1) * RES + 10, int(head * 0.25)), "Result", fill="black", font=F_MID)
    d.text((10, head + RES + int(foot * 0.25)), "Particle nums:", fill="black", font=F_MID)
    for c, (im, txt) in enumerate(panels):
        sheet.paste(im, (LABEL_W + c * RES, head))
        d.text((LABEL_W + c * RES + int(RES * 0.35), head + RES + int(foot * 0.25)), txt, fill="black", font=F_MID)
    sheet.save(out_png); print("saved", out_png, sheet.size, flush=True); sys.exit(0)

head = int(RES * 0.16); ncol = len(built)
sheet = Image.new("RGB", (LABEL_W + ncol * RES, head + 3 * RES + int(RES * 0.12)), "white"); d = ImageDraw.Draw(sheet)
for r, name in enumerate(["GT", "Ours", "Feature\nParticles"]):
    d.multiline_text((20, head + r * RES + int(RES * 0.42)), name, fill="black", font=F_BIG, spacing=8)
for c, (label, pos, rad, layer, L, st, gt, rec, cam) in enumerate(built):
    x = LABEL_W + c * RES; set_camera(cam)
    d.text((x + 12, int(head * 0.18)), label, fill="black", font=F_MID)
    d.text((x + 12, int(head * 0.58)), f"{st['n_fp']:,} feature particles ({14 * st['n_fp'] / 1e6:.2f} M params)", fill="#444444", font=F_SMALL)
    sheet.paste(render_surface(*gt, GT_TINT), (x, head))
    sheet.paste(render_surface(*rec, OURS_TINT), (x, head + RES))
    sheet.paste(render_spheres(pos, rad, layer, L), (x, head + 2 * RES))
y0 = head + 3 * RES + int(RES * 0.03); x0 = LABEL_W + 12
d.text((x0, y0), "particle colour = level of the multi-level grid (sphere size = support radius):", fill="#333333", font=F_SMALL)
xx = x0 + int(RES * 2.05)
Lmax = max(b[4] for b in built)
for k in range(Lmax):
    rgb = tuple(int(255 * v) for v in LEVEL_RGB[Lmax - 1 - k].tolist())
    d.ellipse((xx, y0 + 4, xx + int(RES * 0.03), y0 + 4 + int(RES * 0.03)), fill=rgb)
    d.text((xx + int(RES * 0.04), y0), "coarsest level" if k == 0 else ("finest level" if k == Lmax - 1 else f"level {k}"), fill="#333333", font=F_SMALL)
    xx += int(RES * 0.52)
sheet.save(out_png); print("saved", out_png, sheet.size, flush=True)
