"""Console summary of finished runs: collects the result.json files under the runs directory into text tables (one block per
script layout).

Usage: python scripts/summarize.py <glob> [<glob> ...] [--runs DIR]
  python scripts/summarize.py "table1/*"                  # run_recon rows: IoU, Chamfer, NAE, build / query time
  python scripts/summarize.py "table2/*" "table3/*"       # run_diff rows (result.json = {"result": {...}, "log": [...]})
  python scripts/summarize.py "table5/*" --runs /data/runs

The globs are relative to --runs (default "runs", the runs/ directory below the current working directory, which is where
scripts/launch.py writes when started from the repository root). The layout of every result.json is detected from its
keys: run_recon.py (has "stats"), run_deform.py (has "test" and "volume_loss_pct"), run_diff.py (everything else; the
"result" wrapper is unwrapped). Missing values print as nan; a nan-aware MEAN row closes every block with more than one
row. Columns: Chamfer values are x1e-5; p2p500k falls back to cl2_p2p_500k -> cl2_p2p_100k; NAE is nae_grad (analytic
gradient at the GT samples, run_recon) or nae_mesh (marching-cubes mesh normals, run_diff) ; min = time_total / 60.
"""
import argparse
import glob
import json
import math
import os

NAN = float("nan")


def num(r, *keys, scale=1.0):
    """First present key of r as a float (scaled), else nan."""
    for k in keys:
        v = r.get(k)
        if v is not None:
            try:
                return float(v) * scale
            except (TypeError, ValueError):
                return NAN
    return NAN


def cell(v, w, p):
    if isinstance(v, str):
        return f"{v:>{w}s}"
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'nan':>{w}s}"
    return f"{v:{w}.{p}f}"


def nanmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else NAN


def layout_of(r):
    if "test" in r and "volume_loss_pct" in r:
        return "deform"
    if "stats" in r:
        return "recon"
    return "diff"


# (header, width, decimals, getter)
RECON = [
    ("nfp", 7, 0, lambda r: num(r.get("stats", {}), "n_fp")),
    ("MPar", 5, 2, lambda r: num(r.get("stats", {}), "params", scale=1e-6)),
    ("IoU", 7, 4, lambda r: num(r, "iou")),
    ("IoUmc", 7, 4, lambda r: num(r, "iou_mc_mesh")),
    ("CL2p2m", 7, 3, lambda r: num(r, "cl2_p2m", scale=1e5)),
    ("r2g", 6, 3, lambda r: num(r, "cl2_p2m_r2g", scale=1e5)),
    ("p2p500k", 7, 3, lambda r: num(r, "cl2_p2p_500k", scale=1e5)),
    ("p2p1m", 6, 3, lambda r: num(r, "cl2_p2p_1m", scale=1e5)),
    ("NAEg", 5, 2, lambda r: num(r, "nae_grad")),
    ("NAEm", 5, 2, lambda r: num(r, "nae_mesh_vn", "nae_mesh")),
    ("Hd_r2g", 6, 3, lambda r: num(r, "hausdorff_r2g")),
    ("build", 6, 1, lambda r: num(r.get("timings_warm", {}), "total_build") if "total_build" in r.get("timings_warm", {}) else num(r.get("timings", {}), "total_build")),
    ("q512", 5, 2, lambda r: num(r.get("timings", {}), "query_512")),
]
DIFF = [
    ("IoU", 7, 4, lambda r: num(r, "iou")),
    ("CL2p2m", 7, 3, lambda r: num(r, "cl2_p2m", scale=1e5)),
    ("p2p500k", 7, 3, lambda r: num(r, "cl2_p2p_500k", "cl2_p2p_100k", scale=1e5)),
    ("NAE", 5, 2, lambda r: num(r, "nae_grad", "nae_mesh")),
    ("PSNRtr", 6, 2, lambda r: num(r, "psnr_train")),
    ("PSNRte", 6, 2, lambda r: num(r, "psnr_test")),
    ("nfp", 7, 0, lambda r: num(r, "n_fp")),
    ("min", 6, 1, lambda r: num(r, "time_total", scale=1.0 / 60.0)),
]
DEFORM = [
    ("test", 7, 0, lambda r: str(r.get("test", "nan"))),
    ("fine", 5, 0, lambda r: num(r, "fine_res")),
    ("steps", 6, 0, lambda r: num(r, "steps")),
    ("IoU", 8, 5, lambda r: num(r, "iou")),
    ("vol%", 8, 4, lambda r: num(r, "volume_loss_pct")),
    ("nfp", 7, 0, lambda r: num(r, "n_fp_final")),
]
LAYOUTS = {"recon": ("run_recon", RECON), "diff": ("run_diff", DIFF), "deform": ("run_deform", DEFORM)}


def print_block(title, rows, cols):
    print(f"{title:44s} " + " ".join(f"{h:>{w}s}" for h, w, _, _ in cols))
    vals = []
    for name, r in rows:
        v = [fn(r) for _, _, _, fn in cols]
        vals.append(v)
        print(f"{name:44s} " + " ".join(cell(x, w, p) for x, (_, w, p, _) in zip(v, cols)))
    if len(rows) > 1:
        m = [("" if any(isinstance(x, str) for x in col) else nanmean(col)) for col in zip(*vals)]
        print(f"{'MEAN':44s} " + " ".join(cell(x, w, p) for x, (_, w, p, _) in zip(m, cols)))


def main():
    ap = argparse.ArgumentParser(description="Summarise <runs>/<glob>/result.json files into text tables (one block per script layout).")
    ap.add_argument("patterns", nargs="+", help="run-directory globs relative to --runs, e.g. 'table1/*'")
    ap.add_argument("--runs", default="runs", help="runs directory (default: runs)")
    a = ap.parse_args()
    blocks = {k: [] for k in LAYOUTS}
    n = 0
    for pat in a.patterns:
        for p in sorted(glob.glob(os.path.join(a.runs, pat, "result.json"))):
            try:
                r = json.load(open(p))
            except Exception as e:
                print(f"[skip] {p}: {e}")
                continue
            r = r.get("result", r)          # run_diff writes {"result": {...}, "log": [...]}
            name = os.path.relpath(os.path.dirname(p), a.runs).replace(os.sep, "/")
            blocks[layout_of(r)].append((name, r))
            n += 1
    if n == 0:
        print(f"no result.json under {a.runs} for {a.patterns}")
        return
    first = True
    for key, (title, cols) in LAYOUTS.items():
        if not blocks[key]:
            continue
        if not first:
            print()
        first = False
        print_block(f"{title} ({len(blocks[key])} runs)", blocks[key], cols)


if __name__ == "__main__":
    main()
