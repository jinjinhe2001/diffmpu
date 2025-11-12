# Data

Nothing is redistributed in this directory. Every mesh used by the recipes must be downloaded
from its original source and placed here under the **exact** file name below (the recipes in
`recipes/*.jsonl` reference `data/<name>`; Linux is case-sensitive). `.ply` / `.obj` files in
`data/` are ignored by git (`.gitignore`), only this README is tracked.

**Case warning.** `data/Armadillo.ply` (capital A: the raw Stanford scan, Tables 1, 5, 6) and
`data/armadillo.ply` (lower case: the evaluation mesh of Neural Implicit Evolution, Tables 2 and 4)
are two different meshes. They coexist on Linux but would collide on the case-insensitive default
file systems of Windows and macOS; on those systems keep one of them under a different name and
edit the corresponding recipe lines.

## Files

| file | used by (table / recipe) | source | original file name | preprocessing |
|---|---|---|---|---|
| `data/Armadillo.ply` | Table 1 `table1/armadillo`; Table 5 rotate lines (`init_mesh`); Table 6 `table6/moving_armadillo`, `table6/fixed_armadillo` | Stanford 3D Scanning Repository, http://graphics.stanford.edu/data/3Dscanrep/ | `Armadillo.ply` (from `Armadillo.ply.gz`) | none (used raw) |
| `data/bunny.ply` | Table 1 `table1/bunny` | Stanford 3D Scanning Repository | `bunny.tar.gz` -> `bunny/reconstruction/bun_zipper.ply` | `python scripts/data/make_watertight.py bun_zipper.ply data/bunny.ply` (hole closing) |
| `data/dragon_vrip.ply` | Table 1 `table1/dragon` | Stanford 3D Scanning Repository | `dragon_recon.tar.gz` -> `dragon_recon/dragon_vrip.ply` | none |
| `data/happy_vrip.ply` | Table 1 `table1/happy`; Table 6 `table6/moving_happy`, `table6/fixed_happy` | Stanford 3D Scanning Repository | `happy_recon.tar.gz` -> `happy_recon/happy_vrip.ply` | none |
| `data/lucy.ply` | Table 1 `table1/lucy` | Stanford 3D Scanning Repository | `lucy.tar.gz` -> `lucy.ply` | `scripts/data/make_watertight.py` (hole closing; the raw scan is open) |
| `data/xyzrgb_statuette.ply` | Table 1 `table1/statuette`; Table 6 `table6/moving_statuette`, `table6/fixed_statuette` | Stanford 3D Scanning Repository | `xyzrgb_statuette.ply` (from `xyzrgb_statuette.ply.gz`) | none |
| `data/xyzrgb_dragon.ply` | no recipe (optional additional Stanford model for `scripts/run_recon.py`) | Stanford 3D Scanning Repository | `xyzrgb_dragon.ply` (from `xyzrgb_dragon.ply.gz`) | `scripts/data/make_watertight.py` (hole closing); optional |
| `data/spot.obj` | Table 2 `table2/spot`; Table 3 `table3/spot_25`, `spot_50`, `spot_100`; Table 4 `table4/spot_1024_long` | Keenan Crane's 3D model repository (CC0), https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ | `spot.zip` -> `spot_triangulated.obj`, renamed | none (renamed) |
| `data/fandisk.obj` | Table 2 `table2/fandisk`; Table 4 `table4/fandisk_1024_long` | Hoppe's fandisk, the classic CAD test model (for example `fandisk.off` of the libigl tutorial data, https://github.com/libigl/libigl-tutorial-data, converted to OBJ) | `fandisk.obj` | none |
| `data/armadillo.ply` | Table 2 `table2/armadillo`; Table 4 `table4/armadillo_1536_s0` and the stage-2 lines of `table4_final_flow.jsonl` | a watertight copy of the Stanford armadillo as used by the Neural Implicit Evolution experiments (Mehta et al. 2022); the scripts normalise every mesh, so the Stanford scan `Armadillo.ply` gives the same task | `armadillo.ply` | none |
| `data/armadillo.obj` | Table 3 `table3/armadillo_25`, `armadillo_50`, `armadillo_100` | a remeshed copy of the Stanford armadillo (same remark as above) | `armadillo.obj` | none |
| `data/genus6.ply` | Table 2 `table2/genus6_fd2`; Table 3 `table3/genus6_25_b102`, `genus6_50`, `genus6_100`; Table 4 `table4/genus6_1024_long` | evaluation data of Neural Implicit Evolution (Mehta et al. 2022), `data/` folder of https://github.com/ishit/nie (no licence declared upstream) | `genus6.ply` | none |
| `data/kangaroo.ply` | Table 2 `table2/kangaroo_fd2`; Table 4 `table4/kangaroo_1536_shrink_nosw` | evaluation data of Neural Implicit Evolution (Mehta et al. 2022), `data/` folder of https://github.com/ishit/nie (no licence declared upstream) | `kangaroo.ply` | none |
| `data/dino.ply` | Table 2 `table2/dino_fd2`; Table 4 `table4/dino_1536_shrink_nosw` | evaluation data of Neural Implicit Evolution (Mehta et al. 2022), `data/` folder of https://github.com/ishit/nie (no licence declared upstream) | `dino.ply` | none |
| `data/vbunny.ply` | Table 2 `table2/vbunny`; Table 4 `table4/vbunny_1536_shrink_nosw`; `tools/diagnostics/render_debug.py` | evaluation data of Neural Implicit Evolution (Mehta et al. 2022), `data/` folder of https://github.com/ishit/nie (no licence declared upstream) | `vbunny.ply` | none |
| `data/rind_wide.ply` | Table 2 `table2/rind` | evaluation data of Neural Implicit Evolution (Mehta et al. 2022), `data/` folder of https://github.com/ishit/nie (no licence declared upstream) | `rind_wide.ply` | none |
| `data/thingi10k/<file_id>.ply` (50 files + `list.txt`) | Table 1, Thingi10K rows (no recipe file: `scripts/run_recon.py` with the `table1_stanford.jsonl` flags on every file) | Thingi10K, https://ten-thousand-models.appspot.com/ (per-model licences, see the Thingi10K metadata) | `<file_id>.stl` / `.obj` as served by the `thingi10k` package | `python tools/diagnostics/select_thingi.py --out data/thingi10k` (deterministic selection of 50 clean models, exported as PLY) |

## Coordinate frame

All meshes are centred and scaled by the scripts themselves (`mpu.data.normalize`: bounding-box
centre moved to the origin, then divided by `max |coord| * scale_factor` with `scale_factor = 1.2`,
so the normalised mesh has `max |coord| = 1/1.2` and fits inside `[-1, 1]^3`). Every output of the
scripts (reconstructed meshes, particles, metrics such as Chamfer distances) is expressed in that
normalised frame, not in the original units of the scan. Exception: the Table 5 `rotate` test
normalises its `init_mesh` to a half-extent of `init_scale` (default 0.6) instead, see
`scripts/run_deform.py`.
