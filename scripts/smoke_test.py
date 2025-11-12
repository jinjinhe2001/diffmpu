"""Data-free smoke test of the Python facade (mpu/api.py) on an analytic sphere.

Usage (GPU machine with CUDA + torch + taichi + trimesh + igl): python scripts/smoke_test.py
Exit code 0 = ok, 1 = a check failed. Builds a 3-level MPU (fine 256) from 300k particles sampled on an
icosphere, then checks the field at the sample points and the marching-cubes volume.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Tolerances: measured values are mean |f| 2.9e-5 and a relative volume error of 0.0007 (RTX 5080, torch 2.11, taichi 1.7.3),
# so these limits leave a factor of ten for GPU / library differences.
FIELD_TOL = 3e-4      # mean |f| at the sample particles (normalised units, domain [-1, 1]^3)
VOLUME_TOL = 0.01     # relative volume error of the marching-cubes mesh at 128^3


def main():
    import trimesh
    import mpu

    mpu.init(ti_mem_gb=2)
    ico = trimesh.creation.icosphere(subdivisions=5, radius=0.6)
    V, F = np.asarray(ico.vertices, dtype=np.float64), np.asarray(ico.faces, dtype=np.int64)
    pos, nrm, cur, meta = mpu.particles_from_mesh((V, F), n=300_000)
    cfg = mpu.recon_config(fine_res=256, num_levels=3, n_sp=300_000)
    R = mpu.build_mpu(pos, nrm, cur, cfg=cfg, surface_area=meta["surface_area"])
    n_fp = R.last_stats["n_fp"]

    f, _ = R.eval(pos[:10000])
    field_abs = float(f.abs().mean())
    field_rms = float((f * f).mean().sqrt())

    V2, F2 = R.marching_cubes(128)
    # normalize() rescales the sphere (radius 0.6 -> 0.6 / (0.6 * 1.2) = 0.8333): take the radius from the
    # normalised vertices (all icosphere vertices lie on the sphere) and compare with the analytic volume.
    radius = float(np.linalg.norm(meta["V"], axis=1).mean())
    vol_expected = 4.0 / 3.0 * np.pi * radius ** 3
    vol_mc = abs(float(trimesh.Trimesh(V2, F2, process=False).volume)) if V2.shape[0] > 0 else 0.0
    volume_err = abs(vol_mc - vol_expected) / vol_expected

    failures = []
    if not field_abs < FIELD_TOL:
        failures.append(f"mean |f| at the sample points {field_abs:.3e} >= {FIELD_TOL:g}")
    if not volume_err < VOLUME_TOL:
        failures.append(f"marching-cubes volume {vol_mc:.5f} vs analytic {vol_expected:.5f} (rel err {volume_err:.4f} >= {VOLUME_TOL:g})")
    if failures:
        print(f"[smoke] FAILED n_fp={n_fp} field_abs={field_abs:.3e} field_rms={field_rms:.3e} volume_err={volume_err:.4f}")
        for msg in failures:
            print("[smoke]   " + msg)
        return 1
    print(f"[smoke] ok n_fp={n_fp} field_rms={field_rms:.3e} volume_err={volume_err:.4f} (mean |f| {field_abs:.3e}, "
          f"{V2.shape[0]} mc verts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
