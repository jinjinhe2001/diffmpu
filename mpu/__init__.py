"""Multi-level Partition of Unity on Differentiable Moving Particles.

Official implementation of "Multi-level Partition of Unity on Differentiable Moving Particles"
(ACM Transactions on Graphics 43(6), SIGGRAPH Asia 2024).

Importing this package is cheap: torch and taichi are only imported when one of the names below is
first accessed (PEP 562 lazy attributes). This matters because ``python -m mpu.poisson_worker`` is
spawned as a plain CPU child process and must not pull in the CUDA stack. Call ``mpu.init()`` (taichi
CUDA backend + seeds) before building anything; see mpu/api.py and the README section "Python API".

Typical use::

    import mpu
    mpu.init(ti_mem_gb=8)
    pos, nrm, cur, meta = mpu.particles_from_mesh("data/Armadillo.ply", n=3_000_000)
    R = mpu.build_mpu(pos, nrm, cur, cfg=mpu.recon_config(), surface_area=meta["surface_area"])
    f, grad = R.eval(pos[:1000])          # field at query points, negative inside
    V, F = R.marching_cubes(512)          # normalised frame [-1, 1]^3
    V_world = mpu.to_world(V, meta)       # back to the input mesh frame
"""
import importlib

__version__ = "1.0.0"

# public name -> submodule (relative to this package); resolved on first attribute access
_LAZY = {
    "init": ".api",
    "recon_config": ".api",
    "particles_from_mesh": ".api",
    "build_mpu": ".api",
    "to_world": ".api",
    "Reconstructor": ".recon",
    "default_config": ".recon",
    "run_marching_cubes": ".recon",
    "load_mesh": ".data",
    "normalize": ".data",
    "make_sample_particles": ".data",
    "sample_surface": ".data",
    "DiffMPU": ".diff",
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Lazy attribute access (PEP 562): import the owning submodule on first use and cache the result."""
    modname = _LAZY.get(name)
    if modname is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(modname, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
