"""Open3D screened Poisson reconstruction in a separate process (Open3D's PoissonRecon occasionally segfaults with
"bad average roots"; a crash in the child only costs one resample instead of the whole run).
The child is a plain `python -m mpu.poisson_worker in.npz out.npz depth` call (no fork of the CUDA/taichi parent,
no re-import of the caller's main script), data is exchanged through temporary .npz files."""
import os, sys, subprocess, tempfile
import numpy as np


def _run(in_path, out_path, depth):
    import open3d as o3d
    with np.load(in_path) as d:                      # close the file before the parent deletes it (Windows)
        P, N = np.array(d["P"]), np.array(d["N"])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(P, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(N, dtype=np.float64))
    threads = int(os.environ.get("MPU_POISSON_THREADS", "16"))       # Open3D with all cores of a big node is ~4x SLOWER (OpenMP thrash)
    try:
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth), linear_fit=True, n_threads=threads)
    except TypeError:
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth), linear_fit=True)
    np.savez(out_path, V=np.asarray(mesh.vertices, dtype=np.float64), F=np.asarray(mesh.triangles, dtype=np.int64),
             dens=np.asarray(dens, dtype=np.float64))


def poisson_safe(P, N, depth, retries=2, jitter=1e-5, seed=0, timeout=1800):
    """Returns (V, F, density) or None if every attempt crashed. Retries add a tiny positional jitter (changes the
    octree and therefore the root configuration that triggers the crash)."""
    rng = np.random.RandomState(seed)
    P = np.asarray(P, dtype=np.float64); N = np.asarray(N, dtype=np.float64)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = tempfile.mkdtemp(prefix="poisson_")
    in_path, out_path = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
    try:
        for attempt in range(retries + 1):
            Pa = P if attempt == 0 else P + rng.normal(0.0, jitter, size=P.shape)
            np.savez(in_path, P=Pa, N=N)
            if os.path.exists(out_path):
                os.remove(out_path)
            env = dict(os.environ); env.pop("CUDA_VISIBLE_DEVICES", None)
            try:
                r = subprocess.run([sys.executable, "-m", "mpu.poisson_worker", in_path, out_path, str(int(depth))],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[poisson] child timed out (attempt {attempt + 1}/{retries + 1})", flush=True)
                continue
            if r.returncode == 0 and os.path.exists(out_path):
                with np.load(out_path) as d:          # NpzFile keeps the file open; the finally block below deletes it
                    return np.array(d["V"]), np.array(d["F"]), np.array(d["dens"])
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-2:]
            print(f"[poisson] child failed rc={r.returncode} (attempt {attempt + 1}/{retries + 1}): {' | '.join(tail)}", flush=True)
        return None
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
