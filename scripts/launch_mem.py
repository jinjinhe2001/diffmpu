"""Memory-aware job launcher: starts queued jobs on any GPU whose free memory (nvidia-smi) exceeds the job's need.
jobs jsonl: {"name": ..., "cmd": ..., "log": path, "mem_gb": 30}
usage: python scripts/launch_mem.py jobs.jsonl [--gpus 0,1,...] [--poll 30] [--cooldown 150]"""
import argparse, json, os, subprocess, time, shlex

ap = argparse.ArgumentParser()
ap.add_argument("jobs")
ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
ap.add_argument("--poll", type=float, default=30.0)
ap.add_argument("--cooldown", type=float, default=150.0, help="seconds after launching on a GPU before it is considered again (lets the job allocate)")
ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]
jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
queue = list(jobs)
running = {}       # gpu -> (job, Popen, t_start)
last_launch = {g: 0.0 for g in gpus}


def free_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    fm = {}
    for line in out.strip().splitlines():
        i, u, t = [x.strip() for x in line.split(",")]
        fm[int(i)] = (float(t) - float(u)) / 1024.0
    return fm


print(f"[launch_mem] {len(queue)} jobs on gpus {gpus}", flush=True)
while queue or running:
    for g, (job, p, t0) in list(running.items()):
        if p.poll() is not None:
            print(f"[done gpu{g}] {job['name']} rc={p.returncode} in {time.time() - t0:.0f}s", flush=True)
            del running[g]
    fm = free_mem()
    for g in gpus:
        if not queue:
            break
        if g in running or time.time() - last_launch[g] < a.cooldown:
            continue
        need = float(queue[0].get("mem_gb", 30))
        if fm.get(g, 0.0) >= need:
            job = queue.pop(0)
            log = os.path.join(a.root, job.get("log", f"runs/diag/{job['name'].replace('/', '_')}.log"))
            os.makedirs(os.path.dirname(log), exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            p = subprocess.Popen(job["cmd"], shell=True, cwd=a.root, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[g] = (job, p, time.time()); last_launch[g] = time.time()
            print(f"[start gpu{g} free {fm[g]:.0f}GB] {job['name']}: {job['cmd'][:120]}", flush=True)
    time.sleep(a.poll)
print("[launch_mem] all done", flush=True)
