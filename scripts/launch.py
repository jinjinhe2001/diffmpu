"""Run a list of shell commands over a pool of GPUs (the recipe runner).

jobs file (recipes/*.jsonl): one JSON object per line
    {"name": "table2/armadillo", "cmd": "python scripts/run_diff.py --task chamfer ... --mesh data/armadillo.ply"}
Rules:
  * lines starting with "#" and blank lines are comments and are skipped
  * "name" is the run directory: the job writes to <root>/runs/<name>/ and its stdout + stderr go to runs/<name>/log.txt
  * "--out runs/<name>" is appended to the command when the command contains no --out
  * a job whose runs/<name>/result.json already exists is skipped (--skip_done 0 re-runs it)
  * the command runs from <root> (default: the parent of scripts/) through the shell with CUDA_VISIBLE_DEVICES set to the
    GPU it was assigned (Taichi follows CUDA_VISIBLE_DEVICES as well); a leading "python " is replaced by the interpreter
    that runs this launcher (sys.executable), so the recipes use the environment the launcher was started from
  * --only <substring> keeps only the jobs whose name contains the substring
  * --gpus 0,1,2 runs one job at a time per listed GPU; --per_gpu N runs N jobs concurrently on each GPU
Usage: python scripts/launch.py recipes/table2_chamfer.jsonl [--gpus 0,1,2,3] [--per_gpu 1] [--only spot]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import queue

ap = argparse.ArgumentParser(description="Run the jobs of a recipes/*.jsonl file over a pool of GPUs (see the module docstring for the file format).")
ap.add_argument("jobs", help="jsonl file: one {\"name\": ..., \"cmd\": ...} per line, # comments")
ap.add_argument("--gpus", default="0", help="comma-separated GPU ids (CUDA_VISIBLE_DEVICES values), one worker per id")
ap.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per GPU")
ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help="working directory of the jobs and parent of runs/ (default: the repository root)")
ap.add_argument("--skip_done", type=int, default=1, help="1: skip jobs whose runs/<name>/result.json exists")
ap.add_argument("--only", default=None, help="run only the jobs whose name contains this substring")
args = ap.parse_args()

jobs = [json.loads(l) for l in open(args.jobs) if l.strip() and not l.lstrip().startswith("#")]
if args.only:
    jobs = [j for j in jobs if args.only in j["name"]]
print(f"[launch] {len(jobs)} jobs from {args.jobs}" + (f" (name contains {args.only!r})" if args.only else ""), flush=True)
gpus = [g for g in args.gpus.split(",") for _ in range(args.per_gpu)]
q = queue.Queue()
for j in jobs:
    q.put(j)
lock = threading.Lock()
done = []


def worker(gpu):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        out_dir = os.path.join(args.root, "runs", j["name"])
        os.makedirs(out_dir, exist_ok=True)
        res = os.path.join(out_dir, "result.json")
        if args.skip_done and os.path.exists(res):
            with lock:
                print(f"[skip] {j['name']} (done)", flush=True)
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = j["cmd"]
        if cmd.startswith("python "):
            exe = sys.executable
            if " " in exe:
                exe = f'"{exe}"'
            cmd = exe + cmd[len("python"):]
        if "--out" not in cmd:
            cmd += f" --out {out_dir}"
        t0 = time.time()
        with lock:
            print(f"[start gpu{gpu}] {j['name']}: {cmd}", flush=True)
        with open(os.path.join(out_dir, "log.txt"), "w") as f:
            p = subprocess.run(cmd, shell=True, cwd=args.root, env=env, stdout=f, stderr=subprocess.STDOUT)
        with lock:
            status = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
            print(f"[done gpu{gpu}] {j['name']} {status} in {time.time()-t0:.0f}s", flush=True)
            done.append((j["name"], status))


threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("[launch] all done:", done, flush=True)
