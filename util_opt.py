"""util_opt.py — CPU/RAM utilization controls for floppy-cognition (stdlib only).

Goals (MOGOPs-style measurable targets):
  - Saturate available CPU cores with *productive* work (GEMM tiles + concurrent
    neuro/FMC), not busy-wait.
  - Bound RAM: reuse buffers, avoid unbounded history growth during optim runs.
  - vRAM: N/A on pure-Python laser (CPU-only). Reported as 0 with note.

Pure-Python arithmetic holds the GIL; threads do not speed matmul.
Process-pool row tiling is used for large 2D GEMMs when beneficial.
"""
from __future__ import annotations

import os
import sys
import time
import math
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def cpu_count():
    return max(1, os.cpu_count() or 1)

def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def cpu_times():
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        return int(parts[1]), int(parts[3]), int(parts[4])
    except Exception:
        return 0, 0, 0

def sample_utilization(dt=0.15):
    u1, s1, i1 = cpu_times()
    time.sleep(dt)
    u2, s2, i2 = cpu_times()
    du, ds, di = u2 - u1, s2 - s1, i2 - i1
    total = du + ds + di
    if total <= 0:
        return {"cpu_busy": 0.0, "rss_mb": rss_mb(), "cores": cpu_count(), "vram_note": "cpu-only laser"}
    busy = (du + ds) / total
    return {
        "cpu_busy": round(busy, 4),
        "rss_mb": round(rss_mb(), 1),
        "cores": cpu_count(),
        "vram_mb": 0.0,
        "vram_note": "pure-Python laser has no vRAM path",
    }

def _row_block_matmul(args):
    a_data, b_data, M, K, N, i0, i1 = args
    out = []
    for i in range(i0, i1):
        base_a = i * K
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a_data[base_a + k] * b_data[k * N + j]
            out.append(s)
    return i0, out

def parallel_matmul_2d(a_data, b_data, M, K, N, workers=None, min_cost=250000):
    cost = M * K * N
    workers = workers or max(1, cpu_count())
    if cost < min_cost or workers <= 1 or M < 2 * workers:
        out = []
        for i in range(M):
            base_a = i * K
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a_data[base_a + k] * b_data[k * N + j]
                out.append(s)
        return out
    tile = max(1, (M + workers - 1) // workers)
    tasks = []
    for i0 in range(0, M, tile):
        i1 = min(M, i0 + tile)
        tasks.append((a_data, b_data, M, K, N, i0, i1))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_row_block_matmul, t) for t in tasks]
        for f in as_completed(futs):
            i0, block = f.result()
            results[i0] = block
    out = []
    for i0 in sorted(results):
        out.extend(results[i0])
    return out

def apply_laser_threads(n=None):
    n = n or cpu_count()
    try:
        import laser as torch
        torch.set_num_threads(int(n))
        return torch.get_num_threads()
    except Exception as e:
        return str(e)

def optimise_process():
    notes = []
    try:
        os.nice(-5)
        notes.append("nice=-5")
    except Exception as e:
        notes.append("nice_fail:%s" % e)
    try:
        cores = list(range(cpu_count()))
        os.sched_setaffinity(0, cores)
        notes.append("affinity=%dcores" % len(cores))
    except Exception as e:
        notes.append("affinity_skip:%s" % e)
    t = apply_laser_threads()
    notes.append("laser_threads=%s" % t)
    return notes

def report():
    u = sample_utilization(0.2)
    u["optim"] = optimise_process()
    return u
