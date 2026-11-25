# floppy-cognition-neuro

Pure-Python, stdlib-only cognition stack with neuromodulatory field (SLC L0+L1),
FMC instrumentation, multi-entity user dashboard, and Symbolic Arena.

No torch, no numpy, no GPU required. `laser.py` + `bumpy.py` + `riskytensor.py`.

## Canonical layout

| Path | Role |
|------|------|
| `client_v7.py` | Terminal client (neuro History, `/util`) |
| `dashboard_v8.py` + `dashboard_v8.html` | **User UI** — login, entities, arena, metrics |
| `dashboard_v7.py` | Legacy card dashboard |
| `train2.py` | Trainer (`--cfg 500kb` / `1mb` / `base` / `8mb-1`) |
| `neuro.py` | Signal → NeurochemState + 12-domain summary |
| `neurotransmitter_module.py` | 144 heuristic equations |
| `fmc.py` | Flawless Meta-Cognition Core |
| `util_opt.py` | CPU affinity / utilization + parallel GEMM tiles |
| `ambition.py` | Desire / ambition Phase A |
| `crystal.py` / `reward.py` / `heartbeat.py` / `autotrain.py` | Memory, reward, reflect, digest |
| `models/` | `.riskytensor` checkpoints |
| `users.json` | Dashboard user store (`--useradd`) |
| `logs/` | Per-user JSONL event logs (neuro included) |

## Quickstart

```bash
# Terminal
python3 -I client_v7.py
# /load models/ling-basic-500kb.riskytensor
# /util

# User dashboard (v8)
python3 -I dashboard_v8.py --useradd alice --password secret
python3 -I dashboard_v8.py 8000
# http://127.0.0.1:8000
```

### Dashboard v8

- JSON users: `--useradd NAME --password PASS`
- Multiple **entities** per user; bind any `.riskytensor` from dropdown
- **Ask** with live neurotransmitter panel
- **Symbolic Arena** — same prompt, competing entities
- **Science metrics** dump/download (collapse rate, latency, neuro series)
- Full append-only logs: `logs/user_<name>.jsonl`

### Models

| File | Approx size | Notes |
|------|-------------|--------|
| `models/ling-basic-500kb.riskytensor` | ~528 KB | Linguistics + basic responses |
| `models/yin-creator-1mb.riskytensor` | ~1.01 MiB | Creator/Yin identity |
| `models/8mb-1.riskytensor` | ~2.6 MiB | Larger preset |

## Design notes

- Stdlib only. No pip installs.
- Neuro metrics are heuristic indices, not clinical measurements.
- Paradoxes remain display-only.
- Runtime state (`users.json`, `logs/`) is local.
