# floppy-cognition-neuro

Pure-Python, stdlib-only cognition stack with neuromodulatory field (SLC L0+L1),
FMC instrumentation, multi-entity user dashboard, Symbolic Arena, and BBS ops plane.

No torch, no numpy, no GPU required. `laser.py` + `bumpy.py` + `riskytensor.py`.

## BBS operator shell (48 approaches)

```bash
bash floppy-bbs.sh                 # interactive menus
bash floppy-bbs.sh --cmd selftest
bash floppy-bbs.sh --cmd health
bash floppy-bbs.sh --cmd models
bash floppy-bbs.sh --cmd train-ling
bash floppy-bbs.sh --cmd dash-start 8000
bash floppy-bbs.sh --cmd arena
bash floppy-bbs.sh --cmd metrics creator
```

See `docs/BBS-48.md` for the full approach table.

## Canonical layout

| Path | Role |
|------|------|
| `floppy-bbs.sh` | Science-grade BBS terminal control plane |
| `client_v7.py` | Terminal client (neuro History, `/util`) |
| `dashboard_v8.py` + `dashboard_v8.html` | **User UI** — login, entities, arena, metrics |
| `dashboard_v7.py` | Legacy card dashboard |
| `train2.py` | Trainer (`--cfg 500kb` / `1mb` / `base` / `8mb-1`) |
| `neuro.py` | Signal → NeurochemState + 12-domain summary |
| `neurotransmitter_module.py` | 144 heuristic equations |
| `fmc.py` | Flawless Meta-Cognition Core |
| `util_opt.py` | CPU affinity / utilization + parallel GEMM tiles |
| `models/` | `.riskytensor` checkpoints |
| `users.json` | Dashboard user store (`--useradd`) |
| `logs/` | Per-user JSONL event logs (neuro included) |

## Quickstart

```bash
# Terminal
python3 -I client_v7.py
# /load models/ling-basic-500kb.riskytensor

# User dashboard (v8)
python3 -I dashboard_v8.py --useradd alice --password secret
python3 -I dashboard_v8.py 8000
```

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
