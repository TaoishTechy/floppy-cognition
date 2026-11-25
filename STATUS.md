# STATUS — 2026-11-25 neuro / arena / models

## Pushed this cycle
- README.md (v8 layout)
- util_opt.py (CPU util + parallel GEMM tiles)
- STATUS.md / docs/V8-DASHBOARD.md

## Local tree (release zips; remaining source commits as capacity allows)
- dashboard_v8.py + dashboard_v8.html — user JSON auth, multi-entity, Symbolic Arena, metrics export
- neuro.py — Signal→NeurochemState, never bare None on failure
- fmc.py — monitor/control instrumentation
- train2.py — cfg: tiny, base, 500kb, 1mb, 8mb-*; Creator/Yin + linguistics
- client_v7.py — neuro History, /util
- bumpy.py — optional process-parallel large matmul
- models/ling-basic-500kb.riskytensor (~528 KB)
- models/yin-creator-1mb.riskytensor (~1.01 MiB)

## Run
```bash
python3 -I dashboard_v8.py --useradd NAME --password PASS
python3 -I dashboard_v8.py 8000
python3 -I client_v7.py
```

## Binary models
`.riskytensor` checkpoints are binary; prefer release artifacts / Git LFS rather than embedding in every commit.
