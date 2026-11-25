# STATUS — 2026-11-25 neuro / arena / models / BBS

## Pushed
- README.md (v8 layout + BBS)
- util_opt.py
- STATUS.md / docs/V8-DASHBOARD.md / docs/BBS-48.md
- floppy-bbs.sh — 48-approach science-grade BBS terminal
- training/MODELS.md
- models/*.meta.json

## Run BBS
```bash
bash floppy-bbs.sh
bash floppy-bbs.sh --cmd selftest
bash floppy-bbs.sh --cmd dash-start 8000
```

## Dashboard v8
```bash
python3 -I dashboard_v8.py --useradd NAME --password PASS
python3 -I dashboard_v8.py 8000
```

## Models
- ling-basic-500kb (~528 KB) linguistics + basic responses
- yin-creator-1mb (~1.01 MiB) Creator/Yin
- 8mb-1 (~2.6 MiB)

Binary `.riskytensor` files: use release zip / Git LFS when pushing weights.
