# floppy-cognition

**Proto Stage 3–4 cognition core that still fits on a floppy.**

A 367 k-parameter letter-by-letter transformer that:
- runs on plain DOS (Turbo C 2.0, 486/Pentium era hardware),
- needs ~540 KB conventional memory,
- answers questions from ~3 500 facts in many wordings,
- exports a 380 KB `TINY.BIN` weight file,
- is trained and evaluated with a pure-stdlib stack (no torch, no numpy).

The entire deliverable (code + weights + documentation) is kept under the hard 1.4 MB budget.

## Goal

Outperform Claude-class models on the exact Tiny held-out fact set while remaining three orders of magnitude smaller and fully offline.

## Stack (stdlib only)

| Module | Role | Status |
|--------|------|--------|
| `bumpy.py` | numpy replacement | committed |
| `laser.py` | torch replacement (Tensor + autograd + nn + AdamW) | source ready locally; full commit pending |
| `riskytensor.py` | safetensors / torch.save replacement | blueprint only |
| `train2.py` | model, training, eval, TINY.BIN export | committed |

## Quick start

```bash
# Bench three model sizes
python3 -I train2.py bench

# Train a few minutes (uses laser + bumpy)
python3 -I train2.py train 0.1 smoke.pt

# Evaluate
python3 -I train2.py eval smoke.pt

# Export the DOS weight file
python3 -I train2.py export smoke.pt TINY.BIN
```

Copy `TINY.EXE` + `TINY.BIN` to a DOS machine (or emulator) and run.

## Size contract

- Blueprint document < 100 KB (hard limit 1.4 MB)
- `bumpy.py` < 40 KB
- `laser.py` < 55 KB
- `train2.py` < 16 KB
- Full model weights (int8 per-row) ~ 380 KB

All numbers are measured, not claimed.

## Current repository state

See [STATUS.md](STATUS.md) for the exact commit inventory and next actions.

## Licence

Code: MIT.  
Three derived data files keep their original licences (see DATA_LICENSES.md in the upstream Tiny tree).

---

*Named floppy-cognition because it still fits on a 1.44 MB floppy and still answers questions.*
