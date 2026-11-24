# STATUS — floppy-cognition

**Updated:** 2026-11-23

## Inventory

| Path | State | Notes |
|------|-------|-------|
| `bumpy.py` | present | stdlib ndarray; matmul float fast-path may need local copy if older |
| `laser.py` | **check size** | full ~49KB / ~1132 lines required; repo may still hold placeholder |
| `train2.py` | current | `--tiny`, warmup=5, real tokens, prompt truncate |
| `gen2.py` | current | canonical answers + `load_digested()` |
| `riskytensor.py` | current | HEADER_LEN=88; convert/verify/load |
| `client.py` | current | progress dots, truncate, echo fed prompt |
| `audit.py` | current | 4 audits |
| `training/ingest.py` | Phase 0 | md/json/xml/pdf |
| `training/digest.py` | Phase 1 | core approaches |
| `training/md/example.md` | sample | capitals |
| `neurotransmitter_module.py` | optional | SPECULATIVE 144 eqs |
| `SMOKE.md` | ops | pure-Python gotchas |
| `README.md` | current | honest claims |

## Working path

1. Full `laser.py` locally  
2. `train2.py train … --tiny --bs 4`  
3. `riskytensor.py convert` + `verify`  
4. `client.py` / `audit.py`  
5. Digestion: ingest → digest → `gen2.reload()` → longer train  

## Known issues

- Pure-Python speed: ~seconds/token; GIL ≈ 1 core  
- Short trains → attractor collapse (`eeee` / `....`)  
- Calibration audit needs temperature/top-k (not only greedy)  
- README size/Claude claims from earlier drafts are **not** validated here  

## Next (priority)

1. Ensure full `laser.py` + optimized `bumpy` matmul in tree  
2. Train ≥30–60 min on digested canonical facts (`--tiny` first)  
3. Pass recall audit on exact `what is the capital of france`  
4. Then transfer / abstention  
5. Optional: wire `training_signals()` only after recall moves  
