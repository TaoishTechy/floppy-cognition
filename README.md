# floppy-cognition

**Stdlib-only** proto cognition stack (char-level tiny transformer).
Target envelope: under ~1.4 MB of project code + weights on the critical path.

> Pure-Python `laser` + `bumpy` are correct but **slow** (seconds per step/token).
> Use `--tiny` and small `--bs` for smoke tests.

## Stack

| File | Role |
|------|------|
| `bumpy.py` | ndarray / matmul (stdlib) |
| `laser.py` | Tensor + autograd + nn + AdamW (**need full ~49KB local copy if repo has placeholder**) |
| `train2.py` | train / eval / export |
| `gen2.py` | facts + `load_digested()` |
| `riskytensor.py` | RISKY4 container, convert `.pt` → `.riskytensor` |
| `client.py` | interactive load / generate |
| `audit.py` | recall / transfer / abstention / calibration |
| `training/ingest.py` | parse md/json/xml/pdf → RawUnit |
| `training/digest.py` | DigestedUnit (A1,B7,F36,G37,G38,G41) |
| `neurotransmitter_module.py` | **SPECULATIVE** 144-heuristic toy model (not clinical) |

## Quick start

```bash
git clone https://github.com/TaoishTechy/floppy-cognition.git
cd floppy-cognition

# If laser.py is a placeholder (~6KB), replace with the full local laser (~1132 lines)
wc -l laser.py   # expect ~1100+

# Smoke train (tiny model)
python3 -I train2.py train 5 tiny.pt --tiny --bs 4 --log-every 1 --seed 0

# Convert + client
python3 -I riskytensor.py convert tiny.pt tiny.riskytensor
python3 -I riskytensor.py verify tiny.riskytensor
python3 -I client.py
# > /load tiny.riskytensor
# > what is the capital of france

# Digestion pipeline
python3 -I training/ingest.py parse training/md/example.md
python3 -I training/digest.py --limit 20
python3 -I gen2.py   # should report source: digested if units exist

# Four audits (slow on pure-Python)
python3 -I audit.py tiny.riskytensor --n 5

# Speculative neurochem self-test
python3 -I neurotransmitter_module.py
```

## Train flags

```
python3 -I train2.py train MINUTES OUT.pt [--tiny] [--bs 4] [--seed N] [--threads N] [--lr 0.002]
```

- `--tiny` → `D=48 L=2 H=2 HID=96` (fast iteration)
- default model → `D=96 L=4 H=4 HID=256` (~367k params)
- Warmup is **5 steps** (not 50)
- Token/s uses **real** answer-mask counts

## Honest status

| Claim | Reality |
|-------|---------|
| Pipeline train→convert→load→generate | Works |
| Fact recall after minutes of pure-Python train | Usually still collapsed (`e` / `.`) |
| 3500 facts / beat Claude | **Not measured** — do not claim |
| DOS 540KB runtime | **Not verified** in this tree |
| `neurotransmitter_module` | Toy / SPECULATIVE only |

See `STATUS.md` and `SMOKE.md`.

## Docs

- `docs/Training-Digestion-Blueprint-V5.0.md`
- Riskytensor / consolidated blueprints under `docs/` when present

## License

See `LICENSE`.
