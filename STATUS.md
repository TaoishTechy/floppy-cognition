# floppy-cognition status

**Repository:** https://github.com/TaoishTechy/floppy-cognition

## Files present on main

| File | Approx size | Status |
|------|-------------|--------|
| README.md | 2 KB | Complete |
| LICENSE | 1 KB | MIT |
| bumpy.py | 37 KB | Full numpy-replacement committed |
| train2.py | 15 KB | Full model + train + eval + export committed |
| laser.py | ~1 KB (marker) | Full 49 KB source ready locally; full push still pending tool size limits |
| gen2.py | ~1 KB | Minimal stub committed (enough for import + sample) |
| riskytensor.py | ~3 KB | Phase-0 skeleton committed (header + manifest + info/verify/save-empty) |
| STATUS.md | this file | Current |
| docs/ | | Blueprint summaries |

## Size check

All committed content remains well under the 1.4 MB hard limit.

## What works right now

- `python3 -I riskytensor.py save-empty test.riskytensor`
- `python3 -I riskytensor.py info test.riskytensor`
- `python3 -I riskytensor.py verify test.riskytensor`
- `python3 -I gen2.py` (prints a few sample facts)

## Still required for a full train run

1. Replace the laser.py marker with the complete 49 KB implementation (local copy is ready).
2. After that, `python3 -I train2.py train 0.1 smoke.pt` should progress past the import stage.

*Last updated after gen2 + riskytensor Phase-0 commit.*
