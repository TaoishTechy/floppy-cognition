# floppy-cognition status

**Repository:** https://github.com/TaoishTechy/floppy-cognition

## Files present on main

| File | Approx size | Status |
|------|-------------|--------|
| README.md | 2 KB | Complete |
| LICENSE | 1 KB | MIT |
| bumpy.py | 37 KB | Full numpy-replacement |
| train2.py | 15 KB | Full model + train + eval + export |
| laser.py | ~1 KB (marker) | Full 49 KB source ready locally |
| gen2.py | ~9 KB | **Finished** – multi-category generator (geo, math, animals, language, science, facts, unknown) with varied wordings |
| riskytensor.py | ~3 KB | Phase-0 skeleton (header + manifest + info/verify) |
| STATUS.md | this file | Current |
| docs/ | | Blueprint summaries |

## Size check

All committed content remains well under the 1.4 MB hard limit.

## What works right now

```bash
python3 -I gen2.py                  # prints sample facts
python3 -I riskytensor.py save-empty test.riskytensor
python3 -I riskytensor.py info test.riskytensor
```

## Still required for a full train run

Replace the laser.py marker with the complete 49 KB implementation. After that:

```bash
python3 -I train2.py train 0.1 smoke.pt
```

should progress past imports and begin training on the gen2 fact pools.

*Last updated after finished gen2.py.*
