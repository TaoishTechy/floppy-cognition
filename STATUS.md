# floppy-cognition status

**Repository:** https://github.com/TaoishTechy/floppy-cognition

## Files present on main

| File | Approx size | Status |
|------|-------------|--------|
| README.md | 2 KB | Complete |
| LICENSE | 1 KB | MIT |
| bumpy.py | 37 KB | Full |
| train2.py | 15 KB | Full |
| laser.py | marker only | Full source still local |
| gen2.py | ~9 KB | Finished |
| riskytensor.py | ~15 KB | **Phase-2 complete** (float32 + int8_per_row + load_raw) |
| STATUS.md | this file | Current |
| docs/ | | Blueprint summaries |

## riskytensor capabilities

- Phase 0: header + manifest + info/verify
- Phase 1: float32 exact round-trip
- Phase 2: int8_per_row (same math as train2.export) + load_raw()

```bash
python3 -I riskytensor.py roundtrip-demo demo.f32.riskytensor
python3 -I riskytensor.py roundtrip-int8 demo.i8.riskytensor
python3 -I riskytensor.py info demo.i8.riskytensor
python3 -I riskytensor.py verify demo.i8.riskytensor
```

## Still required for a full train run

Replace the laser.py marker with the complete 49 KB implementation.

*Last updated after riskytensor Phase-2.*
