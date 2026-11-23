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
| riskytensor.py | ~16 KB | **Phase-3 complete** (float32 + int8_per_row + zlib) |
| STATUS.md | this file | Current |
| docs/ | | Blueprint summaries |

## riskytensor capabilities

| Phase | Feature |
|-------|--------|
| 0 | header + manifest + info/verify |
| 1 | float32 exact round-trip |
| 2 | int8_per_row (train2.export math) + load_raw |
| 3 | optional per-tensor zlib compression |

```bash
python3 -I riskytensor.py roundtrip-demo demo.f32.riskytensor
python3 -I riskytensor.py roundtrip-int8 demo.i8.riskytensor
python3 -I riskytensor.py roundtrip-zlib demo.z.riskytensor
python3 -I riskytensor.py verify demo.z.riskytensor
```

## Still required for a full train run

Replace the laser.py marker with the complete 49 KB implementation.

*Last updated after riskytensor Phase-3.*
