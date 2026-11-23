# floppy-cognition status

**Repository:** https://github.com/TaoishTechy/floppy-cognition

## Files on main

| File | Status |
|------|--------|
| bumpy.py | Full |
| train2.py | Full + **rich logging** (tok/s, ETA, sample Q&A) |
| gen2.py | Finished multi-category |
| riskytensor.py | Phase-3 + **convert .pt → .riskytensor** |
| client.py | **New** interactive test client |
| laser.py | Still placeholder (full source local) |
| STATUS.md | Current |

## New capabilities

```bash
# Convert laser checkpoint to riskytensor
python3 -I riskytensor.py convert model.pt model.riskytensor [--float32] [--compress]

# Interactive client
python3 -I client.py model.pt
python3 -I client.py --info model.riskytensor
python3 -I client.py --verify model.riskytensor
python3 -I client.py --convert model.pt

# Trainer now prints:
#   step, elapsed, lr, loss, best, tokens/s, ETA
#   periodic live sample Q → A
```

## Still required for a full train run

Replace the laser.py marker with the complete 49 KB implementation.

*Last updated after convert + client + richer trainer.*
