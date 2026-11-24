# Smoke-test (pure-Python laser)

## Prerequisites

```bash
wc -l laser.py    # need ~1132 (full), not ~150 (placeholder)
grep -n "class AdamW" laser.py
grep -n "pure-Python laser" train2.py
```

If `laser.py` is a placeholder, copy the full local file before training.

## Minimal train

```bash
python3 -I train2.py train 2 tiny.pt --tiny --bs 2 --log-every 1 --seed 0
```

Banner must show `batch 2` (or 4) and model `D=48 L=2` when using `--tiny`.

## Convert / verify / client

```bash
python3 -I riskytensor.py convert tiny.pt tiny.riskytensor
python3 -I riskytensor.py verify tiny.riskytensor
python3 -I client.py
> /load tiny.riskytensor
> hello
```

Expect `generating....` progress. Empty or `eeee`/`....` is normal until longer train.

## Digestion smoke

```bash
python3 -I training/ingest.py parse training/md/example.md
python3 -I training/digest.py --limit 20
python3 -I gen2.py
```

## Why 6% CPU?

CPython GIL: laser matmuls are pure Python → one core. Extra threads do not speed the hot path.
