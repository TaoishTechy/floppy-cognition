# Smoke-test instructions (pure-Python laser)

The laser backend is pure Python. Large batches take minutes per step and only use ~1 core (GIL).

## 1. Make sure you have the latest trainer

```bash
cd ~/Desktop/floppy-cognition-main
git pull
# OR copy the latest train2.py from the repo
```

The banner **must** show:
```
batch    4 (pure-Python laser: keep small)
NOTE     pure-Python backend ~minutes/step at large bs
```
If you still see only `model D=96 ... parameters 366528` with no `batch` line, you are on an old train2.py.

## 2. Run a minimal smoke

```bash
python3 -I train2.py train 0.05 smoke.pt --bs 2 --log-every 1 --threads 1
```

Expected: a line after **each** step within ~30–90s:
```
step    1 |    45s | lr 0.00004 | loss 4.xxxx | 45.0s/step | ETA ...
```

## 3. If it still hangs with no step line

- Confirm laser is the full 49KB file: `wc -l laser.py` → ~1132
- Confirm train2 is new: `grep -n "pure-Python laser" train2.py`
- Try even smaller:
```bash
python3 -I train2.py train 0.03 smoke.pt --bs 1 --log-every 1
```

## Why 6% CPU?

CPython’s GIL allows only one thread to execute Python bytecode at a time. laser matmuls are pure Python loops, so extra threads do not help and RSS stays modest (~0.5–1 GB).
