# Installing the full laser.py

The complete torch-replacement (`laser.py`, ~49 KB / 1132 lines) is larger than the single-file push limit of the current tooling, so it ships as a local artifact.

## Quick install

```bash
# From the machine that has the original conversation artifacts:
cp /home/workdir/artifacts/laser.py ./laser.py

# Verify
wc -l laser.py                    # expect ~1132
grep -c "class AdamW" laser.py    # expect >= 1
grep -c "class Module" laser.py   # expect >= 1
```

The file must begin with:

```python
"""laser v4.0 - torch replacement, stdlib + bumpy only. No torch import."""
```

and end with an `__all__` list that includes `"nn"`, `"optim"`, `"Tensor"`, etc.

After the full file is in place:

```bash
python3 -I train2.py train 0.1 smoke.pt --threads 4
```

should progress past the banner and start reporting steps / loss / tok/s.
