# floppy-bbs transport

The full operator shell is **`floppy-bbs.sh`** (science-grade BBS, 48 approaches).

## From a release / local tree

```bash
bash floppy-bbs.sh
bash floppy-bbs.sh --cmd selftest
bash floppy-bbs.sh --cmd health
bash floppy-bbs.sh --cmd dash-start 8000
```

## Docs

- `docs/BBS-48.md` — all 48 approaches
- `STATUS.md` — current push status

## Note on git transport

Large shell payloads are shipped via the project release zip (`floppy-cognition-neuro-REPO.zip`) when API push limits apply. Keep `floppy-bbs.sh` next to `train2.py` / `dashboard_v8.py`.
