# STATUS

## BBS install (important)

If `./floppy-bbs-install.sh` produced a **tiny** `floppy-bbs.sh` (~720 bytes) and running it does nothing, the GitHub **part files were incomplete** (API size limits during push).

### Fix

1. Use the full script from the release zip / local build (must be **~25KB**, not <1KB).
2. Or after a complete multi-part push lands, re-run:

```bash
chmod +x ./floppy-bbs-install.sh
./floppy-bbs-install.sh
# must print: OK floppy-bbs.sh 25xxx bytes
bash floppy-bbs.sh --cmd health
```

### Working commands once script is complete

```bash
bash floppy-bbs.sh --cmd health
bash floppy-bbs.sh --cmd selftest
bash floppy-bbs.sh --cmd models
bash floppy-bbs.sh
```

Docs: `docs/BBS-48.md`
