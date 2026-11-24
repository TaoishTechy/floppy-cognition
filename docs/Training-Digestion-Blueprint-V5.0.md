# Training Digestion Blueprint V5.0

Meta-cognitive ingestion for `/training/` — `.md` `.json` `.xml` `.pdf`

**Version:** 5.0  
**Date:** 2026-11-23  
**Project:** floppy-cognition  
**Constraint:** Tiny offline model. Digestion can be heavier offline. Output fits the floppy contract.

See repository root conversation / full blueprint text for the complete 48 approaches, schemas, audits, and roadmap.

## Quick summary

1. Parse `/training/{md,json,xml,pdf}` → `RawUnit`
2. Digest with core approaches A1, B7, F36, G37, G38 → `DigestedUnit`
3. Load digested JSON into `gen2.py`
4. Train with `train2.py`
5. Run four audits: recall, transfer, abstention, calibration

## Next action

```bash
python3 -I training/ingest.py scan
python3 -I training/ingest.py parse training/md/example.md
python3 -I training/digest.py --limit 20
```
