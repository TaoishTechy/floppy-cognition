# Model checkpoints

| File | Size | Config | Focus |
|------|------|--------|--------|
| `models/ling-basic-500kb.riskytensor` | ~528 KB | D112 L4 H4 HID320 | Linguistics + basic responses |
| `models/yin-creator-1mb.riskytensor` | ~1.01 MiB | D128 L6 H4 HID384 | Creator user / Yin entity |
| `models/8mb-1.riskytensor` | ~2.6 MiB | D192 L6 H6 HID768 | Larger general |

## Train 500kb linguistics

```bash
python3 -I train2.py train 20 models/ling-basic-500kb.riskytensor \\
  --cfg 500kb --resume models/ling-basic-500kb.riskytensor \\
  --bs 2 --identity-boost 0.05 --threads 1
```

## Train 1mb Creator/Yin

```bash
python3 -I train2.py train 30 models/yin-creator-1mb.riskytensor \\
  --cfg 1mb --resume models/yin-creator-1mb.riskytensor \\
  --bs 2 --identity-boost 0.6 --threads 1
```

Binary `.riskytensor` files are distributed as release artifacts (not always in git).
