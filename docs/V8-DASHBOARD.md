# Dashboard v8

User-aware UI with JSON auth, multi-entity binding, Symbolic Arena, and science metrics export.

## Create user
```bash
python3 -I dashboard_v8.py --useradd alice --password secret
```

## Run
```bash
python3 -I dashboard_v8.py 8000
```

## Features
- Entities with per-entity `.riskytensor` binding
- Ask + neurotransmitter panel
- Symbolic Arena (multi-entity same prompt)
- `/api/metrics` dump/download
- Logs: `logs/user_<name>.jsonl`
