# Floppy-Cognition BBS — 48 approaches

Science-grade terminal control plane (`floppy-bbs.sh`).

| # | Approach | Implementation |
|---|----------|----------------|
| 01 | Deterministic session identity | `SESSION_ID` UTC+pid |
| 02 | ANSI semantic palette | Phosphor-style 256-color roles |
| 03 | JSONL event bus | `logs/bbs/session-*.jsonl` |
| 04 | Adaptive telemetry header | cores / RAM / model count |
| 05 | Pure menu registry | `bbs_register` arrays |
| 06 | Isolated Python runner | `python3 -I` + `PYTHONPATH` |
| 07 | Model lattice inventory | size + meta.json |
| 08 | Health / gate probes | import + py_compile |
| 09 | Neuro field smoke | collapse vs clean |
| 10 | FMC pure monitor/control | determinism probe |
| 11 | Cfg params lattice | train2 params |
| 12 | Segmented train launcher | resume-safe |
| 13 | Identity audit | train2 audit-identity |
| 14 | Dashboard lifecycle | pidfile start/stop |
| 15 | JSON user provisioning | dashboard_v8 --useradd |
| 16 | Science metrics export | export_metrics JSON |
| 17 | Symbolic arena sim | multi-model same prompt |
| 18 | Client REPL entry | client_v7 |
| 19 | Util optimise | util_opt.report |
| 20 | Forensic log tail | session + user JSONL |
| 21 | Reproducibility hash | SHA256 of `*.py` |
| 22 | Runtime GC | pyc / locks |
| 23 | Release packaging | zip |
| 24 | Paradox display-only | never train |
| 25 | Heartbeat reflect | module check |
| 26 | Crystal memory | module + db |
| 27 | Reward ledger | module + jsonl |
| 28 | Ambition/desire phase | module check |
| 29 | Continuous eval | train2 eval |
| 30 | Entity model bind | users.json |
| 31 | Port/process sentinel | ss/netstat + pidfile |
| 32 | Self-test battery | chained probes |
| 33 | Util watch mode | sampled busy fraction |
| 34 | Panic stop | kill train/dash |
| 35 | Approaches index | this list |
| 36 | MOGOPs size gates | file bytes vs target |
| 37 | Collapse-risk triage | log aggregate |
| 38 | Linguistics curriculum | 500kb train preset |
| 39 | Creator/Yin identity lock | 1mb train preset |
| 40 | Multi-cfg compare | MEASURED params |
| 41 | JSONL science notebook | markdown dump |
| 42 | One-shot CLI mode | `--cmd` |
| 43 | Non-interactive once | `--once` alias |
| 44 | Soft-fail operators | menu never aborts shell |
| 45 | Path sandbox | always `cd ROOT` |
| 46 | Pidfile dashboard | single instance |
| 47 | Metrics directory mirror | `logs/bbs/metrics/` |
| 48 | Operator UX | classic BBS numbered menus |

## Usage

```bash
bash floppy-bbs.sh
bash floppy-bbs.sh --cmd selftest
bash floppy-bbs.sh --cmd train-ling
bash floppy-bbs.sh --cmd dash-start 8000
bash floppy-bbs.sh --cmd metrics creator
```
