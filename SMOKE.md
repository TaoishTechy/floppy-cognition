# Smoke checks

```bash
python3 -I -m py_compile dashboard_v8.py neuro.py fmc.py util_opt.py train2.py client_v7.py
python3 -I dashboard_v8.py --useradd smoke --password smoke
python3 -I train2.py params 500kb
python3 -I train2.py params 1mb
```
