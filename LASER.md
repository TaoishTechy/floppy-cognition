# laser.py status

The GitHub tree may contain a **placeholder** (~6 KB).

Full laser (~49 KB, ~1132 lines) is required for training.

```bash
# after obtaining the full file:
cp /path/to/full/laser.py ./laser.py
wc -l laser.py          # expect ~1132
grep -n "class AdamW" laser.py
grep -n "scaled_dot_product_attention" laser.py
```

Required symbols: `Tensor`, `nn.Module`, `nn.Linear`, `nn.Embedding`, `AdamW`,
`scaled_dot_product_attention`, `save` / `load`.

Without the full file, `train2.py` fails with:
`AttributeError: module 'laser' has no attribute 'nn'`.
