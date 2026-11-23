# riskytensor — Blueprint v4.0 (summary)

**Status:** Blueprint only, not yet built.
**Stack:** stdlib only + bumpy + laser. No torch, no numpy, no safetensors, no LZ4, no AES.
**Replaces:** The v1.0 safe2risky blueprint and the v3.0 NecroScale upgrade — same goal (a safe, portable tensor file), rewritten without the fake physics.

## What it is

A single-file format + stdlib-only reader/writer for model weights.

- One `.riskytensor` file holds a full state_dict: names → tensors, plus a small JSON manifest.
- It replaces `torch.save` / `torch.load` (pickle) and `safetensors` for weight exchange.
- It does **not** replace `TINY.BIN`. TINY.BIN stays byte-identical, with its own exporter in train2.py.

## File format (v4.0)

Header (64 bytes) + Manifest (JSON) + Payload (tensor blobs).

- Magic: `b"RISKY4\n"`
- Per-tensor SHA-256, payload SHA-256, deterministic sorted keys.
- Storage: `float32` (exact) or `int8_per_row` (same math as train2.export).
- Compression: `none` or `zlib` (per-tensor).

## API surface

```python
riskytensor.save(state_dict, path, recipe=None)
state_dict = riskytensor.load(path)          # returns dict[str, laser.Tensor]
info = riskytensor.info(path)               # manifest only
riskytensor.convert_safetensors(src, out)
```

## Build order (from Consolidated Blueprint §9)

- Phase 0: header + manifest + info()
- Phase 1: float32 save/load
- Phase 2: int8_per_row + load_raw
- Phase 3: zlib + verify CLI
- Phase 4: convert_safetensors
- Phase 5: train2 integration

Do not start later phases before earlier ones pass the §10 conformance suite.

Full specification lives in the Consolidated Blueprint and the original Riskytensor-Blueprint-V4-0.md.
