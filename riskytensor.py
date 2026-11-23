"""riskytensor Phase-2 — float32 + int8_per_row save/load.

Implements Consolidated Blueprint v4.0 Phases 0–2:
- 64-byte header, deterministic manifest, SHA-256 checks
- float32 storage (exact)
- int8_per_row storage (same per-row scale + clip math as train2.export)
- load() returns dequantized float32 Tensors
- load_raw() returns (int8 bytes, scales) for int8 tensors

No pickle / eval / exec. Parser is pure stdlib.
"""
import struct
import json
import hashlib
import os
import sys
import math

MAGIC = b"RISKY4\n"
VERSION = 4
HEADER_LEN = 64

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
_np = None
_Tensor = None

def _try_import_backends():
    global _np, _Tensor
    if _np is not None:
        return
    try:
        import bumpy as bp
        _np = bp
    except ImportError:
        _np = None
    try:
        import laser
        _Tensor = laser.Tensor
    except ImportError:
        _Tensor = None

def _as_float_list_and_shape(obj):
    """Return (flat list of float, shape tuple)."""
    _try_import_backends()
    if _Tensor is not None and isinstance(obj, _Tensor):
        arr = obj._arr
        shape = tuple(obj.shape)
        if hasattr(arr, "astype") and arr.dtype.name != "float32":
            arr = arr.astype("float32")
        return list(arr._data) if hasattr(arr, "_data") else list(arr.tolist()), shape
    if _np is not None and isinstance(obj, _np.ndarray):
        shape = tuple(obj.shape)
        data = list(obj._data) if hasattr(obj, "_data") else list(obj.tolist())
        return [float(x) for x in data], shape
    if isinstance(obj, (list, tuple)):
        flat = []
        def flatten(x):
            if isinstance(x, (list, tuple)):
                for y in x:
                    flatten(y)
            else:
                flat.append(float(x))
        flatten(obj)
        if not obj:
            shape = (0,)
        elif isinstance(obj[0], (list, tuple)):
            shape = (len(obj), len(obj[0]))
        else:
            shape = (len(obj),)
        return flat, shape
    raise TypeError("riskytensor: unsupported type %r" % type(obj))

def _from_floats(vals, shape, prefer_tensor=True):
    _try_import_backends()
    if prefer_tensor and _Tensor is not None and _np is not None:
        arr = _np.ndarray([float(v) for v in vals], tuple(shape), _np.float32)
        return _Tensor(arr, requires_grad=False)
    if len(shape) == 0:
        return vals[0] if vals else 0.0
    if len(shape) == 1:
        return list(vals)
    rows, cols = shape[0], shape[1]
    return [list(vals[i*cols:(i+1)*cols]) for i in range(rows)]

# ---------------------------------------------------------------------------
# Quantization helpers (identical math to train2.export / quantized_copy)
# ---------------------------------------------------------------------------

def _quantize_int8_per_row(flat, shape):
    """Return (blob_bytes, scales_list).
    blob layout: for each row: float32 scale + cols int8 values.
    """
    if len(shape) != 2:
        raise ValueError("int8_per_row requires 2-D tensor, got shape %s" % (shape,))
    rows, cols = shape
    scales = []
    parts = []
    for r in range(rows):
        row = flat[r*cols:(r+1)*cols]
        max_abs = max(abs(x) for x in row) if row else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        scales.append(scale)
        q = [max(-127, min(127, int(round(x / scale)))) for x in row]
        parts.append(struct.pack("<f", scale))
        parts.append(struct.pack("<%db" % cols, *q))
    return b"".join(parts), scales

def _dequantize_int8_per_row(blob, shape):
    """Return flat list of dequantized floats."""
    rows, cols = shape
    vals = []
    off = 0
    for r in range(rows):
        scale = struct.unpack_from("<f", blob, off)[0]
        off += 4
        q = struct.unpack_from("<%db" % cols, blob, off)
        off += cols
        vals.extend([qq * scale for qq in q])
    return vals

def _float32_bytes(flat):
    return struct.pack("<%df" % len(flat), *flat)

def _float32_from_bytes(data):
    n = len(data) // 4
    return list(struct.unpack("<%df" % n, data))

# ---------------------------------------------------------------------------
# Header / manifest
# ---------------------------------------------------------------------------

def info(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
    if len(header) < HEADER_LEN:
        raise ValueError("riskytensor: file too short for header")
    if header[:7] != MAGIC:
        raise ValueError("riskytensor: bad magic (expected RISKY4)")
    version = struct.unpack_from("<H", header, 7)[0]
    if version != VERSION:
        raise ValueError("riskytensor: unsupported version %d" % version)
    header_len = struct.unpack_from("<H", header, 9)[0]
    if header_len != HEADER_LEN:
        raise ValueError("riskytensor: unexpected header_len %d" % header_len)
    manifest_len = struct.unpack_from("<Q", header, 11)[0]
    manifest_sha = header[19:51]
    payload_sha = header[51:83]

    with open(path, "rb") as f:
        f.seek(HEADER_LEN)
        manifest_bytes = f.read(manifest_len)
    if len(manifest_bytes) != manifest_len:
        raise ValueError("riskytensor: truncated manifest")
    if _sha256(manifest_bytes) != manifest_sha:
        raise ValueError("riskytensor: manifest SHA-256 mismatch")

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("riskytensor: invalid manifest JSON: %s" % e) from e
    manifest["_payload_sha"] = payload_sha.hex()
    return manifest

def _write_header(manifest_bytes: bytes, payload_sha: bytes) -> bytes:
    manifest_sha = _sha256(manifest_bytes)
    header = bytearray(HEADER_LEN)
    header[0:7] = MAGIC
    struct.pack_into("<H", header, 7, VERSION)
    struct.pack_into("<H", header, 9, HEADER_LEN)
    struct.pack_into("<Q", header, 11, len(manifest_bytes))
    header[19:51] = manifest_sha
    header[51:83] = payload_sha
    return bytes(header)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(state_dict, path, recipe=None):
    """Save state_dict.

    recipe:
      None or "int8_per_row" -> 2-D tensors use int8_per_row, 1-D stay float32
      "float32"              -> everything float32
      dict[name, storage]    -> per-tensor override
    """
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict")

    default_storage = "int8_per_row"
    exceptions = {}
    if recipe == "float32":
        default_storage = "float32"
    elif isinstance(recipe, dict):
        exceptions = recipe

    tensors_meta = {}
    blobs = []
    offset = 0

    for name in sorted(state_dict.keys()):
        obj = state_dict[name]
        flat, shape = _as_float_list_and_shape(obj)

        storage = exceptions.get(name, default_storage)
        if storage == "int8_per_row" and len(shape) != 2:
            storage = "float32"          # 1-D cannot be int8_per_row

        if storage == "int8_per_row":
            data, scales = _quantize_int8_per_row(flat, shape)
            meta = {
                "shape": list(shape),
                "dtype": "float32",
                "storage": "int8_per_row",
                "scale_dtype": "float32",
                "offset": offset,
                "length": len(data),
                "compressed_length": len(data),
                "compression": "none",
                "sha256": _sha256_hex(data),
            }
        else:
            data = _float32_bytes(flat)
            meta = {
                "shape": list(shape),
                "dtype": "float32",
                "storage": "float32",
                "offset": offset,
                "length": len(data),
                "compressed_length": len(data),
                "compression": "none",
                "sha256": _sha256_hex(data),
            }

        tensors_meta[name] = meta
        blobs.append(data)
        offset += len(data)

    payload = b"".join(blobs)
    payload_sha = _sha256(payload)

    manifest = {
        "format": "riskytensor",
        "version": VERSION,
        "created_by": "riskytensor Phase-2",
        "tensors": tensors_meta,
        "quantization": {
            "default": default_storage,
            "exceptions": exceptions,
        },
        "metadata": {},
    }

    manifest_bytes = json.dumps(
        manifest, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = _write_header(manifest_bytes, payload_sha)

    with open(path, "wb") as f:
        f.write(header)
        f.write(manifest_bytes)
        f.write(payload)
    return os.path.getsize(path)

def load(path):
    """Load and return dict[str, Tensor] with dequantized float32 values."""
    manifest = info(path)
    tensors_meta = manifest.get("tensors", {})
    if not tensors_meta:
        return {}

    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        manifest_len = struct.unpack_from("<Q", header, 11)[0]
        f.seek(HEADER_LEN + manifest_len)
        payload = f.read()

    if _sha256(payload) != bytes.fromhex(manifest["_payload_sha"]):
        raise ValueError("riskytensor: payload SHA-256 mismatch")

    out = {}
    for name in sorted(tensors_meta.keys()):
        meta = tensors_meta[name]
        off = meta["offset"]
        length = meta["length"]
        data = payload[off:off+length]
        if len(data) != length:
            raise ValueError("riskytensor: truncated tensor data for %s" % name)
        if _sha256_hex(data) != meta["sha256"]:
            raise ValueError("riskytensor: SHA-256 mismatch for tensor %s" % name)

        shape = tuple(meta["shape"])
        storage = meta.get("storage", "float32")

        if storage == "float32":
            vals = _float32_from_bytes(data)
        elif storage == "int8_per_row":
            vals = _dequantize_int8_per_row(data, shape)
        else:
            raise NotImplementedError(
                "riskytensor: unsupported storage %r for %s" % (storage, name)
            )

        out[name] = _from_floats(vals, shape, prefer_tensor=True)
    return out

def load_raw(path):
    """Return dict[name, info] for debugging / re-quantization.

    For float32 tensors: {"storage": "float32", "data": bytes, "shape": tuple}
    For int8_per_row:   {"storage": "int8_per_row", "blob": bytes,
                         "shape": tuple, "scales": list[float], "int8": list[list[int]]}
    """
    manifest = info(path)
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        manifest_len = struct.unpack_from("<Q", header, 11)[0]
        f.seek(HEADER_LEN + manifest_len)
        payload = f.read()

    out = {}
    for name, meta in manifest.get("tensors", {}).items():
        off = meta["offset"]
        length = meta["length"]
        data = payload[off:off+length]
        shape = tuple(meta["shape"])
        storage = meta.get("storage", "float32")

        if storage == "float32":
            out[name] = {"storage": "float32", "data": data, "shape": shape}
        elif storage == "int8_per_row":
            rows, cols = shape
            scales = []
            int8_rows = []
            pos = 0
            for _ in range(rows):
                scale = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                q = list(struct.unpack_from("<%db" % cols, data, pos))
                pos += cols
                scales.append(scale)
                int8_rows.append(q)
            out[name] = {
                "storage": "int8_per_row",
                "blob": data,
                "shape": shape,
                "scales": scales,
                "int8": int8_rows,
            }
        else:
            out[name] = {"storage": storage, "data": data, "shape": shape}
    return out

def verify(path):
    try:
        sd = load(path)
        print("OK  %s  (%d tensors)" % (path, len(sd)))
        return True
    except Exception as e:
        print("FAIL  %s: %s" % (path, e))
        return False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 -I riskytensor.py info <file>")
        print("  python3 -I riskytensor.py verify <file>")
        print("  python3 -I riskytensor.py save-empty <file>")
        print("  python3 -I riskytensor.py roundtrip-demo <file>")
        print("  python3 -I riskytensor.py roundtrip-int8 <file>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "info" and len(sys.argv) == 3:
        m = info(sys.argv[2])
        m.pop("_payload_sha", None)
        print(json.dumps(m, indent=2))

    elif cmd == "verify" and len(sys.argv) == 3:
        sys.exit(0 if verify(sys.argv[2]) else 1)

    elif cmd == "save-empty" and len(sys.argv) == 3:
        n = save({}, sys.argv[2])
        print("wrote %d bytes" % n)

    elif cmd == "roundtrip-demo" and len(sys.argv) == 3:
        demo = {
            "emb.weight": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "nf.g": [1.0, 1.0],
        }
        n = save(demo, sys.argv[2], recipe="float32")
        print("wrote %d bytes (float32)" % n)
        loaded = load(sys.argv[2])
        print("loaded keys:", sorted(loaded.keys()))
        for k, v in loaded.items():
            print(" ", k, "->", v)
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    elif cmd == "roundtrip-int8" and len(sys.argv) == 3:
        demo = {
            "emb.weight": [[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]],
            "blocks.0.q.weight": [[1.0, -1.0], [0.5, -0.5]],
            "nf.g": [1.0, 1.0, 1.0],          # 1-D stays float32
        }
        n = save(demo, sys.argv[2])          # default = int8_per_row for 2-D
        print("wrote %d bytes (int8_per_row for 2-D)" % n)
        loaded = load(sys.argv[2])
        print("loaded keys:", sorted(loaded.keys()))
        for k, v in loaded.items():
            print(" ", k, "->", v)
        raw = load_raw(sys.argv[2])
        for k, info in raw.items():
            print(" raw", k, info["storage"],
                  ("scales=" + str(info.get("scales"))) if "scales" in info else "")
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    else:
        print("Unknown command or bad arguments")
        sys.exit(1)
