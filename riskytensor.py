"""riskytensor Phase-3 — float32 + int8_per_row + optional zlib.

Implements Consolidated Blueprint v4.0 Phases 0–3:
- 64-byte header, deterministic manifest, SHA-256 checks
- float32 (exact) and int8_per_row (train2.export math) storage
- optional per-tensor zlib compression
- payload SHA is over *decompressed* bytes
- load() returns dequantized float32; load_raw() exposes scales / int8

No pickle / eval / exec. Parser is pure stdlib (+ zlib).
"""
import struct
import json
import hashlib
import os
import sys
import zlib

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
    _try_import_backends()
    if _Tensor is not None and isinstance(obj, _Tensor):
        arr = obj._arr
        shape = tuple(obj.shape)
        if hasattr(arr, "astype") and getattr(arr.dtype, "name", "") != "float32":
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
# Quantization (identical to train2.export)
# ---------------------------------------------------------------------------

def _quantize_int8_per_row(flat, shape):
    if len(shape) != 2:
        raise ValueError("int8_per_row requires 2-D tensor, got shape %s" % (shape,))
    rows, cols = shape
    parts = []
    for r in range(rows):
        row = flat[r*cols:(r+1)*cols]
        max_abs = max(abs(x) for x in row) if row else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        q = [max(-127, min(127, int(round(x / scale)))) for x in row]
        parts.append(struct.pack("<f", scale))
        parts.append(struct.pack("<%db" % cols, *q))
    return b"".join(parts)

def _dequantize_int8_per_row(blob, shape):
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
# Compression helpers
# ---------------------------------------------------------------------------

def _maybe_compress(raw: bytes, use_zlib: bool):
    """Return (stored_bytes, compression_name, raw_length)."""
    if not use_zlib:
        return raw, "none", len(raw)
    compressed = zlib.compress(raw, level=6)
    # Only keep compression if it actually helps
    if len(compressed) < len(raw):
        return compressed, "zlib", len(raw)
    return raw, "none", len(raw)

def _decompress(stored: bytes, compression: str, raw_length: int) -> bytes:
    if compression == "none":
        return stored
    if compression == "zlib":
        raw = zlib.decompress(stored)
        if len(raw) != raw_length:
            raise ValueError("riskytensor: decompressed length mismatch")
        return raw
    raise ValueError("riskytensor: unknown compression %r" % compression)

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

def save(state_dict, path, recipe=None, compress=False):
    """Save state_dict.

    recipe:
      None / "int8_per_row" -> 2-D use int8_per_row, 1-D stay float32
      "float32"             -> everything float32
      dict[name, storage]   -> per-tensor override

    compress: if True, try zlib on each tensor blob (kept only when smaller).
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
    stored_blobs = []          # what is written to the payload
    raw_for_hash = []          # decompressed bytes (for payload SHA)
    offset = 0

    for name in sorted(state_dict.keys()):
        obj = state_dict[name]
        flat, shape = _as_float_list_and_shape(obj)

        storage = exceptions.get(name, default_storage)
        if storage == "int8_per_row" and len(shape) != 2:
            storage = "float32"

        if storage == "int8_per_row":
            raw = _quantize_int8_per_row(flat, shape)
        else:
            raw = _float32_bytes(flat)

        stored, compression, raw_len = _maybe_compress(raw, compress)

        meta = {
            "shape": list(shape),
            "dtype": "float32",
            "storage": storage,
            "offset": offset,
            "length": raw_len,                 # decompressed length
            "compressed_length": len(stored),
            "compression": compression,
            "sha256": _sha256_hex(raw),        # SHA of decompressed data
        }
        if storage == "int8_per_row":
            meta["scale_dtype"] = "float32"

        tensors_meta[name] = meta
        stored_blobs.append(stored)
        raw_for_hash.append(raw)
        offset += len(stored)

    payload = b"".join(stored_blobs)
    # Blueprint: payload_sha is over the *decompressed* concatenation
    payload_sha = _sha256(b"".join(raw_for_hash))

    manifest = {
        "format": "riskytensor",
        "version": VERSION,
        "created_by": "riskytensor Phase-3",
        "tensors": tensors_meta,
        "quantization": {
            "default": default_storage,
            "exceptions": exceptions,
        },
        "metadata": {"compress": bool(compress)},
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

    # Reconstruct decompressed payload for the global SHA check
    raw_parts = []
    out = {}
    for name in sorted(tensors_meta.keys()):
        meta = tensors_meta[name]
        off = meta["offset"]
        comp_len = meta["compressed_length"]
        stored = payload[off:off+comp_len]
        if len(stored) != comp_len:
            raise ValueError("riskytensor: truncated tensor data for %s" % name)

        raw = _decompress(stored, meta.get("compression", "none"), meta["length"])
        if _sha256_hex(raw) != meta["sha256"]:
            raise ValueError("riskytensor: SHA-256 mismatch for tensor %s" % name)
        raw_parts.append(raw)

        shape = tuple(meta["shape"])
        storage = meta.get("storage", "float32")

        if storage == "float32":
            vals = _float32_from_bytes(raw)
        elif storage == "int8_per_row":
            vals = _dequantize_int8_per_row(raw, shape)
        else:
            raise NotImplementedError(
                "riskytensor: unsupported storage %r for %s" % (storage, name)
            )
        out[name] = _from_floats(vals, shape, prefer_tensor=True)

    # Global payload SHA (over decompressed concatenation)
    if _sha256(b"".join(raw_parts)) != bytes.fromhex(manifest["_payload_sha"]):
        raise ValueError("riskytensor: payload SHA-256 mismatch")

    return out

def load_raw(path):
    """Return per-tensor raw info (after decompression)."""
    manifest = info(path)
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        manifest_len = struct.unpack_from("<Q", header, 11)[0]
        f.seek(HEADER_LEN + manifest_len)
        payload = f.read()

    out = {}
    for name, meta in manifest.get("tensors", {}).items():
        off = meta["offset"]
        comp_len = meta["compressed_length"]
        stored = payload[off:off+comp_len]
        raw = _decompress(stored, meta.get("compression", "none"), meta["length"])
        shape = tuple(meta["shape"])
        storage = meta.get("storage", "float32")

        entry = {
            "storage": storage,
            "shape": shape,
            "compression": meta.get("compression", "none"),
            "raw": raw,
        }
        if storage == "int8_per_row":
            rows, cols = shape
            scales, int8_rows = [], []
            pos = 0
            for _ in range(rows):
                scale = struct.unpack_from("<f", raw, pos)[0]
                pos += 4
                q = list(struct.unpack_from("<%db" % cols, raw, pos))
                pos += cols
                scales.append(scale)
                int8_rows.append(q)
            entry["scales"] = scales
            entry["int8"] = int8_rows
        out[name] = entry
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
        print("  python3 -I riskytensor.py roundtrip-zlib <file>")
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
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    elif cmd == "roundtrip-int8" and len(sys.argv) == 3:
        demo = {
            "emb.weight": [[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]],
            "blocks.0.q.weight": [[1.0, -1.0], [0.5, -0.5]],
            "nf.g": [1.0, 1.0, 1.0],
        }
        n = save(demo, sys.argv[2])
        print("wrote %d bytes (int8_per_row)" % n)
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    elif cmd == "roundtrip-zlib" and len(sys.argv) == 3:
        demo = {
            "emb.weight": [[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]] * 20,
            "blocks.0.q.weight": [[1.0, -1.0], [0.5, -0.5]] * 10,
            "nf.g": [1.0] * 32,
        }
        n1 = save(demo, sys.argv[2] + ".raw", compress=False)
        n2 = save(demo, sys.argv[2], compress=True)
        print("raw      %d bytes" % n1)
        print("zlib     %d bytes" % n2)
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    else:
        print("Unknown command or bad arguments")
        sys.exit(1)
