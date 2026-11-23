"""riskytensor Phase-1 — float32 save/load + header/manifest/SHA-256.

Implements the Consolidated Blueprint v4.0 Phase-0 and Phase-1 requirements:
- 64-byte header (magic, version, lengths, SHAs)
- deterministic JSON manifest (sorted keys)
- per-tensor and payload SHA-256
- float32 storage (exact round-trip)
- info() / verify() / save() / load()

Later phases (int8_per_row, zlib, convert_safetensors) are still pending.
No pickle, no eval, no exec. Parser is pure stdlib.
"""
import struct
import json
import hashlib
import os
import sys

MAGIC = b"RISKY4\n"
VERSION = 4
HEADER_LEN = 64

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------------------------------------------------------------------------
# Optional laser / bumpy for returned Tensors (load path)
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

def _to_float32_bytes(obj):
    """Convert Tensor / ndarray / list to little-endian float32 bytes + shape + dtype name."""
    _try_import_backends()
    # Prefer laser.Tensor
    if _Tensor is not None and isinstance(obj, _Tensor):
        arr = obj._arr if hasattr(obj, "_arr") else obj
        shape = tuple(obj.shape)
        # force float32
        if hasattr(arr, "astype"):
            arr = arr.astype("float32") if arr.dtype.name != "float32" else arr
        data = arr.tobytes() if hasattr(arr, "tobytes") else struct.pack("<%df" % arr.size, *arr._data)
        return data, shape, "float32"
    # bumpy ndarray
    if _np is not None and isinstance(obj, _np.ndarray):
        shape = tuple(obj.shape)
        if obj.dtype.name != "float32":
            obj = obj.astype("float32")
        return obj.tobytes(), shape, "float32"
    # plain list / nested list
    if isinstance(obj, (list, tuple)):
        flat = []
        def flatten(x):
            if isinstance(x, (list, tuple)):
                for y in x:
                    flatten(y)
            else:
                flat.append(float(x))
        flatten(obj)
        # infer shape (only 1-D or rectangular 2-D for simplicity)
        if not obj:
            shape = (0,)
        elif isinstance(obj[0], (list, tuple)):
            shape = (len(obj), len(obj[0]))
        else:
            shape = (len(obj),)
        data = struct.pack("<%df" % len(flat), *flat)
        return data, shape, "float32"
    raise TypeError("riskytensor: unsupported type %r" % type(obj))

def _from_float32_bytes(data, shape, prefer_tensor=True):
    """Rebuild a laser.Tensor (preferred) or list from float32 bytes."""
    _try_import_backends()
    n = 1
    for s in shape:
        n *= s
    if len(data) != n * 4:
        raise ValueError("riskytensor: float32 data length mismatch")
    vals = list(struct.unpack("<%df" % n, data))
    if prefer_tensor and _Tensor is not None and _np is not None:
        arr = _np.ndarray(vals, tuple(shape), _np.float32)
        return _Tensor(arr, requires_grad=False)
    # fallback: nested lists
    if len(shape) == 0:
        return vals[0] if vals else 0.0
    if len(shape) == 1:
        return vals
    # 2-D
    rows, cols = shape[0], shape[1]
    return [vals[i*cols:(i+1)*cols] for i in range(rows)]

# ---------------------------------------------------------------------------
# Header / manifest helpers
# ---------------------------------------------------------------------------

def info(path: str) -> dict:
    """Return the manifest only, without loading weights."""
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
    manifest["_payload_sha"] = payload_sha.hex()  # convenience for callers
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
# Phase-1 public API
# ---------------------------------------------------------------------------

def save(state_dict, path, recipe=None):
    """Save a state_dict to a .riskytensor file (float32 only in Phase-1).

    state_dict: dict[str, Tensor | ndarray | list]
    recipe: ignored in Phase-1 (everything stored as float32)
    Returns file size in bytes.
    Deterministic: same inputs -> identical bytes.
    """
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict")

    tensors_meta = {}
    blobs = []
    offset = 0

    # Deterministic order
    for name in sorted(state_dict.keys()):
        obj = state_dict[name]
        data, shape, dtype_name = _to_float32_bytes(obj)
        sha = _sha256_hex(data)
        tensors_meta[name] = {
            "shape": list(shape),
            "dtype": dtype_name,
            "storage": "float32",
            "offset": offset,
            "length": len(data),
            "compressed_length": len(data),
            "compression": "none",
            "sha256": sha,
        }
        blobs.append(data)
        offset += len(data)

    payload = b"".join(blobs)
    payload_sha = _sha256(payload)

    manifest = {
        "format": "riskytensor",
        "version": VERSION,
        "created_by": "riskytensor Phase-1",
        "tensors": tensors_meta,
        "quantization": {
            "default": "float32",
            "exceptions": {},
        },
        "metadata": {},
    }
    if recipe is not None:
        manifest["metadata"]["recipe"] = str(recipe)

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
    """Load a .riskytensor file and return dict[str, Tensor] (or lists if laser unavailable).

    Verifies magic, version, manifest SHA, payload SHA and per-tensor SHA.
    Raises ValueError naming the problem on any mismatch.
    """
    manifest = info(path)
    tensors_meta = manifest.get("tensors", {})
    if not tensors_meta:
        return {}

    # Re-read payload
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        manifest_len = struct.unpack_from("<Q", header, 11)[0]
        f.seek(HEADER_LEN + manifest_len)
        payload = f.read()

    expected_payload_sha = bytes.fromhex(manifest["_payload_sha"])
    if _sha256(payload) != expected_payload_sha:
        raise ValueError("riskytensor: payload SHA-256 mismatch")

    out = {}
    for name in sorted(tensors_meta.keys()):
        meta = tensors_meta[name]
        if meta.get("storage") != "float32":
            raise NotImplementedError(
                "riskytensor Phase-1 only supports float32 storage (got %s for %s)"
                % (meta.get("storage"), name)
            )
        off = meta["offset"]
        length = meta["length"]
        data = payload[off:off+length]
        if len(data) != length:
            raise ValueError("riskytensor: truncated tensor data for %s" % name)
        if _sha256_hex(data) != meta["sha256"]:
            raise ValueError("riskytensor: SHA-256 mismatch for tensor %s" % name)
        shape = tuple(meta["shape"])
        out[name] = _from_float32_bytes(data, shape, prefer_tensor=True)
    return out

def load_raw(path):
    """Return dict[name, (bytes, shape)] for debugging."""
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
        out[name] = (payload[off:off+length], tuple(meta["shape"]))
    return out

def verify(path):
    """Re-hash everything and print OK / FAIL."""
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
        # Tiny demo state_dict using pure lists (no laser required)
        demo = {
            "emb.weight": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "nf.g": [1.0, 1.0],
        }
        n = save(demo, sys.argv[2])
        print("wrote %d bytes" % n)
        loaded = load(sys.argv[2])
        print("loaded keys:", sorted(loaded.keys()))
        for k, v in loaded.items():
            print(" ", k, "->", v)
        print("verify:", "OK" if verify(sys.argv[2]) else "FAIL")

    else:
        print("Unknown command or bad arguments")
        sys.exit(1)
