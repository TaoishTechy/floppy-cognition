"""bumpy - numpy replacement, stdlib only. No numpy import.
Provides ndarray + dtype + the ops laser/train2 need.
"""
import builtins as _builtins
import math
import struct

_py_abs = _builtins.abs
_py_round = _builtins.round
_py_sum = _builtins.sum
_py_max = _builtins.max
_py_min = _builtins.min
_py_float = _builtins.float
_py_int = _builtins.int

class DType:
    def __init__(self, name, kind, itemsize, fmt):
        self.name = name
        self.kind = kind  # 'f','i','u','b'
        self.itemsize = itemsize
        self.fmt = fmt
    def __repr__(self):
        return "bumpy.%s" % self.name
    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        if isinstance(other, str):
            try:
                return self == dtype(other)
            except Exception:
                return False
        return False
    def __hash__(self):
        return hash(self.name)

float32 = DType("float32", "f", 4, "f")
float64 = DType("float64", "f", 8, "d")
int8 = DType("int8", "i", 1, "b")
int16 = DType("int16", "i", 2, "h")
int32 = DType("int32", "i", 4, "i")
int64 = DType("int64", "i", 8, "q")
uint8 = DType("uint8", "u", 1, "B")
bool_ = DType("bool", "b", 1, "B")
float_ = float64
int_ = int64

_STR_TO_DTYPE = {
    "float32": float32, "<f4": float32, ">f4": float32, "f4": float32, "<f": float32,
    "float64": float64, "<f8": float64, ">f8": float64, "f8": float64, "<d": float64,
    "int8": int8, "i1": int8, "<i1": int8, "b": int8,
    "int16": int16, "<i2": int16, "i2": int16,
    "int32": int32, "<i4": int32, "i4": int32,
    "int64": int64, "<i8": int64, "i8": int64,
    "uint8": uint8, "u1": uint8, "<u1": uint8,
    "bool": bool_, "bool_": bool_,
}
def dtype(x):
    if isinstance(x, DType):
        return x
    if x is _py_float or x is float:
        return float64
    if x is _py_int or x is int:
        return int64
    if x is bool:
        return bool_
    if isinstance(x, str):
        if x in _STR_TO_DTYPE:
            return _STR_TO_DTYPE[x]
        raise ValueError("unknown dtype %r" % x)
    raise TypeError("cannot convert %r to dtype" % (x,))

_dtype_fn = dtype
def _cast_value(v, dt):
    if dt.kind == "f":
        if dt is float32:
            return struct.unpack("f", struct.pack("f", _py_float(v)))[0]
        return _py_float(v)
    if dt.kind == "b":
        return bool(v)
    if dt.kind == "u":
        return _py_int(v) & 0xFF
    # int
    iv = _py_int(v)
    if dt is int8:
        iv = ((iv + 128) % 256) - 128
    elif dt is int16:
        iv = ((iv + 32768) % 65536) - 32768
    elif dt is int32:
        iv = ((iv + 2**31) % 2**32) - 2**31
    elif dt is int64:
        iv = ((iv + 2**63) % 2**64) - 2**63
    return iv

def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n

def _infer_shape(data):
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        first = _infer_shape(data[0]) if isinstance(data[0], (list, tuple)) else ()
        return (len(data),) + first
    return ()

def _flatten(data):
    if isinstance(data, (list, tuple)):
        out = []
        for x in data:
            out.extend(_flatten(x))
        return out
    return [data]

def _broadcast_shape(s1, s2):
    n = _py_max(len(s1), len(s2))
    a = [1]*(n-len(s1)) + list(s1)
    b = [1]*(n-len(s2)) + list(s2)
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError("shapes %s and %s not broadcastable" % (s1, s2))
    return tuple(out)

def _expand_data(data, shape, target):
    if tuple(shape) == tuple(target):
        return list(data)
    n = len(target)
    s = [1]*(n-len(shape)) + list(shape)
    strides = [1]*n
    for i in range(n-2, -1, -1):
        strides[i] = strides[i+1]*s[i+1]
    total = _prod(target)
    tstrides = [1]*n
    for i in range(n-2, -1, -1):
        tstrides[i] = tstrides[i+1]*target[i+1]
    out = [0]*total
    for flat in range(total):
        rem = flat
        src = 0
        for i in range(n):
            idx = rem // tstrides[i] if tstrides[i] else 0
            rem = rem % tstrides[i] if tstrides[i] else 0
            sidx = 0 if s[i] == 1 else idx
            src += sidx * strides[i]
        out[flat] = data[src]
    return out

def _binary_op(a, b, func, dt):
    shape = _broadcast_shape(a.shape, b.shape)
    ad = _expand_data(a._data, a.shape, shape)
    bd = _expand_data(b._data, b.shape, shape)
    vals = [_cast_value(func(x, y), dt) for x, y in zip(ad, bd)]
    return ndarray(vals, shape, dt)

def _scalar_wrap(v, dt):
    if dt.kind == "f":
        return _py_float(v)
    if dt.kind == "b":
        return bool(v)
    return _py_int(v)

def _collect_setitem_indices(shape, key_tuple):
    ndim = len(shape)
    keys = list(key_tuple)
    if Ellipsis in keys:
        ei = keys.index(Ellipsis)
        n_missing = ndim - (len(keys)-1)
        keys = keys[:ei] + [slice(None)]*n_missing + keys[ei+1:]
    keys = [k for k in keys if k is not None]
    while len(keys) < ndim:
        keys.append(slice(None))
    keys = keys[:ndim]
    per_dim = []
    tgt = []
    for dim, (k, sz) in enumerate(zip(keys, shape)):
        if isinstance(k, int):
            idx = k if k >= 0 else sz + k
            per_dim.append([idx])
        elif isinstance(k, slice):
            idxs = list(range(*k.indices(sz)))
            per_dim.append(idxs)
            tgt.append(len(idxs))
        else:
            raise TypeError("unsupported key %r" % (k,))
    strides = [1]*ndim
    for i in range(ndim-2, -1, -1):
        strides[i] = strides[i+1]*shape[i+1]
    indices = []
    def rec(d, base):
        if d == ndim:
            indices.append(base)
            return
        for ix in per_dim[d]:
            rec(d+1, base + ix*strides[d])
    rec(0, 0)
    return indices, tuple(tgt)

class ndarray:
    def __init__(self, data=None, shape=None, dtype_=None):
        dt = dtype(dtype_) if dtype_ is not None else None
        if data is None:
            if shape is None:
                shape = ()
            self._shape = tuple(shape)
            self._dtype = dt if dt is not None else float64
            self._data = [0]*_prod(self._shape)
            return
        if isinstance(data, ndarray):
            if dt is None:
                dt = data.dtype
            if shape is None:
                self._shape = tuple(data.shape)
                self._data = [_cast_value(v, dt) for v in data._data]
                self._dtype = dt
                return
            else:
                flat = list(data._data)
                self._shape = tuple(shape)
                if _prod(self._shape) != len(flat):
                    raise ValueError("reshape size mismatch")
                self._data = [_cast_value(v, dt) for v in flat]
                self._dtype = dt
                return
        if shape is not None:
            self._shape = tuple(shape)
            n = _prod(self._shape)
            if isinstance(data, (bytes, bytearray)):
                raise ValueError("use frombuffer for bytes")
            if isinstance(data, (int, float, bool)):
                if n != 1:
                    raise ValueError("scalar data with shape %s" % (self._shape,))
                if dt is None:
                    dt = float64 if isinstance(data, float) else int64
                self._dtype = dt
                self._data = [_cast_value(data, dt)]
                return
            flat = list(data)
            if len(flat) != n:
                raise ValueError("data length %d != shape %s (%d)" % (len(flat), self._shape, n))
            if dt is None:
                dt = int64 if all(isinstance(v, int) and not isinstance(v, bool) for v in flat) else float64
                if flat and all(isinstance(v, bool) for v in flat):
                    dt = bool_
            self._dtype = dt
            self._data = [_cast_value(v, dt) for v in flat]
            return
        if isinstance(data, (int, float, bool)):
            if dt is None:
                dt = float64 if isinstance(data, float) else (bool_ if isinstance(data, bool) else int64)
            self._shape = ()
            self._dtype = dt
            self._data = [_cast_value(data, dt)]
            return
        if isinstance(data, (list, tuple)):
            sh = _infer_shape(data)
            flat = _flatten(data)
            if dt is None:
                if len(flat) == 0:
                    dt = float64
                elif all(isinstance(v, bool) for v in flat):
                    dt = bool_
                elif all(isinstance(v, int) and not isinstance(v, bool) for v in flat):
                    dt = int64
                else:
                    dt = float64
            self._shape = tuple(sh)
            self._dtype = dt
            self._data = [_cast_value(v, dt) for v in flat]
            return
        raise TypeError("cannot make ndarray from %r" % type(data))

    @property
    def shape(self):
        return self._shape
    @property
    def dtype(self):
        return self._dtype
    @property
    def ndim(self):
        return len(self._shape)
    @property
    def size(self):
        return _prod(self._shape)
    @property
    def T(self):
        if self.ndim < 2:
            return self.copy()
        return self.transpose(tuple(reversed(range(self.ndim))))
    def copy(self):
        return ndarray(list(self._data), tuple(self._shape), self._dtype)
    def astype(self, dt):
        dt2 = dtype(dt)
        return ndarray([_cast_value(v, dt2) for v in self._data], tuple(self._shape), dt2)
    def tobytes(self):
        fmt = self._dtype.fmt
        if self._dtype is bool_:
            fmt = "B"
            return struct.pack("<%dB" % len(self._data), *[1 if v else 0 for v in self._data])
        if len(self._data) == 0:
            return b""
        return struct.pack("<%d%s" % (len(self._data), fmt), *self._data)
    def tolist(self):
        def build(data, shape):
            if len(shape) == 0:
                return _scalar_wrap(data[0], self._dtype) if data else 0
            if len(shape) == 1:
                return [_scalar_wrap(v, self._dtype) for v in data]
            step = _prod(shape[1:])
            return [build(data[i*step:(i+1)*step], shape[1:]) for i in range(shape[0])]
        if self._shape == ():
            return _scalar_wrap(self._data[0], self._dtype)
        return build(self._data, self._shape)
    def item(self):
        if self.size != 1:
            raise ValueError("item() only for size-1")
        return _scalar_wrap(self._data[0], self._dtype)
    def __float__(self):
        return _py_float(self.item())
    def __int__(self):
        return _py_int(self.item())
    def __bool__(self):
        if self.size != 1:
            raise ValueError("bool only for size-1")
        return bool(self._data[0])
    def __len__(self):
        if len(self._shape) == 0:
            raise TypeError("len of 0-d")
        return self._shape[0]
    def __repr__(self):
        return "array(%s, shape=%s, dtype=%s)" % (self.tolist(), self._shape, self._dtype.name)
    def __iter__(self):
        if self.ndim == 0:
            yield _scalar_wrap(self._data[0], self._dtype)
            return
        if self.ndim == 1:
            for v in self._data:
                yield _scalar_wrap(v, self._dtype)
            return
        step = _prod(self._shape[1:])
        for i in range(self._shape[0]):
            chunk = self._data[i*step:(i+1)*step]
            yield ndarray(chunk, tuple(self._shape[1:]), self._dtype)
    def __getitem__(self, key):
        return _getitem_impl(self, key)
    def __setitem__(self, key, value):
        _setitem_impl(self, key, value)
    def __add__(self, other):
        if isinstance(other, ndarray):
            dt = _result_dtype(self._dtype, other.dtype)
            return _binary_op(self, other, lambda x, y: x+y, dt)
        if isinstance(other, (int, float, bool)):
            dt = _result_dtype(self._dtype, _infer_scalar_dtype(other))
            return ndarray([_cast_value(v+other, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, ndarray):
            dt = _result_dtype(self._dtype, other.dtype)
            return _binary_op(self, other, lambda x, y: x-y, dt)
        if isinstance(other, (int, float, bool)):
            dt = _result_dtype(self._dtype, _infer_scalar_dtype(other))
            return ndarray([_cast_value(v-other, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __rsub__(self, other):
        if isinstance(other, (int, float, bool)):
            dt = _result_dtype(self._dtype, _infer_scalar_dtype(other))
            return ndarray([_cast_value(other-v, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __mul__(self, other):
        if isinstance(other, ndarray):
            dt = _result_dtype(self._dtype, other.dtype)
            return _binary_op(self, other, lambda x, y: x*y, dt)
        if isinstance(other, (int, float, bool)):
            dt = _result_dtype(self._dtype, _infer_scalar_dtype(other))
            return ndarray([_cast_value(v*other, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, ndarray):
            dt = float64 if (self._dtype.kind in "iu" and other.dtype.kind in "iu") else _result_dtype(self._dtype, other.dtype)
            if self._dtype is float32 and other.dtype is float32:
                dt = float32
            elif self._dtype.kind == "f" or other.dtype.kind == "f":
                dt = float64 if (self._dtype is float64 or other.dtype is float64) else float32
            return _binary_op(self, other, lambda x, y: x/y, dt)
        if isinstance(other, (int, float, bool)):
            if self._dtype is float32:
                dt = float32
            elif self._dtype.kind == "f":
                dt = float64
            else:
                dt = float64
            return ndarray([_cast_value(v/other, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __rtruediv__(self, other):
        if isinstance(other, (int, float, bool)):
            dt = float32 if self._dtype is float32 else float64
            return ndarray([_cast_value(other/v, dt) for v in self._data], tuple(self._shape), dt)
        return NotImplemented
    def __neg__(self):
        return ndarray([_cast_value(-v, self._dtype) for v in self._data], tuple(self._shape), self._dtype)
    def __pow__(self, exp):
        if isinstance(exp, (int, float)):
            dt = self._dtype
            if isinstance(exp, float) or self._dtype.kind == "f":
                dt = float64 if self._dtype is float64 else float32
            return ndarray([_cast_value(v ** exp, dt) for v in self._data], tuple(self._shape), dt)
        if isinstance(exp, ndarray):
            dt = _result_dtype(self._dtype, exp.dtype)
            return _binary_op(self, exp, lambda x, y: x ** y, dt)
        return NotImplemented
    def __eq__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x == y else 0.0, bool_)
        if isinstance(other, (int, float, bool)):
            return ndarray([1.0 if v == other else 0.0 for v in self._data], tuple(self._shape), bool_)
        return NotImplemented
    def __ne__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x != y else 0.0, bool_)
        if isinstance(other, (int, float, bool)):
            return ndarray([1.0 if v != other else 0.0 for v in self._data], tuple(self._shape), bool_)
        return NotImplemented
    def __gt__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x > y else 0.0, bool_)
        return ndarray([1.0 if v > other else 0.0 for v in self._data], tuple(self._shape), bool_)
    def __lt__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x < y else 0.0, bool_)
        return ndarray([1.0 if v < other else 0.0 for v in self._data], tuple(self._shape), bool_)
    def __ge__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x >= y else 0.0, bool_)
        return ndarray([1.0 if v >= other else 0.0 for v in self._data], tuple(self._shape), bool_)
    def __le__(self, other):
        if isinstance(other, ndarray):
            return _binary_op(self, other, lambda x, y: 1.0 if x <= y else 0.0, bool_)
        return ndarray([1.0 if v <= other else 0.0 for v in self._data], tuple(self._shape), bool_)
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        if -1 in shape:
            known = 1
            neg = None
            for i, s in enumerate(shape):
                if s == -1:
                    if neg is not None:
                        raise ValueError("only one -1 allowed")
                    neg = i
                else:
                    known *= s
            if neg is not None:
                shape = list(shape)
                shape[neg] = self.size // known
                shape = tuple(shape)
        if _prod(shape) != self.size:
            raise ValueError("cannot reshape")
        return ndarray(list(self._data), shape, self._dtype)
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError("bad axes")
        new_shape = tuple(self._shape[a] for a in axes)
        old_strides = [1]*self.ndim
        for i in range(self.ndim-2, -1, -1):
            old_strides[i] = old_strides[i+1]*self._shape[i+1]
        new_strides = [1]*self.ndim
        for i in range(self.ndim-2, -1, -1):
            new_strides[i] = new_strides[i+1]*new_shape[i+1]
        out = [0]*self.size
        for flat in range(self.size):
            rem = flat
            idx = [0]*self.ndim
            for i in range(self.ndim):
                idx[i] = rem // new_strides[i] if new_strides[i] else 0
                rem = rem % new_strides[i] if new_strides[i] else 0
            src = 0
            for i in range(self.ndim):
                src += idx[i]*old_strides[axes[i]]
            out[flat] = self._data[src]
        return ndarray(out, new_shape, self._dtype)
    def flatten(self):
        return ndarray(list(self._data), (self.size,), self._dtype)
    def mean(self, axis=None, keepdims=False):
        return _reduce(self, "mean", axis, keepdims)
    def sum(self, axis=None, keepdims=False):
        return _reduce(self, "sum", axis, keepdims)
    def max(self, axis=None, keepdims=False):
        return _reduce(self, "max", axis, keepdims)
    def min(self, axis=None, keepdims=False):
        return _reduce(self, "min", axis, keepdims)
    def any(self):
        return any(v != 0 for v in self._data)
    def all(self):
        return all(v != 0 for v in self._data)

def _infer_scalar_dtype(v):
    if isinstance(v, bool):
        return bool_
    if isinstance(v, int):
        return int64
    return float64

def _result_dtype(a, b):
    if a is float64 or b is float64:
        return float64
    if a is float32 or b is float32:
        return float32
    if a.kind == "f" or b.kind == "f":
        return float64
    if a is int64 or b is int64:
        return int64
    if a is int32 or b is int32:
        return int32
    return a

def _reduce(arr, op, axis, keepdims):
    if axis is None:
        if op == "mean":
            v = _py_sum(arr._data)/len(arr._data) if arr._data else 0.0
            dt = float32 if arr.dtype is float32 else float64
            return ndarray(_cast_value(v, dt), (), dt)
        if op == "sum":
            v = _py_sum(arr._data)
            dt = arr.dtype
            return ndarray(_cast_value(v, dt), (), dt)
        if op == "max":
            v = _py_max(arr._data) if arr._data else 0
            return ndarray(_cast_value(v, arr.dtype), (), arr.dtype)
        if op == "min":
            v = _py_min(arr._data) if arr._data else 0
            return ndarray(_cast_value(v, arr.dtype), (), arr.dtype)
    ax = axis if axis >= 0 else arr.ndim + axis
    if ax != arr.ndim-1:
        axes = [d for d in range(arr.ndim) if d != ax] + [ax]
        t = arr.transpose(tuple(axes))
    else:
        t = arr
        axes = list(range(arr.ndim))
    last = t.shape[-1]
    outer = t.size // last if last else 0
    out_shape = tuple(t.shape[:-1])
    vals = []
    for i in range(outer):
        chunk = t._data[i*last:(i+1)*last]
        if op == "mean":
            vals.append(_py_sum(chunk)/len(chunk) if chunk else 0.0)
        elif op == "sum":
            vals.append(_py_sum(chunk))
        elif op == "max":
            vals.append(_py_max(chunk) if chunk else 0)
        elif op == "min":
            vals.append(_py_min(chunk) if chunk else 0)
    dt = arr.dtype
    if op == "mean":
        dt = float32 if arr.dtype is float32 else (float64 if arr.dtype.kind == "f" else float64)
        if arr.dtype.kind in "iu":
            dt = float64
    if keepdims:
        osh = list(arr.shape)
        osh[ax] = 1
        return ndarray([_cast_value(v, dt) for v in vals], tuple(osh), dt)
    return ndarray([_cast_value(v, dt) for v in vals], out_shape, dt)

def _getitem_impl(arr, key):
    if isinstance(key, (int,)):
        return _getitem_single(arr, key)
    if isinstance(key, slice):
        return _getitem_tuple(arr, (key,))
    if isinstance(key, tuple):
        return _getitem_tuple(arr, key)
    if key is None:
        return ndarray(list(arr._data), (1,)+tuple(arr.shape), arr.dtype)
    if key is Ellipsis:
        return arr.copy()
    raise TypeError("unsupported key %r" % (key,))

def _getitem_single(arr, key):
    return _getitem_tuple(arr, (key,))

def _getitem_tuple(arr, keys):
    keys = list(keys)
    if Ellipsis in keys:
        ei = keys.index(Ellipsis)
        n_needed = arr.ndim - (len([k for k in keys if k is not Ellipsis and k is not None]))
        keys = keys[:ei] + [slice(None)]*n_needed + keys[ei+1:]
    out_shape = []
    per_dim_idx = []
    dim = 0
    for k in keys:
        if k is None:
            out_shape.append(1)
            per_dim_idx.append(None)
            continue
        if dim >= arr.ndim:
            raise IndexError("too many indices")
        sz = arr.shape[dim]
        if isinstance(k, int):
            idx = k if k >= 0 else sz + k
            if idx < 0 or idx >= sz:
                raise IndexError("index out of range")
            per_dim_idx.append([idx])
        elif isinstance(k, slice):
            idxs = list(range(*k.indices(sz)))
            per_dim_idx.append(idxs)
            out_shape.append(len(idxs))
        else:
            raise TypeError("unsupported key %r" % (k,))
        dim += 1
    while dim < arr.ndim:
        per_dim_idx.append(list(range(arr.shape[dim])))
        out_shape.append(arr.shape[dim])
        dim += 1
    strides = [1]*arr.ndim
    for i in range(arr.ndim-2, -1, -1):
        strides[i] = strides[i+1]*arr.shape[i+1]
    out_data = []
    def rec(pos, base):
        if pos == len(per_dim_idx):
            out_data.append(arr._data[base])
            return
        entry = per_dim_idx[pos]
        if entry is None:
            rec(pos+1, base)
            return
        arr_dim = sum(1 for e in per_dim_idx[:pos] if e is not None)
        for ix in entry:
            rec(pos+1, base + ix*strides[arr_dim])
    rec(0, 0)
    if len(out_shape) == 0:
        return ndarray(out_data[0], (), arr.dtype)
    return ndarray(out_data, tuple(out_shape), arr.dtype)

def _setitem_impl(arr, key, value):
    if not isinstance(key, tuple):
        key = (key,)
    indices, tgt_shape = _collect_setitem_indices(arr.shape, key)
    if isinstance(value, ndarray):
        src = value._data
        src_shape = value.shape
        if src_shape != tgt_shape:
            if len(src) == 1:
                src = src * len(indices)
            else:
                try:
                    src = _expand_data(src, src_shape, tgt_shape)
                except Exception:
                    if len(src) != len(indices):
                        raise ValueError("setitem value shape %s vs target %s" % (src_shape, tgt_shape))
    elif isinstance(value, (list, tuple)):
        tmp = array(value)
        src = tmp._data
        if len(src) == 1:
            src = src * len(indices)
        elif len(src) != len(indices):
            src = _expand_data(src, tmp.shape, tgt_shape) if tmp.shape != tgt_shape else src
    elif isinstance(value, (int, float, bool)):
        src = [value]*len(indices)
    else:
        tmp = array(value)
        src = tmp._data
        if len(src) == 1:
            src = src * len(indices)
    if len(src) != len(indices):
        raise ValueError("setitem length mismatch %d vs %d" % (len(src), len(indices)))
    for idx, v in zip(indices, src):
        arr._data[idx] = _cast_value(v, arr.dtype)

def array(data, dtype=None):
    if isinstance(data, ndarray):
        if dtype is None:
            return data.copy()
        return data.astype(dtype)
    return ndarray(data, dtype_=dtype)
def zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = _dtype_fn(dtype) if dtype is not None else float64
    if dt is _py_float:
        dt = float32
    return ndarray([0]*_prod(tuple(shape)), tuple(shape), dt)
def ones(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    dt = _dtype_fn(dtype) if dtype is not None else float64
    if dt is _py_float:
        dt = float32
    return ndarray([1]*_prod(tuple(shape)), tuple(shape), dt)
def empty(shape, dtype=None):
    return zeros(shape, dtype=dtype)
def arange(*args, dtype=None):
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError("arange expects 1-3 args")
    vals = []
    v = start
    if step > 0:
        while v < stop:
            vals.append(v); v += step
    else:
        while v > stop:
            vals.append(v); v += step
    dt = _dtype_fn(dtype) if dtype is not None else (int64 if all(isinstance(x, int) for x in vals) else float64)
    if dtype is _py_float:
        dt = float32
    return ndarray(vals, (len(vals),), dt)
def eye(n, dtype=None):
    dt = _dtype_fn(dtype) if dtype is not None else float64
    data = [0]*(n*n)
    for i in range(n):
        data[i*n+i] = 1
    return ndarray(data, (n, n), dt)
def frombuffer(buffer, dtype=None, count=-1, offset=0):
    dt = _dtype_fn(dtype) if dtype is not None else float64
    b = bytes(buffer)
    fmt = dt.fmt
    if dt is bool_:
        fmt = "B"
    sz = dt.itemsize
    avail = (len(b)-offset)//sz
    if count is None or count < 0:
        count = avail
    else:
        count = min(count, avail)
    if count == 0:
        return ndarray([], (0,), dt)
    vals = list(struct.unpack_from("<%d%s" % (count, fmt), b, offset))
    return ndarray(vals, (count,), dt)
def concatenate(arrays, axis=0):
    if not arrays:
        raise ValueError("concatenate empty")
    arrs = [a if isinstance(a, ndarray) else array(a) for a in arrays]
    if len(arrs) == 1:
        return arrs[0].copy()
    ax = axis if axis >= 0 else arrs[0].ndim + axis
    for a in arrs[1:]:
        if a.ndim != arrs[0].ndim:
            raise ValueError("concatenate ndim mismatch")
        for d in range(a.ndim):
            if d != ax and a.shape[d] != arrs[0].shape[d]:
                raise ValueError("concatenate shape mismatch")
    out_shape = list(arrs[0].shape)
    out_shape[ax] = _py_sum(a.shape[ax] for a in arrs)
    if arrs[0].ndim == 1:
        data = []
        for a in arrs:
            data.extend(a._data)
        return ndarray(data, tuple(out_shape), arrs[0].dtype)
    outer = _prod(arrs[0].shape[:ax]) if ax > 0 else 1
    inner = _prod(arrs[0].shape[ax+1:]) if ax+1 < arrs[0].ndim else 1
    out = []
    for o in range(outer):
        for a in arrs:
            ax_len = a.shape[ax]
            start = o*ax_len*inner
            out.extend(a._data[start:start+ax_len*inner])
    return ndarray(out, tuple(out_shape), arrs[0].dtype)
def matmul(a, b):
    if not isinstance(a, ndarray):
        a = array(a)
    if not isinstance(b, ndarray):
        b = array(b)
    if a.ndim == 1 and b.ndim == 1:
        if a.shape[0] != b.shape[0]:
            raise ValueError("matmul 1d mismatch")
        dt = _result_dtype(a.dtype, b.dtype)
        v = _py_sum(x*y for x, y in zip(a._data, b._data))
        return ndarray(_cast_value(v, dt), (), dt)
    if a.ndim == 1:
        a2 = a.reshape((1, a.shape[0]))
        r = matmul(a2, b)
        if r.ndim >= 2:
            return r.reshape(tuple(r.shape[:-2]) + (r.shape[-1],))
        return r
    if b.ndim == 1:
        b2 = b.reshape((b.shape[0], 1))
        r = matmul(a, b2)
        return r.reshape(tuple(r.shape[:-2]) + (r.shape[-2],))
    if a.ndim == 2 and b.ndim == 2:
        M, K = a.shape; K2, N = b.shape
        if K != K2:
            raise ValueError("matmul shape %s vs %s" % (a.shape, b.shape))
        dt = _result_dtype(a.dtype, b.dtype)
        if dt is float64 and a.dtype is float32 and b.dtype is float32:
            dt = float32
        out = []
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a._data[i*K+k]*b._data[k*N+j]
                out.append(_cast_value(s, dt))
        return ndarray(out, (M, N), dt)
    a_batch = a.shape[:-2]; b_batch = b.shape[:-2]
    batch = _broadcast_shape(a_batch, b_batch) if a_batch or b_batch else ()
    M, K = a.shape[-2], a.shape[-1]; K2, N = b.shape[-2], b.shape[-1]
    if K != K2:
        raise ValueError("matmul K mismatch")
    dt = _result_dtype(a.dtype, b.dtype)
    if a.dtype is float32 and b.dtype is float32:
        dt = float32
    def get_block(arr, batch_idx, batch_shape):
        arr_batch = arr.shape[:-2]
        offset = len(batch_shape)-len(arr_batch)
        src_batch = []
        for i, bi in enumerate(batch_idx):
            if i < offset:
                src_batch.append(0)
            else:
                src_batch.append(0 if arr_batch[i-offset] == 1 else bi)
        strides = []
        s = arr.shape[-2]*arr.shape[-1]
        for d in reversed(arr_batch):
            strides.append(s); s *= d
        strides = list(reversed(strides))
        base = 0
        for idx, st in zip(src_batch, strides):
            base += idx*st
        block = arr._data[base:base+arr.shape[-2]*arr.shape[-1]]
        return block
    total_batch = _prod(batch) if batch else 1
    out_blocks = []
    for bi in range(total_batch):
        idx = []
        rem = bi
        strides_b = [1]*len(batch)
        for i in range(len(batch)-2, -1, -1):
            strides_b[i] = strides_b[i+1]*batch[i+1]
        for i in range(len(batch)):
            idx.append(rem // strides_b[i] if strides_b[i] else 0)
            rem = rem % strides_b[i] if strides_b[i] else 0
        idx_t = tuple(idx)
        ab = get_block(a, idx_t, batch)
        bb = get_block(b, idx_t, batch)
        for i in range(M):
            for j in range(N):
                ssum = 0.0
                for k in range(K):
                    ssum += ab[i*K+k]*bb[k*N+j]
                out_blocks.append(_cast_value(ssum, dt))
    out_shape = tuple(batch) + (M, N)
    return ndarray(out_blocks, out_shape, dt)

def abs(x):
    if isinstance(x, ndarray):
        return ndarray([_py_abs(v) for v in x._data], tuple(x._shape), x._dtype)
    return _py_abs(x)
def round(x, decimals=0):
    if isinstance(x, ndarray):
        if decimals == 0:
            return ndarray([_py_float(_py_round(v)) for v in x._data], tuple(x._shape), x._dtype)
        f = 10 ** decimals
        return ndarray([_py_round(v*f)/f for v in x._data], tuple(x._shape), x._dtype)
    return _py_round(x)
def clip(x, a_min, a_max):
    if isinstance(x, ndarray):
        return ndarray([_py_min(_py_max(v, a_min), a_max) for v in x._data], tuple(x._shape), x._dtype)
    return _py_min(_py_max(x, a_min), a_max)
def allclose(a, b, rtol=1e-5, atol=1e-8):
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        if a.shape != b.shape:
            return False
        return all(_py_abs(x-y) <= atol + rtol*_py_abs(y) for x, y in zip(a._data, b._data))
    return _py_abs(a-b) <= atol + rtol*_py_abs(b)
def sqrt(x):
    if isinstance(x, ndarray):
        return ndarray([math.sqrt(v) if v >= 0 else 0.0 for v in x._data], tuple(x._shape), x._dtype)
    return math.sqrt(x)

# aliases for laser/train2
ndarray = ndarray
BumpyArray = ndarray
