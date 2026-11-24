"""laser v4.0 - torch replacement, stdlib + bumpy only. No torch import."""
import builtins as _builtins
import math
import os
import pickle
import struct
import sys
import types

_py_abs = _builtins.abs
_py_max = _builtins.max
_py_min = _builtins.min
_py_sum = _builtins.sum
_py_float = _builtins.float
_py_int = _builtins.int

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import bumpy as _np

float32 = _np.float32
float64 = _np.float64
int8 = _np.int8
int16 = _np.int16
int32 = _np.int32
int64 = _np.int64
uint8 = _np.uint8
bool_ = _np.bool_
half = float32
double = float64
short = int16
byte = int8
longlong = int64

_grad_enabled = True
_num_threads = 2
_rng_state = 0x123456789ABCDEF
_rng_has_spare = False
_rng_spare = 0.0
_MASK64 = (1 << 64) - 1

def manual_seed(seed):
    global _rng_state, _rng_has_spare
    _rng_state = _py_int(seed) & _MASK64
    _rng_has_spare = False
    return seed
def seed(s):
    return manual_seed(s)
def initial_seed():
    return _rng_state
def _next_u64():
    global _rng_state
    _rng_state = (_rng_state + 0x9E3779B97F4A7C15) & _MASK64
    z = _rng_state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64
def _uniform():
    return ((_next_u64() >> 11) * (1.0 / 9007199254740992.0))
def _normal_single():
    global _rng_has_spare, _rng_spare
    if _rng_has_spare:
        _rng_has_spare = False
        return _rng_spare
    u1 = _uniform()
    while u1 <= 0.0:
        u1 = _uniform()
    u2 = _uniform()
    r = math.sqrt(-2.0 * math.log(u1))
    th = 2.0 * math.pi * u2
    _rng_spare = r * math.sin(th)
    _rng_has_spare = True
    return r * math.cos(th)
def _normal_array(shape, mean=0.0, std=1.0):
    n = 1
    for s in shape:
        n *= s
    vals = [struct.unpack("f", struct.pack("f", _py_float(mean + std * _normal_single())))[0] for _ in range(n)]
    return _np.ndarray(vals, tuple(shape), float32)
def set_num_threads(n):
    global _num_threads
    _num_threads = _py_int(n)
def get_num_threads():
    return _num_threads

def _accum(t, grad_arr):
    if t.grad is None:
        t.grad = grad_arr
    else:
        t.grad = t.grad + grad_arr
def _unbroadcast(grad, target_shape):
    if grad.shape == tuple(target_shape):
        return grad
    if len(target_shape) == 0:
        return grad.sum()
    g = grad
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    for dim, sz in enumerate(target_shape):
        if sz == 1 and g.shape[dim] != 1:
            g = g.sum(axis=dim, keepdims=True)
    if g.shape != tuple(target_shape):
        g = g.reshape(tuple(target_shape))
    return g
def _expand_to(grad, target_shape):
    if grad.shape == tuple(target_shape):
        return grad
    data = _np._expand_data(grad._data, grad.shape, tuple(target_shape))
    return _np.ndarray(data, tuple(target_shape), grad.dtype)
def _to_tensor(x, dtype=None):
    if isinstance(x, Tensor):
        if dtype is not None and x.dtype != dtype:
            return x.to(dtype)
        return x
    if isinstance(x, _np.ndarray):
        return Tensor(x, dtype=dtype)
    if isinstance(x, (list, tuple)):
        return Tensor(_np.array(x, dtype=dtype))
    if isinstance(x, (int, float, bool)):
        dt = dtype if dtype is not None else (float32 if isinstance(x, float) else int64)
        return Tensor(_np.ndarray(x, (), dt))
    raise TypeError("cannot convert %r to Tensor" % type(x))
def _softmax_last_axis(arr):
    if arr.ndim == 0:
        return arr
    last = arr.shape[-1]
    outer = arr.size // last if last else 0
    out = []
    for i in range(outer):
        chunk = arr._data[i*last:(i+1)*last]
        m = _py_max(chunk)
        exps = [math.exp(v - m) for v in chunk]
        s = _py_sum(exps)
        out.extend([e / s for e in exps])
    return _np.ndarray([struct.unpack("f", struct.pack("f", v))[0] for v in out], tuple(arr.shape), float32)

class Tensor:
    __slots__ = ("_arr", "grad", "requires_grad", "_prev", "_backward_fn", "_op")
    def __init__(self, data=None, dtype=None, requires_grad=False, _prev=(), _backward_fn=None, _op=""):
        if data is None:
            arr = _np.zeros((), dtype=float32)
        elif isinstance(data, Tensor):
            arr = data._arr.copy()
        elif isinstance(data, _np.ndarray):
            arr = data.copy()
        elif isinstance(data, (list, tuple)):
            arr = _np.array(data, dtype=dtype)
        elif isinstance(data, (int, float, bool)):
            dt = dtype if dtype is not None else (float32 if isinstance(data, float) else int64)
            arr = _np.ndarray(data, (), dt)
        else:
            arr = _np.array(data, dtype=dtype)
        if dtype is not None:
            dt = dtype
            if dt is _py_float:
                dt = float32
            elif dt is _py_int:
                dt = int64
            if arr.dtype != dt:
                arr = arr.astype(dt)
        self._arr = arr
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._prev = tuple(_prev)
        self._backward_fn = _backward_fn
        self._op = _op
    @property
    def shape(self):
        return self._arr.shape
    @property
    def dtype(self):
        return self._arr.dtype
    @property
    def ndim(self):
        return self._arr.ndim
    @property
    def T(self):
        if self.ndim < 2:
            return self
        return self.transpose(self.ndim-2, self.ndim-1)
    def dim(self):
        return self._arr.ndim
    def numel(self):
        return self._arr.size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    def __len__(self):
        return self.shape[0]
    def __repr__(self):
        return "tensor(shape=%s, dtype=%s, requires_grad=%s)" % (self.shape, self.dtype.name, self.requires_grad)
    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]
    def __getitem__(self, key):
        out_arr = self._arr[key]
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        orig_shape = self.shape
        if isinstance(key, tuple) and None in key:
            key_no_none = tuple(k for k in key if k is not None)
            none_positions = [i for i, k in enumerate(key) if k is None]
            def _backward(grad):
                g = grad
                if none_positions:
                    new_shape = tuple(s for i, s in enumerate(g.shape) if i not in none_positions)
                    if new_shape != g.shape:
                        g = g.reshape(new_shape)
                base = _np.zeros(orig_shape, dtype=g.dtype)
                indices, _ts = _np._collect_setitem_indices(orig_shape, key_no_none)
                for flat_idx, v in zip(indices, g._data):
                    base._data[flat_idx] = base._data[flat_idx] + v
                _accum(self, base)
            return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="getitem")
        def _backward2(grad):
            base = _np.zeros(orig_shape, dtype=grad.dtype)
            k = key if isinstance(key, tuple) else (key,)
            indices, _ts = _np._collect_setitem_indices(orig_shape, k)
            for flat_idx, v in zip(indices, grad._data):
                base._data[flat_idx] = base._data[flat_idx] + v
            _accum(self, base)
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward2, _op="getitem")
    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self._arr[key] = value._arr
        elif isinstance(value, _np.ndarray):
            self._arr[key] = value
        else:
            self._arr[key] = value
    def __add__(self, other):
        o = _to_tensor(other)
        out_arr = self._arr + o._arr
        if not _grad_enabled or (not self.requires_grad and not o.requires_grad):
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            if self.requires_grad:
                _accum(self, _unbroadcast(grad, self.shape))
            if o.requires_grad:
                _accum(o, _unbroadcast(grad, o.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self, o), _backward_fn=_backward, _op="add")
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        o = _to_tensor(other)
        out_arr = self._arr - o._arr
        if not _grad_enabled or (not self.requires_grad and not o.requires_grad):
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            if self.requires_grad:
                _accum(self, _unbroadcast(grad, self.shape))
            if o.requires_grad:
                neg = _np.ndarray([-v for v in grad._data], tuple(grad.shape), grad.dtype)
                _accum(o, _unbroadcast(neg, o.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self, o), _backward_fn=_backward, _op="sub")
    def __rsub__(self, other):
        return _to_tensor(other).__sub__(self)
    def __mul__(self, other):
        o = _to_tensor(other)
        out_arr = self._arr * o._arr
        if not _grad_enabled or (not self.requires_grad and not o.requires_grad):
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr; b_arr = o._arr
        def _backward(grad):
            if self.requires_grad:
                _accum(self, _unbroadcast(grad * b_arr, self.shape))
            if o.requires_grad:
                _accum(o, _unbroadcast(grad * a_arr, o.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self, o), _backward_fn=_backward, _op="mul")
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        o = _to_tensor(other)
        out_arr = self._arr / o._arr
        if not _grad_enabled or (not self.requires_grad and not o.requires_grad):
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr; b_arr = o._arr
        def _backward(grad):
            if self.requires_grad:
                _accum(self, _unbroadcast(grad / b_arr, self.shape))
            if o.requires_grad:
                # -grad*a / b^2 in out shape
                g_exp = _expand_to(grad, out_arr.shape) if grad.shape != out_arr.shape else grad
                a_exp = _expand_to(a_arr, out_arr.shape) if a_arr.shape != out_arr.shape else a_arr
                b_exp = _expand_to(b_arr, out_arr.shape) if b_arr.shape != out_arr.shape else b_arr
                gb = _np.ndarray([-g * a / (b*b) for g, a, b in zip(g_exp._data, a_exp._data, b_exp._data)], tuple(out_arr.shape), grad.dtype)
                _accum(o, _unbroadcast(gb, o.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self, o), _backward_fn=_backward, _op="div")
    def __rtruediv__(self, other):
        return _to_tensor(other).__truediv__(self)
    def __neg__(self):
        out_arr = -self._arr
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            _accum(self, _np.ndarray([-v for v in grad._data], tuple(grad.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="neg")
    def __pow__(self, exp):
        return self.pow(exp)
    def pow(self, exp):
        if isinstance(exp, Tensor):
            raise NotImplementedError("Tensor exponent")
        out_arr = self._arr ** exp
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr
        def _backward(grad):
            vals = [g * exp * (a ** (exp - 1) if not (a == 0 and exp-1 < 0) else 0.0) for g, a in zip(grad._data, a_arr._data)]
            _accum(self, _unbroadcast(_np.ndarray(vals, tuple(grad.shape), grad.dtype), self.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="pow")
    def __matmul__(self, other):
        o = _to_tensor(other)
        out_arr = _np.matmul(self._arr, o._arr)
        if not _grad_enabled or (not self.requires_grad and not o.requires_grad):
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr; b_arr = o._arr
        def _backward(grad):
            def _t2(arr):
                if arr.ndim < 2:
                    return arr
                axes = list(range(arr.ndim-2)) + [arr.ndim-1, arr.ndim-2]
                return arr.transpose(tuple(axes))
            if self.requires_grad:
                _accum(self, _unbroadcast(_np.matmul(grad, _t2(b_arr)), self.shape))
            if o.requires_grad:
                _accum(o, _unbroadcast(_np.matmul(_t2(a_arr), grad), o.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self, o), _backward_fn=_backward, _op="matmul")
    def __rmatmul__(self, other):
        return _to_tensor(other).__matmul__(self)
    def mean(self, dim=None, keepdim=False):
        if dim is None:
            out_arr = self._arr.mean()
            n = self.numel()
            if not _grad_enabled or not self.requires_grad:
                return Tensor(out_arr, requires_grad=False)
            def _backward(grad):
                _accum(self, _np.ndarray([grad._data[0] / n] * self.numel(), tuple(self.shape), grad.dtype))
            return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="mean")
        ax = dim if dim >= 0 else self.ndim + dim
        out_arr = self._arr.mean(axis=ax, keepdims=keepdim)
        n = self.shape[ax]
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            if keepdim:
                g_exp = _expand_to(grad, self.shape)
            else:
                g_shape = list(grad.shape); g_shape.insert(ax, 1)
                g_exp = _expand_to(grad.reshape(tuple(g_shape)), self.shape)
            _accum(self, _np.ndarray([v / n for v in g_exp._data], tuple(self.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="mean")
    def sum(self, dim=None, keepdim=False):
        if dim is None:
            out_arr = self._arr.sum()
            if not _grad_enabled or not self.requires_grad:
                return Tensor(out_arr, requires_grad=False)
            def _backward(grad):
                _accum(self, _np.ndarray([grad._data[0]] * self.numel(), tuple(self.shape), grad.dtype))
            return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="sum")
        ax = dim if dim >= 0 else self.ndim + dim
        out_arr = self._arr.sum(axis=ax, keepdims=keepdim)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            if keepdim:
                g_exp = _expand_to(grad, self.shape)
            else:
                g_shape = list(grad.shape); g_shape.insert(ax, 1)
                g_exp = _expand_to(grad.reshape(tuple(g_shape)), self.shape)
            _accum(self, g_exp)
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="sum")
    def abs(self):
        out_arr = _np.abs(self._arr)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr
        def _backward(grad):
            vals = [g * (1.0 if a > 0 else (-1.0 if a < 0 else 0.0)) for g, a in zip(grad._data, a_arr._data)]
            _accum(self, _np.ndarray(vals, tuple(grad.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="abs")
    def clamp(self, min_val=None, max_val=None):
        lo = -1e30 if min_val is None else min_val
        hi = 1e30 if max_val is None else max_val
        out_arr = _np.clip(self._arr, lo, hi)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr
        def _backward(grad):
            vals = [g if lo <= a <= hi else 0.0 for g, a in zip(grad._data, a_arr._data)]
            _accum(self, _np.ndarray(vals, tuple(grad.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="clamp")
    def clamp_min(self, min_val):
        return self.clamp(min_val, None)
    def clamp_max(self, max_val):
        return self.clamp(None, max_val)
    def round(self):
        out_arr = _np.round(self._arr)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            _accum(self, _np.zeros(tuple(grad.shape), dtype=grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="round")
    def rsqrt(self):
        vals = [1.0 / math.sqrt(v) if v > 0 else 0.0 for v in self._arr._data]
        out_arr = _np.ndarray([struct.unpack("f", struct.pack("f", x))[0] for x in vals], tuple(self.shape), float32)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr
        def _backward(grad):
            gvals = [g * (-0.5) * (a ** -1.5) if a > 0 else 0.0 for g, a in zip(grad._data, a_arr._data)]
            _accum(self, _np.ndarray(gvals, tuple(grad.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="rsqrt")
    def sqrt(self):
        vals = [math.sqrt(v) if v >= 0 else 0.0 for v in self._arr._data]
        out_arr = _np.ndarray([struct.unpack("f", struct.pack("f", x))[0] for x in vals], tuple(self.shape), float32)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        a_arr = self._arr
        def _backward(grad):
            gvals = [g * 0.5 / math.sqrt(a) if a > 0 else 0.0 for g, a in zip(grad._data, a_arr._data)]
            _accum(self, _np.ndarray(gvals, tuple(grad.shape), grad.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="sqrt")
    def amax(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._arr.max(), requires_grad=False)
        ax = dim if dim >= 0 else self.ndim + dim
        return Tensor(self._arr.max(axis=ax, keepdims=keepdim), requires_grad=False)
    def amin(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._arr.min(), requires_grad=False)
        ax = dim if dim >= 0 else self.ndim + dim
        return Tensor(self._arr.min(axis=ax, keepdims=keepdim), requires_grad=False)
    def argmax(self, dim=None, keepdim=False):
        if dim is None:
            best = 0; bestv = self._arr._data[0]
            for i, v in enumerate(self._arr._data):
                if v > bestv:
                    bestv = v; best = i
            return Tensor(_np.ndarray(best, (), int64), requires_grad=False)
        ax = dim if dim >= 0 else self.ndim + dim
        if ax != self.ndim - 1:
            axes = [d for d in range(self.ndim) if d != ax] + [ax]
            t = self._arr.transpose(tuple(axes))
            last = t.shape[-1]; outer = t.size // last
            out_vals = []
            for i in range(outer):
                chunk = t._data[i*last:(i+1)*last]
                out_vals.append(_py_max(range(last), key=lambda j: chunk[j]))
            out_shape = tuple(t.shape[:-1]) + ((1,) if keepdim else ())
            return Tensor(_np.ndarray(out_vals, out_shape, int64), requires_grad=False)
        last = self.shape[-1]; outer = self.numel() // last
        out_vals = []
        for i in range(outer):
            chunk = self._arr._data[i*last:(i+1)*last]
            out_vals.append(_py_max(range(last), key=lambda j: chunk[j]))
        out_shape = tuple(self.shape[:-1]) + ((1,) if keepdim else ())
        return Tensor(_np.ndarray(out_vals, out_shape, int64), requires_grad=False)
    def transpose(self, dim0, dim1):
        axes = list(range(self.ndim)); axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        out_arr = self._arr.transpose(tuple(axes))
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        inv_axes = [0]*len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i
        def _backward(grad):
            _accum(self, grad.transpose(tuple(inv_axes)))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="transpose")
    def t(self):
        if self.ndim < 2:
            return self
        return self.transpose(self.ndim-2, self.ndim-1)
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        out_arr = self._arr.reshape(shape)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        orig = self.shape
        def _backward(grad):
            _accum(self, grad.reshape(orig))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="view")
    def reshape(self, *shape):
        return self.view(*shape)
    def contiguous(self):
        return self
    def squeeze(self, dim=None):
        if dim is None:
            new_shape = tuple(s for s in self.shape if s != 1)
        else:
            ax = dim if dim >= 0 else self.ndim + dim
            new_shape = list(self.shape)
            if new_shape[ax] == 1:
                del new_shape[ax]
            new_shape = tuple(new_shape)
        return self.view(new_shape)
    def unsqueeze(self, dim):
        ax = dim if dim >= 0 else self.ndim + 1 + dim
        new_shape = list(self.shape); new_shape.insert(ax, 1)
        return self.view(tuple(new_shape))
    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out_arr = _np.ndarray(_np._expand_data(self._arr._data, self._arr.shape, tuple(shape)), tuple(shape), self.dtype)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            _accum(self, _unbroadcast(grad, self.shape))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="expand")
    def float(self):
        return self.to(float32)
    def to(self, dtype=None, device=None):
        if dtype is None:
            return self
        dt = dtype
        if dt is _py_float:
            dt = float32
        elif dt is _py_int:
            dt = int64
        elif isinstance(dt, str):
            dt = _np.dtype(dt)
        if dt == self.dtype:
            return self
        out_arr = self._arr.astype(dt)
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            _accum(self, grad.astype(self.dtype))
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="to")
    def cpu(self):
        return self
    def cuda(self, device=None):
        return self
    def detach(self):
        return Tensor(self._arr.copy(), requires_grad=False)
    def clone(self):
        out_arr = self._arr.copy()
        if not _grad_enabled or not self.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            _accum(self, grad)
        return Tensor(out_arr, requires_grad=True, _prev=(self,), _backward_fn=_backward, _op="clone")
    def tolist(self):
        return self._arr.tolist()
    def item(self):
        return self._arr.item()
    def numpy(self):
        return self._arr.copy()
    def __float__(self):
        return _py_float(self._arr.item())
    def __int__(self):
        return _py_int(self._arr.item())
    def copy_(self, other):
        if isinstance(other, Tensor):
            src = other._arr
        elif isinstance(other, _np.ndarray):
            src = other
        elif isinstance(other, (list, tuple)):
            src = _np.array(other, dtype=self.dtype)
        else:
            src = _np.ndarray(other, (), self.dtype)
        if src.shape != self.shape:
            if src.size == 1:
                self._arr._data = [src._data[0]] * self.numel()
                return self
            raise ValueError("copy_ shape %s vs %s" % (src.shape, self.shape))
        self._arr._data = [_np._cast_value(v, self.dtype) for v in src._data]
        return self
    def zero_(self):
        self._arr._data = [0] * self.numel()
        return self
    def fill_(self, value):
        self._arr._data = [_np._cast_value(value, self.dtype)] * self.numel()
        return self
    def normal_(self, mean=0.0, std=1.0):
        arr = _normal_array(self.shape, mean, std)
        self._arr._data = list(arr._data)
        return self
    def requires_grad_(self, val=True):
        self.requires_grad = bool(val)
        return self
    def backward(self, gradient=None):
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad_arr = _np.ones(self.shape, dtype=self.dtype)
        elif isinstance(gradient, Tensor):
            grad_arr = gradient._arr
        elif isinstance(gradient, _np.ndarray):
            grad_arr = gradient
        else:
            grad_arr = _np.array(gradient)
        topo = []; visited = set()
        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)
        build(self)
        if self.grad is None:
            self.grad = grad_arr
        else:
            self.grad = self.grad + grad_arr
        for v in reversed(topo):
            if v._backward_fn is not None and v.grad is not None:
                v._backward_fn(v.grad)

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)

def tensor(data, dtype=None, requires_grad=False):
    if isinstance(data, Tensor):
        return Tensor(data._arr.copy(), dtype=dtype, requires_grad=requires_grad)
    if isinstance(data, _np.ndarray):
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)
    return Tensor(_np.array(data, dtype=dtype), requires_grad=requires_grad)
def zeros(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dt = dtype if dtype is not None else float32
    if dt is _py_float:
        dt = float32
    return Tensor(_np.zeros(tuple(shape), dtype=dt), requires_grad=requires_grad)
def ones(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    dt = dtype if dtype is not None else float32
    if dt is _py_float:
        dt = float32
    return Tensor(_np.ones(tuple(shape), dtype=dt), requires_grad=requires_grad)
def empty(*shape, dtype=None, requires_grad=False):
    return zeros(*shape, dtype=dtype, requires_grad=requires_grad)
def arange(*args, dtype=None):
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = tuple(args[0])
    return Tensor(_np.arange(*args, dtype=dtype), requires_grad=False)
def eye(n, dtype=None):
    return Tensor(_np.eye(n, dtype=dtype), requires_grad=False)
def cat(tensors, dim=0):
    arrs = [t._arr if isinstance(t, Tensor) else _np.array(t) for t in tensors]
    out_arr = _np.concatenate(arrs, axis=dim)
    req = any(isinstance(t, Tensor) and t.requires_grad for t in tensors)
    if not _grad_enabled or not req:
        return Tensor(out_arr, requires_grad=False)
    shapes = [a.shape for a in arrs]
    def _backward(grad):
        offset = 0
        for t, shp in zip(tensors, shapes):
            if isinstance(t, Tensor) and t.requires_grad:
                slices = [slice(None)] * grad.ndim
                slices[dim] = slice(offset, offset + shp[dim])
                _accum(t, grad[tuple(slices)])
            offset += shp[dim]
    return Tensor(out_arr, requires_grad=True, _prev=tuple(t for t in tensors if isinstance(t, Tensor)), _backward_fn=_backward, _op="cat")
def stack(tensors, dim=0):
    exp = [t.unsqueeze(dim) if isinstance(t, Tensor) else _to_tensor(t).unsqueeze(dim) for t in tensors]
    return cat(exp, dim=dim)
def clamp(input, min_val=None, max_val=None):
    return _to_tensor(input).clamp(min_val, max_val)
def round(input):
    return _to_tensor(input).round()
def rsqrt(input):
    return _to_tensor(input).rsqrt()
def abs(input):
    return _to_tensor(input).abs()
def exp(input):
    t = _to_tensor(input)
    vals = [math.exp(v) for v in t._arr._data]
    out = _np.ndarray([struct.unpack("f", struct.pack("f", v))[0] for v in vals], tuple(t.shape), float32)
    return Tensor(out, requires_grad=False)
def log(input):
    t = _to_tensor(input)
    vals = [math.log(v) if v > 0 else -1e30 for v in t._arr._data]
    out = _np.ndarray([struct.unpack("f", struct.pack("f", v))[0] for v in vals], tuple(t.shape), float32)
    return Tensor(out, requires_grad=False)

class _NoGrad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled; _grad_enabled = False; return self
    def __exit__(self, *a):
        global _grad_enabled; _grad_enabled = self._prev; return False
    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            global _grad_enabled
            prev = _grad_enabled; _grad_enabled = False
            try:
                return fn(*args, **kwargs)
            finally:
                _grad_enabled = prev
        return wrapper
def no_grad():
    return _NoGrad()
class _EnableGrad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled; _grad_enabled = True; return self
    def __exit__(self, *a):
        global _grad_enabled; _grad_enabled = self._prev; return False
    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            global _grad_enabled
            prev = _grad_enabled; _grad_enabled = True
            try:
                return fn(*args, **kwargs)
            finally:
                _grad_enabled = prev
        return wrapper
def enable_grad():
    return _EnableGrad()
def inference_mode():
    return _NoGrad()

def _to_serializable(obj):
    if isinstance(obj, Tensor):
        return {"__laser_tensor__": True, "shape": list(obj.shape), "dtype": obj.dtype.name, "data": list(obj._arr._data)}
    if isinstance(obj, _np.ndarray):
        return {"__laser_bumpy__": True, "shape": list(obj.shape), "dtype": obj.dtype.name, "data": list(obj._data)}
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj
def _from_serializable(obj):
    if isinstance(obj, dict) and obj.get("__laser_tensor__"):
        dt = _np.dtype(obj["dtype"])
        return Tensor(_np.ndarray(list(obj["data"]), tuple(obj["shape"]), dt), requires_grad=False)
    if isinstance(obj, dict) and obj.get("__laser_bumpy__"):
        dt = _np.dtype(obj["dtype"])
        return _np.ndarray(list(obj["data"]), tuple(obj["shape"]), dt)
    if isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    return obj
def save(obj, f):
    data = _to_serializable(obj)
    if isinstance(f, str):
        with open(f, "wb") as fh:
            pickle.dump(data, fh, protocol=4)
    else:
        pickle.dump(data, f, protocol=4)
def load(f, map_location=None, weights_only=None):
    if isinstance(f, str):
        with open(f, "rb") as fh:
            data = pickle.load(fh)
    else:
        data = pickle.load(f)
    return _from_serializable(data)

class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)
    def __setattr__(self, name, value):
        if name in ("_parameters", "_modules", "training"):
            object.__setattr__(self, name, value); return
        if isinstance(value, Parameter):
            self._parameters[name] = value; object.__setattr__(self, name, value); return
        if isinstance(value, Module):
            self._modules[name] = value; object.__setattr__(self, name, value); return
        object.__setattr__(self, name, value)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def parameters(self):
        for v in self._parameters.values():
            yield v
        for m in self._modules.values():
            yield from m.parameters()
    def named_parameters(self, prefix=""):
        for k, v in self._parameters.items():
            yield (prefix + "." + k if prefix else k), v
        for mk, m in self._modules.items():
            pfx = prefix + "." + mk if prefix else mk
            yield from m.named_parameters(prefix=pfx)
    def state_dict(self):
        return {k: v for k, v in self.named_parameters()}
    def load_state_dict(self, sd):
        for k, v in self.named_parameters():
            if k in sd:
                src = sd[k]
                if isinstance(src, Tensor):
                    v.copy_(src)
                elif isinstance(src, _np.ndarray):
                    v.copy_(Tensor(src))
                else:
                    v.copy_(Tensor(_np.array(src)))
    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
    def eval(self):
        return self.train(False)
    def float(self):
        for _, p in self.named_parameters():
            if p.dtype != float32:
                p._arr = p._arr.astype(float32)
        return self
    def to(self, dtype=None, device=None):
        if dtype is not None:
            for _, p in self.named_parameters():
                p._arr = p._arr.astype(dtype)
        return self
    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules:
            for m in modules:
                self.append(m)
    def append(self, module):
        idx = str(len(self._list))
        self._list.append(module)
        self._modules[idx] = module
        object.__setattr__(self, idx, module)
    def __len__(self):
        return len(self._list)
    def __getitem__(self, idx):
        return self._list[idx]
    def __iter__(self):
        return iter(self._list)
    def parameters(self):
        for m in self._list:
            yield from m.parameters()
    def named_parameters(self, prefix=""):
        for i, m in enumerate(self._list):
            pfx = (prefix + "." + str(i)) if prefix else str(i)
            yield from m.named_parameters(prefix=pfx)
    def train(self, mode=True):
        self.training = mode
        for m in self._list:
            m.train(mode)
        return self

class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.weight = Parameter(Tensor(_normal_array((out_features, in_features), 0.0, 0.02)))
        if bias:
            self.bias = Parameter(Tensor(_np.zeros((out_features,), dtype=float32)))
        else:
            self.bias = None
    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings; self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor(_normal_array((num_embeddings, embedding_dim), 0.0, 0.02)))
    def forward(self, idx):
        idx_arr = idx._arr if isinstance(idx, Tensor) else _np.array(idx)
        flat_idx = [int(v) for v in idx_arr._data]
        out_data = []
        w = self.weight._arr; D = self.embedding_dim
        for ix in flat_idx:
            base = ix * D
            out_data.extend(w._data[base:base+D])
        out_shape = tuple(idx_arr.shape) + (D,)
        out_arr = _np.ndarray(out_data, out_shape, float32)
        if not _grad_enabled or not self.weight.requires_grad:
            return Tensor(out_arr, requires_grad=False)
        def _backward(grad):
            gw = _np.zeros(self.weight.shape, dtype=float32)
            for n, ix in enumerate(flat_idx):
                base_g = n * D; base_w = ix * D
                for d in range(D):
                    gw._data[base_w + d] += grad._data[base_g + d]
            _accum(self.weight, gw)
        return Tensor(out_arr, requires_grad=True, _prev=(self.weight,), _backward_fn=_backward, _op="embedding")

class _Init:
    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0):
        if isinstance(tensor, Tensor):
            tensor.normal_(mean, std)
        return tensor
    @staticmethod
    def zeros_(tensor):
        if isinstance(tensor, Tensor):
            tensor.zero_()
        return tensor
    @staticmethod
    def ones_(tensor):
        if isinstance(tensor, Tensor):
            tensor.fill_(1.0)
        return tensor
    @staticmethod
    def constant_(tensor, val):
        if isinstance(tensor, Tensor):
            tensor.fill_(val)
        return tensor
    @staticmethod
    def xavier_uniform_(tensor, gain=1.0):
        return _Init.normal_(tensor, 0.0, 0.02)
init = _Init()

class _Utils:
    @staticmethod
    def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
        params = list(parameters)
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            for v in p.grad._data:
                total += v * v
        total_norm = math.sqrt(total) if total > 0 else 0.0
        clip_coef = max_norm / (total_norm + 1e-6) if total_norm > max_norm else 1.0
        if clip_coef < 1.0:
            for p in params:
                if p.grad is not None:
                    p.grad._data = [v * clip_coef for v in p.grad._data]
        return total_norm
utils = _Utils()

def _relu_fn(x):
    t = _to_tensor(x)
    out_arr = _np.ndarray([v if v > 0 else 0.0 for v in t._arr._data], tuple(t.shape), float32)
    if not _grad_enabled or not t.requires_grad:
        return Tensor(out_arr, requires_grad=False)
    a_arr = t._arr
    def _backward(grad):
        _accum(t, _np.ndarray([g if a > 0 else 0.0 for g, a in zip(grad._data, a_arr._data)], tuple(grad.shape), grad.dtype))
    return Tensor(out_arr, requires_grad=True, _prev=(t,), _backward_fn=_backward, _op="relu")

def _cross_entropy_fn(logits, target, reduction="none"):
    lg = _to_tensor(logits); tg = _to_tensor(target)
    N, C = lg.shape[0], lg.shape[1]
    logits_arr = lg._arr; target_data = [int(v) for v in tg._arr._data]
    out_vals = []; soft = []
    for n in range(N):
        row = logits_arr._data[n*C:(n+1)*C]
        m = _py_max(row)
        exps = [math.exp(v - m) for v in row]
        s = _py_sum(exps)
        probs = [e / s for e in exps]
        soft.extend(probs)
        out_vals.append(-math.log(probs[target_data[n]] + 1e-30))
    soft_arr = _np.ndarray(soft, (N, C), float32)
    if reduction == "mean":
        out_arr = _np.ndarray([_py_sum(out_vals)/N], (), float32)
    elif reduction == "sum":
        out_arr = _np.ndarray([_py_sum(out_vals)], (), float32)
    else:
        out_arr = _np.ndarray([struct.unpack("f", struct.pack("f", v))[0] for v in out_vals], (N,), float32)
    if not _grad_enabled or not lg.requires_grad:
        return Tensor(out_arr, requires_grad=False)
    def _backward(grad):
        gvals = []
        if reduction == "none":
            for n in range(N):
                gn = grad._data[n]
                for c in range(C):
                    gvals.append(gn * (soft_arr._data[n*C + c] - (1.0 if c == target_data[n] else 0.0)))
        elif reduction == "mean":
            gn = grad._data[0] / N
            for n in range(N):
                for c in range(C):
                    gvals.append(gn * (soft_arr._data[n*C + c] - (1.0 if c == target_data[n] else 0.0)))
        else:
            gn = grad._data[0]
            for n in range(N):
                for c in range(C):
                    gvals.append(gn * (soft_arr._data[n*C + c] - (1.0 if c == target_data[n] else 0.0)))
        _accum(lg, _np.ndarray(gvals, (N, C), float32))
    return Tensor(out_arr, requires_grad=True, _prev=(lg,), _backward_fn=_backward, _op="cross_entropy")

def _sdpa_fn(q, k, v, is_causal=False, attn_mask=None, dropout_p=0.0):
    Q = _to_tensor(q); K = _to_tensor(k); V = _to_tensor(v)
    B, H, T, Dh = Q.shape
    scale = 1.0 / math.sqrt(Dh)
    Q_arr, K_arr, V_arr = Q._arr, K._arr, V._arr
    K_t = K_arr.transpose(tuple(list(range(K_arr.ndim-2)) + [K_arr.ndim-1, K_arr.ndim-2]))
    scores = _np.matmul(Q_arr, K_t)
    scores = _np.ndarray([x * scale for x in scores._data], tuple(scores.shape), float32)
    if is_causal:
        B_, H_, T_, _ = scores.shape
        new_data = list(scores._data)
        for b in range(B_):
            for h in range(H_):
                base = (b*H_ + h)*T_*T_
                for i in range(T_):
                    for j in range(T_):
                        if j > i:
                            new_data[base + i*T_ + j] = -1e9
        scores = _np.ndarray(new_data, tuple(scores.shape), float32)
    P = _softmax_last_axis(scores)
    out_arr = _np.matmul(P, V_arr)
    req = Q.requires_grad or K.requires_grad or V.requires_grad
    if not _grad_enabled or not req:
        return Tensor(out_arr, requires_grad=False)
    def _backward(grad):
        def _t2(arr):
            return arr.transpose(tuple(list(range(arr.ndim-2)) + [arr.ndim-1, arr.ndim-2]))
        grad_P = _np.matmul(grad, _t2(V_arr))
        grad_V = _np.matmul(_t2(P), grad)
        last = P.shape[-1]; outer = P.size // last
        grad_scores_data = []
        for i in range(outer):
            p_row = P._data[i*last:(i+1)*last]
            gp_row = grad_P._data[i*last:(i+1)*last]
            s = _py_sum([gp * p for gp, p in zip(gp_row, p_row)])
            for gp, p in zip(gp_row, p_row):
                grad_scores_data.append(p * (gp - s))
        grad_scores = _np.ndarray(grad_scores_data, tuple(P.shape), float32)
        if is_causal:
            B_, H_, T_, _ = grad_scores.shape
            nd = list(grad_scores._data)
            for b in range(B_):
                for h in range(H_):
                    base = (b*H_ + h)*T_*T_
                    for i in range(T_):
                        for j in range(T_):
                            if j > i:
                                nd[base + i*T_ + j] = 0.0
            grad_scores = _np.ndarray(nd, tuple(grad_scores.shape), float32)
        grad_Q = _np.ndarray([x * scale for x in _np.matmul(grad_scores, K_arr)._data], tuple(grad_scores.shape[:-1] + (K_arr.shape[-1],)), float32) if False else _np.ndarray([x * scale for x in _np.matmul(grad_scores, K_arr)._data], tuple(_np.matmul(grad_scores, K_arr).shape), float32)
        grad_K = _np.ndarray([x * scale for x in _np.matmul(_t2(grad_scores), Q_arr)._data], tuple(_np.matmul(_t2(grad_scores), Q_arr).shape), float32)
        if Q.requires_grad:
            _accum(Q, _unbroadcast(grad_Q, Q.shape))
        if K.requires_grad:
            _accum(K, _unbroadcast(grad_K, K.shape))
        if V.requires_grad:
            _accum(V, _unbroadcast(grad_V, V.shape))
    return Tensor(out_arr, requires_grad=True, _prev=(Q, K, V), _backward_fn=_backward, _op="sdpa")

_functional = types.ModuleType("laser.nn.functional")
_functional.relu = _relu_fn
_functional.cross_entropy = _cross_entropy_fn
_functional.scaled_dot_product_attention = _sdpa_fn
_functional.softmax = lambda x, dim=-1: Tensor(_softmax_last_axis(_to_tensor(x)._arr), requires_grad=False)
_functional.log_softmax = lambda x, dim=-1: Tensor(_np.ndarray([math.log(v + 1e-30) for v in _softmax_last_axis(_to_tensor(x)._arr)._data], tuple(_to_tensor(x).shape), float32), requires_grad=False)
_functional.linear = lambda input, weight, bias=None: (_to_tensor(input) @ _to_tensor(weight).t() + _to_tensor(bias)) if bias is not None else (_to_tensor(input) @ _to_tensor(weight).t())

_nn = types.ModuleType("laser.nn")
_nn.Module = Module; _nn.Parameter = Parameter; _nn.Linear = Linear; _nn.Embedding = Embedding; _nn.ModuleList = ModuleList; _nn.init = init; _nn.utils = utils; _nn.functional = _functional
nn = _nn

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            self.param_groups = []
            for g in params:
                pg = dict(g); pg.setdefault("lr", lr); pg.setdefault("betas", betas); pg.setdefault("eps", eps); pg.setdefault("weight_decay", weight_decay)
                pg["params"] = list(g["params"]); self.param_groups.append(pg)
        else:
            self.param_groups = [{"params": list(params), "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}]
        self.state = {}; self._step = 0
    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad._data = [0.0]*len(p.grad._data)
    def step(self, closure=None):
        self._step += 1
        for group in self.param_groups:
            lr = group["lr"]; b1, b2 = group["betas"]; eps = group["eps"]; wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad; pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {"m": _np.zeros(p.shape, dtype=float32), "v": _np.zeros(p.shape, dtype=float32), "step": 0}
                st = self.state[pid]; st["step"] += 1; t = st["step"]
                if wd != 0.0:
                    p._arr._data = [x - lr*wd*x for x in p._arr._data]
                m = st["m"]; v = st["v"]
                m._data = [b1*mi + (1-b1)*gi for mi, gi in zip(m._data, g._data)]
                v._data = [b2*vi + (1-b2)*gi*gi for vi, gi in zip(v._data, g._data)]
                bc1 = 1 - (b1 ** t); bc2 = 1 - (b2 ** t)
                new_data = []
                for pi_, mi, vi in zip(p._arr._data, m._data, v._data):
                    new_data.append(pi_ - lr * (mi / bc1) / (math.sqrt(vi / bc2) + eps))
                p._arr._data = [struct.unpack("f", struct.pack("f", x))[0] for x in new_data]
    def state_dict(self):
        return {"param_groups": self.param_groups}
    def load_state_dict(self, sd):
        self.param_groups = sd.get("param_groups", self.param_groups)

_optim = types.ModuleType("laser.optim")
_optim.AdamW = AdamW; _optim.Adam = AdamW
optim = _optim
sys.modules["laser.nn"] = _nn
sys.modules["laser.nn.functional"] = _functional
sys.modules["laser.optim"] = _optim
_nn_init = types.ModuleType("laser.nn.init")
_nn_init.normal_ = init.normal_; _nn_init.zeros_ = init.zeros_; _nn_init.ones_ = init.ones_; _nn_init.constant_ = init.constant_
sys.modules["laser.nn.init"] = _nn_init
_nn_utils = types.ModuleType("laser.nn.utils")
_nn_utils.clip_grad_norm_ = utils.clip_grad_norm_
sys.modules["laser.nn.utils"] = _nn_utils

def __getattr__(name):
    if name == "float":
        return float32
    if name == "int":
        return int64
    if name == "bool":
        return bool_
    if name == "long":
        return int64
    if name == "double":
        return float64
    if name == "half":
        return float32
    raise AttributeError("module 'laser' has no attribute %r" % name)

__all__ = ["Tensor", "Parameter", "tensor", "zeros", "ones", "empty", "arange", "eye", "cat", "stack", "clamp", "round", "rsqrt", "abs", "exp", "log", "no_grad", "enable_grad", "inference_mode", "manual_seed", "seed", "set_num_threads", "save", "load", "nn", "optim", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool_"]
