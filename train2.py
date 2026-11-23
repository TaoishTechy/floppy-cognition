"""Train TINY version 2 (laser + bumpy, no torch / numpy).

  python3 -I train2.py bench
  python3 -I train2.py train MINUTES OUT.pt [--from TINY.BIN] [--resume X] [--lr X] [--decay X] [--eval-every X] [--seed N]
  python3 -I train2.py eval  MODEL.pt|MODEL.riskytensor|TINY.BIN [show]
  python3 -I train2.py export MODEL.pt|MODEL.riskytensor TINY.BIN

Optimised for the floppy-cognition stack:
- riskytensor checkpoint support
- deterministic seed control
- rich progress logging (loss, tokens/s, ETA, sample Q&A)
- same model architecture and CLI surface as before
"""
import sys, os, time, math, random, struct, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bumpy as np
import laser as torch
nn = torch.nn
F = torch.nn.functional

try:
    import riskytensor
    _HAS_RISKY = True
except ImportError:
    _HAS_RISKY = False

import gen2

VOCAB, CTX = 97, 128
torch.set_num_threads(2)


def encode(s):
    return [1 if c == '\n' else ord(c) - 30 for c in s]


def decode(ids):
    return ''.join('\n' if i == 1 else chr(i + 30) for i in ids if i > 0)


class Cfg:
    def __init__(self, D=96, L=4, H=4, HID=256):
        self.D, self.L, self.H, self.HID = D, L, H, HID

    def __repr__(self):
        return 'D=%d L=%d H=%d HID=%d' % (self.D, self.L, self.H, self.HID)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype) * self.g


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.n1, self.n2 = Norm(c.D), Norm(c.D)
        self.q = nn.Linear(c.D, c.D, bias=False)
        self.k = nn.Linear(c.D, c.D, bias=False)
        self.v = nn.Linear(c.D, c.D, bias=False)
        self.o = nn.Linear(c.D, c.D, bias=False)
        self.f1 = nn.Linear(c.D, c.HID, bias=False)
        self.f2 = nn.Linear(c.HID, c.D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.c.H
        h = self.n1(x)
        sp = lambda t: t.view(B, T, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(self.q(h)), sp(self.k(h)), sp(self.v(h)), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(VOCAB, c.D)
        self.pos = nn.Embedding(CTX, c.D)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.L)])
        self.nf = Norm(c.D)
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.normal_(self.pos.weight, std=0.05)
        for b in self.blocks:
            for m in (b.q, b.k, b.v, b.o, b.f1, b.f2):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.nf(x) @ self.emb.weight.t()


# ---------------------------------------------------------------- old TINY.BIN
def load_old_bin(path, D=96, L=3, HID=192):
    b = open(path, 'rb').read()
    off = [0]

    def table(r, c):
        s = struct.unpack_from('<f', b, off[0])[0]
        w = np.frombuffer(b, np.int8, r * c, off[0] + 4).reshape(r, c).astype(np.float32) * s
        off[0] += 4 + r * c
        return torch.tensor(w)

    def gains():
        g = np.frombuffer(b, '<f4', D, off[0]).copy()
        off[0] += 4 * D
        return torch.tensor(g)

    m = Model(Cfg(D, L, 4, HID))
    with torch.no_grad():
        m.emb.weight.copy_(table(VOCAB, D))
        m.pos.weight.copy_(table(CTX, D))
        for blk in m.blocks:
            blk.n1.g.copy_(gains())
            for lin in (blk.q, blk.k, blk.v, blk.o):
                lin.weight.copy_(table(D, D))
            blk.n2.g.copy_(gains())
            blk.f1.weight.copy_(table(HID, D))
            blk.f2.weight.copy_(table(D, HID))
        m.nf.g.copy_(gains())
    assert off[0] == len(b), 'size mismatch'
    return m


def grow(old, c):
    m = Model(c)
    with torch.no_grad():
        m.emb.weight.copy_(old.emb.weight)
        m.pos.weight.copy_(old.pos.weight)
        m.nf.g.copy_(old.nf.g)
        for i, blk in enumerate(m.blocks):
            if i < len(old.blocks):
                o = old.blocks[i]
                for n in ('n1', 'n2'):
                    getattr(blk, n).g.copy_(getattr(o, n).g)
                for n in ('q', 'k', 'v', 'o'):
                    getattr(blk, n).weight.copy_(getattr(o, n).weight)
                h = o.f1.weight.shape[0]
                blk.f1.weight[:h] = o.f1.weight
                blk.f2.weight.zero_()
                blk.f2.weight[:, :h] = o.f2.weight
            else:
                blk.o.weight.zero_()
                blk.f2.weight.zero_()
    return m


# ---------------------------------------------------------------- data
_POOL = []


def batch(rng, n):
    if not _POOL:
        ex = []
        for _ in range(n * 32):
            q, a, _ = gen2.sample(rng)
            ids = encode(q + '\n' + a + '\n')[:CTX]
            mask = [0] * (len(q) + 1) + [1] * (len(ids) - len(q) - 1)
            ex.append((ids, mask))
        ex.sort(key=lambda e: len(e[0]))
        chunks = [ex[i:i + n] for i in range(0, len(ex), n)]
        rng.shuffle(chunks)
        _POOL.extend(chunks)
    seqs, masks = zip(*_POOL.pop())
    T = max(len(s) for s in seqs)
    x = torch.zeros(n, T, dtype=torch.long)
    m = torch.zeros(n, T)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float)
    return x, m


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    l = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction='none')
    w = m[:, 1:].reshape(-1)
    return (l * w).sum() / w.sum()


# ---------------------------------------------------------------- evaluation
STOP = set('a an the it is are was of in on and to its it\'s they there people speak use money capital city '
           'called comes after before has have says say lives eats one two answer number letter letters starts '
           'ends with month day the'.split())


def words_of(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def eval_set(n_per_kind=60, seed=123):
    rng = random.Random(seed)
    out = []
    for f in [f for f in gen2.TXT if f.get('test_q')]:
        out.append((f['test_q'][0], f['answers'], f.get('kind', 'facts')))
    rng.shuffle(out)
    out = out[:120]
    for name, _, _ in gen2.GENS:
        if name == 'facts':
            continue
        seen = set()
        while sum(1 for o in out if o[2] == name) < n_per_kind:
            q, a, _ = gen2.sample(rng, True, name)
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, [a], name))
    return out


UNKNOWN_START = ('i do not', 'i have never', 'sorry', 'hmm')


def correct(q, gen, refs, kind=None):
    if kind == 'unknown' and refs[0].lower().startswith(UNKNOWN_START):
        if not gen.lower().startswith(UNKNOWN_START):
            return False
        names = [w for w in words_of(refs[0]) if w in words_of(q)]
        return all(w in words_of(gen) for w in names)
    g = set(words_of(gen))
    qw = set(words_of(q))
    for r in refs:
        rw = words_of(r)
        if rw and rw[0] in ('yes', 'no') and (not words_of(gen) or words_of(gen)[0] != rw[0]):
            continue
        key = [w for w in rw if w not in STOP and w not in qw]
        if not key:
            key = [w for w in rw if w not in STOP] or rw
        if all(w in g for w in key):
            return True
    return False


@torch.no_grad()
def answer_many(model, qs):
    model.eval()
    res = {}
    bylen = collections.defaultdict(list)
    for q in qs:
        bylen[len(q)].append(q)
    for n, group in bylen.items():
        x = torch.tensor([encode(q + '\n') for q in group])
        done = [False] * len(group)
        outs = [[] for _ in group]
        while x.shape[1] < CTX and not all(done):
            nxt = model(x)[:, -1].argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    if t <= 1:
                        done[i] = True
                    else:
                        outs[i].append(t)
            x = torch.cat([x, nxt[:, None]], 1)
        for q, o in zip(group, outs):
            res[q] = decode(o)
    model.train()
    return res


def evaluate(model, show=0):
    es = eval_set()
    ans = answer_many(model, [e[0] for e in es])
    by = collections.defaultdict(lambda: [0, 0])
    bad = []
    for q, refs, kind in es:
        ok = correct(q, ans[q], refs, kind)
        by[kind][0] += ok
        by[kind][1] += 1
        if not ok:
            bad.append((kind, q, ans[q], refs[0]))
    tot = sum(v[0] for v in by.values()) / max(1, sum(v[1] for v in by.values()))
    line = '  '.join('%s %d%%' % (k, 100 * v[0] / max(1, v[1])) for k, v in by.items())
    if show:
        random.Random(0).shuffle(bad)
        for b in bad[:show]:
            print('   x [%s] %s -> %s   (want: %s)' % b)
    return tot, line


# ---------------------------------------------------------------- export / checkpoints
def export(model, path):
    c = model.c
    sd = {k: v.float() for k, v in model.state_dict().items()}
    with open(path, 'wb') as f:
        f.write(b'TNY2' + struct.pack('<4h', c.D, c.L, c.H, c.HID))

        def table(w):
            w = w.numpy()
            for row in w:
                s = max(float(np.abs(row).max()) / 127.0, 1e-8)
                f.write(struct.pack('<f', s))
                f.write(np.clip(np.round(row / s), -127, 127).astype(np.int8).tobytes())

        def gains(g):
            f.write(g.numpy().astype('<f4').tobytes())

        table(sd['emb.weight'])
        table(sd['pos.weight'])
        for i in range(c.L):
            p = 'blocks.%d.' % i
            gains(sd[p + 'n1.g'])
            for n in 'qkvo':
                table(sd[p + n + '.weight'])
            gains(sd[p + 'n2.g'])
            table(sd[p + 'f1.weight'])
            table(sd[p + 'f2.weight'])
        gains(sd['nf.g'])
    return os.path.getsize(path)


def quantized_copy(model):
    m = Model(model.c)
    m.load_state_dict(model.state_dict())
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                s = (p.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-8)
                p.copy_(torch.clamp(torch.round(p / s), -127, 127) * s)
    return m


def load_model(path):
    lower = path.lower()
    if lower.endswith('.bin'):
        return load_old_bin(path)
    if lower.endswith('.riskytensor') and _HAS_RISKY:
        sd = riskytensor.load(path)
        info = riskytensor.info(path)
        cfg_dict = info.get('metadata', {}).get('cfg', {})
        if not cfg_dict:
            emb = sd.get('emb.weight')
            D = emb.shape[1] if emb is not None else 96
            cfg_dict = {'D': D, 'L': 4, 'H': 4, 'HID': 256}
        m = Model(Cfg(**cfg_dict))
        m.load_state_dict(sd)
        return m
    ck = torch.load(path, map_location='cpu')
    m = Model(Cfg(**ck['cfg']))
    m.load_state_dict(ck['sd'])
    return m


def save_model(m, path):
    lower = path.lower()
    if lower.endswith('.riskytensor') and _HAS_RISKY:
        riskytensor.save(m.state_dict(), path, recipe=None,
                         metadata={'cfg': vars(m.c)})
        return os.path.getsize(path)
    torch.save(dict(cfg=vars(m.c), sd=m.state_dict()), path)
    return os.path.getsize(path)


# ---------------------------------------------------------------- training loop
def train(minutes, out, start=None, cfg=Cfg(), peak=2e-3, bs=64, log_every=50,
          resume=None, decay_frac=None, eval_every=0, seed=None):
    """Train for `minutes` wall-clock time with rich progress output."""
    if seed is not None:
        torch.manual_seed(int(seed))
        rng = random.Random(int(seed))
    else:
        torch.manual_seed(0)
        rng = random.Random(int(time.time()))

    if resume:
        model = load_model(resume)
        print('resumed from', resume, flush=True)
    else:
        model = grow(load_old_bin(start), cfg) if start else Model(cfg)

    cfg = model.c
    n_params = sum(p.numel() for p in model.parameters())
    print('=' * 60, flush=True)
    print('model   ', cfg, flush=True)
    print('params  ', n_params, flush=True)
    print('batch   ', bs, flush=True)
    print('peak lr ', peak, flush=True)
    print('minutes ', minutes, flush=True)
    print('out     ', out, flush=True)
    print('=' * 60, flush=True)

    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [dict(params=decay, weight_decay=0.01),
         dict(params=other, weight_decay=0.0)],
        lr=peak, betas=(0.9, 0.98))

    t0 = time.time()
    total = minutes * 60.0
    step = 0
    run = None
    eval_no = 0
    best_loss = float('inf')
    tokens_seen = 0
    last_log_t = t0
    last_log_step = 0

    while True:
        el = time.time() - t0
        if el >= total:
            break

        warm = min(1.0, (step + 1) / 200)
        if decay_frac is None:
            lr = peak * warm * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * el / total)))
        else:
            left = (total - el) / (decay_frac * total)
            lr = peak * warm * min(1.0, max(0.0, left))

        if eval_every and el > (eval_no + 1) * eval_every * 60:
            eval_no += 1
            save_model(model, out)
            tot, line = evaluate(quantized_copy(model))
            print('  EVAL at %.1f min: %.1f%%  %s' % (el / 60, 100 * tot, line), flush=True)

        for g in opt.param_groups:
            g['lr'] = lr

        x, msk = batch(rng, bs)
        loss = loss_fn(model, x, msk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        loss_val = loss.item()
        run = loss_val if run is None else 0.98 * run + 0.02 * loss_val
        if loss_val < best_loss:
            best_loss = loss_val
        tokens_seen += int(msk.sum().item()) if hasattr(msk, 'sum') else bs * CTX

        if step % log_every == 0:
            now = time.time()
            dt = now - last_log_t
            dstep = step - last_log_step
            tok_s = (tokens_seen / max(1e-6, now - t0))
            eta = max(0.0, total - el)
            print('step %5d | %5.0fs | lr %.5f | loss %.4f (best %.4f) | %.0f tok/s | ETA %.0fs' %
                  (step, el, lr, run, best_loss, tok_s, eta), flush=True)
            last_log_t = now
            last_log_step = step

            # occasional live sample
            if step % (log_every * 4) == 0:
                try:
                    q, a, k = gen2.sample(rng)
                    pred = answer_many(model, [q]).get(q, '')
                    print('  sample [%s] Q: %s' % (k, q), flush=True)
                    print('           A: %s' % a, flush=True)
                    print('           → %s' % pred, flush=True)
                except Exception:
                    pass

    save_model(model, out)
    elapsed = time.time() - t0
    print('=' * 60, flush=True)
    print('saved   ', out, flush=True)
    print('steps   ', step, flush=True)
    print('time    ', '%.0fs' % elapsed, flush=True)
    print('best    ', '%.4f' % best_loss, flush=True)
    print('tokens  ', tokens_seen, flush=True)
    print('tok/s   ', '%.0f' % (tokens_seen / max(1e-6, elapsed)), flush=True)
    print('=' * 60, flush=True)
    return model


# ---------------------------------------------------------------- CLI
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    def arg(k, default=None):
        return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default

    if cmd == 'bench':
        rng = random.Random(0)
        for c in [Cfg(96, 3, 4, 192), Cfg(96, 4, 4, 256), Cfg(112, 4, 4, 224)]:
            m = Model(c)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for i in range(13):
                if i == 3:
                    t = time.time()
                x, k = batch(rng, 96)
                l = loss_fn(m, x, k)
                opt.zero_grad()
                l.backward()
                opt.step()
            print(c, sum(p.numel() for p in m.parameters()),
                  '%.3f s/step' % ((time.time() - t) / 10), flush=True)

    elif cmd == 'train':
        m = train(
            float(sys.argv[2]), sys.argv[3],
            start=arg('--from'),
            resume=arg('--resume'),
            peak=float(arg('--lr', 2e-3)),
            decay_frac=float(arg('--decay')) if arg('--decay') else None,
            eval_every=float(arg('--eval-every', 0)),
            seed=int(arg('--seed')) if arg('--seed') else None,
        )
        print('\n--- final evaluation ---', flush=True)
        tot, line = evaluate(m, show=0)
        print('eval (float) %.1f%%  %s' % (100 * tot, line), flush=True)
        tot, line = evaluate(quantized_copy(m), show=15)
        print('eval (8-bit) %.1f%%  %s' % (100 * tot, line), flush=True)

    elif cmd == 'eval':
        m = load_model(sys.argv[2])
        show = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        tot, line = evaluate(m, show=show)
        print('eval %.1f%%  %s' % (100 * tot, line))

    elif cmd == 'export':
        n = export(load_model(sys.argv[2]), sys.argv[3])
        print('wrote', n, 'bytes')

    else:
        print('unknown command:', cmd)
        print(__doc__)
        sys.exit(1)
