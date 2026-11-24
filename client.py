"""Simple interactive client for floppy-cognition models and riskytensor files."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_NEW_TOKENS = 16
CTX = 128


def _load_model(path):
    import train2
    return train2.load_model(path)


def _prepare_prompt(question):
    """Truncate prompt so prompt + max_new fits in CTX. Returns (fed_string, truncated)."""
    # reserve room for generation
    budget = CTX - MAX_NEW_TOKENS - 1  # -1 for the newline we append
    q = question
    truncated = False
    if len(q) > budget:
        q = q[-budget:]
        truncated = True
    return q, truncated


def _generate(model, question, max_new=MAX_NEW_TOKENS):
    import train2
    import laser as torch

    fed, truncated = _prepare_prompt(question)
    if truncated:
        print('  [warn] prompt truncated to last %d chars (CTX=%d)' % (len(fed), CTX), flush=True)
    print('  [fed] len=%d  %r' % (len(fed), fed[:60] + ('...' if len(fed) > 60 else '')), flush=True)

    model.eval()
    ids = train2.encode(fed + '\n')
    x = torch.tensor([ids])
    out = []
    t0 = time.time()
    print('  generating', end='', flush=True)
    try:
        for step in range(max_new):
            logits = model(x)
            nxt = int(logits[0, -1].argmax(-1).item())
            if nxt <= 1:
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
            print('.', end='', flush=True)
        print(' done (%.1fs, %d toks)' % (time.time() - t0, len(out)), flush=True)
    except Exception as e:
        print(' error: %s' % e, flush=True)
    finally:
        model.train()
    return train2.decode(out)


def interactive(model_path=None):
    model = None
    if model_path:
        print('Loading %s ...' % model_path, flush=True)
        t0 = time.time()
        model = _load_model(model_path)
        print('Loaded in %.2fs  (%s)' % (time.time() - t0, model.c), flush=True)

    print()
    print('floppy-cognition client')
    print('  /load PATH     load a model')
    print('  /info PATH     riskytensor manifest')
    print('  /verify PATH   verify riskytensor')
    print('  /convert SRC [DST]')
    print('  /eval [N]      evaluate (slow)')
    print('  /quit')
    print('  other text     generate (max %d new tokens)' % MAX_NEW_TOKENS)
    print('  NOTE: pure-Python laser ~seconds/token')
    print()

    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ('/quit', '/exit', 'quit', 'exit'):
            break

        if line.startswith('/load '):
            path = line[6:].strip()
            try:
                t0 = time.time()
                model = _load_model(path)
                print('Loaded %s in %.2fs  (%s)' % (path, time.time() - t0, model.c), flush=True)
            except Exception as e:
                print('load failed:', e, flush=True)
            continue

        if line.startswith('/info '):
            path = line[6:].strip()
            try:
                import riskytensor, json
                m = riskytensor.info(path)
                m.pop('_payload_sha', None)
                print(json.dumps(m, indent=2))
            except Exception as e:
                print('info failed:', e)
            continue

        if line.startswith('/verify '):
            path = line[8:].strip()
            try:
                import riskytensor
                riskytensor.verify(path)
            except Exception as e:
                print('verify failed:', e)
            continue

        if line.startswith('/convert '):
            parts = line[9:].strip().split()
            if not parts:
                print('usage: /convert SRC.pt [DST.riskytensor]')
                continue
            src = parts[0]
            dst = parts[1] if len(parts) > 1 else None
            try:
                import riskytensor
                riskytensor.convert(src, out_path=dst)
            except Exception as e:
                print('convert failed:', e)
            continue

        if line.startswith('/eval'):
            if model is None:
                print('no model loaded')
                continue
            show = 5
            bits = line.split()
            if len(bits) > 1:
                try:
                    show = int(bits[1])
                except ValueError:
                    pass
            print('evaluating (slow)...', flush=True)
            import train2
            tot, report = train2.evaluate(model, show=show)
            print('eval %.1f%%  %s' % (100 * tot, report), flush=True)
            continue

        if model is None:
            print('no model loaded — use /load PATH first')
            continue

        answer = _generate(model, line)
        print('  -> %s' % (answer if answer else '(empty / immediate EOS)'), flush=True)


def main():
    if len(sys.argv) == 1:
        interactive()
        return
    if sys.argv[1] == '--info' and len(sys.argv) == 3:
        import riskytensor, json
        m = riskytensor.info(sys.argv[2])
        m.pop('_payload_sha', None)
        print(json.dumps(m, indent=2))
        return
    if sys.argv[1] == '--verify' and len(sys.argv) == 3:
        import riskytensor
        sys.exit(0 if riskytensor.verify(sys.argv[2]) else 1)
    if sys.argv[1] == '--convert':
        import riskytensor
        if len(sys.argv) < 3:
            print('usage: client.py --convert MODEL.pt [OUT.riskytensor]')
            sys.exit(1)
        riskytensor.convert(sys.argv[2], out_path=sys.argv[3] if len(sys.argv) > 3 else None)
        return
    interactive(sys.argv[1])


if __name__ == '__main__':
    main()
