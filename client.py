"""Simple interactive client for floppy-cognition models and riskytensor files.

Usage:
  python3 -I client.py                     # interactive REPL
  python3 -I client.py MODEL.pt|MODEL.riskytensor|TINY.BIN
  python3 -I client.py --info FILE.riskytensor
  python3 -I client.py --verify FILE.riskytensor
  python3 -I client.py --convert MODEL.pt [OUT.riskytensor]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _load_model(path):
    import train2
    return train2.load_model(path)

def _ask(model, question):
    import train2
    ans = train2.answer_many(model, [question])
    return ans.get(question, "")

def interactive(model_path=None):
    model = None
    if model_path:
        print("Loading %s ..." % model_path, flush=True)
        t0 = time.time()
        model = _load_model(model_path)
        print("Loaded in %.2fs  (%s)" % (time.time() - t0, model.c), flush=True)

    print()
    print("floppy-cognition client")
    print("  /load PATH     load a model (.pt / .riskytensor / .BIN)")
    print("  /info PATH     show riskytensor manifest")
    print("  /verify PATH   verify riskytensor integrity")
    print("  /convert SRC [DST]  convert .pt -> .riskytensor")
    print("  /eval [N]      run evaluate() and show N failures")
    print("  /quit          exit")
    print("  anything else  asked as a question to the loaded model")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "quit", "exit"):
            break

        if line.startswith("/load "):
            path = line[6:].strip()
            try:
                t0 = time.time()
                model = _load_model(path)
                print("Loaded %s in %.2fs  (%s)" % (path, time.time() - t0, model.c))
            except Exception as e:
                print("load failed:", e)
            continue

        if line.startswith("/info "):
            path = line[6:].strip()
            try:
                import riskytensor, json
                m = riskytensor.info(path)
                m.pop("_payload_sha", None)
                print(json.dumps(m, indent=2))
            except Exception as e:
                print("info failed:", e)
            continue

        if line.startswith("/verify "):
            path = line[8:].strip()
            try:
                import riskytensor
                riskytensor.verify(path)
            except Exception as e:
                print("verify failed:", e)
            continue

        if line.startswith("/convert "):
            parts = line[9:].strip().split()
            if not parts:
                print("usage: /convert SRC.pt [DST.riskytensor]")
                continue
            src = parts[0]
            dst = parts[1] if len(parts) > 1 else None
            try:
                import riskytensor
                riskytensor.convert(src, out_path=dst)
            except Exception as e:
                print("convert failed:", e)
            continue

        if line.startswith("/eval"):
            if model is None:
                print("no model loaded")
                continue
            show = 10
            bits = line.split()
            if len(bits) > 1:
                try:
                    show = int(bits[1])
                except ValueError:
                    pass
            import train2
            tot, report = train2.evaluate(model, show=show)
            print("eval %.1f%%  %s" % (100 * tot, report))
            continue

        # treat as a question
        if model is None:
            print("no model loaded — use /load PATH first")
            continue
        t0 = time.time()
        answer = _ask(model, line)
        dt = time.time() - t0
        print("  -> %s  (%.2fs)" % (answer, dt))

def main():
    if len(sys.argv) == 1:
        interactive()
        return

    if sys.argv[1] == "--info" and len(sys.argv) == 3:
        import riskytensor, json
        m = riskytensor.info(sys.argv[2])
        m.pop("_payload_sha", None)
        print(json.dumps(m, indent=2))
        return

    if sys.argv[1] == "--verify" and len(sys.argv) == 3:
        import riskytensor
        sys.exit(0 if riskytensor.verify(sys.argv[2]) else 1)

    if sys.argv[1] == "--convert":
        import riskytensor
        if len(sys.argv) < 3:
            print("usage: client.py --convert MODEL.pt [OUT.riskytensor]")
            sys.exit(1)
        src = sys.argv[2]
        dst = sys.argv[3] if len(sys.argv) > 3 else None
        riskytensor.convert(src, out_path=dst)
        return

    # otherwise treat first arg as model path and enter interactive
    interactive(sys.argv[1])

if __name__ == "__main__":
    main()
