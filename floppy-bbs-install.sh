#!/usr/bin/env bash
# Rebuild floppy-bbs.sh from plain text parts (reliable; no base64).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d bbs ]]; then
  echo "error: bbs/ directory missing (need bbs/part-*.sh.txt)" >&2
  exit 1
fi
shopt -s nullglob
parts=(bbs/part-*.sh.txt)
if [[ ${#parts[@]} -eq 0 ]]; then
  echo "error: no bbs/part-*.sh.txt files found" >&2
  exit 1
fi
cat "${parts[@]}" > floppy-bbs.sh
chmod +x floppy-bbs.sh
bytes=$(wc -c < floppy-bbs.sh | tr -d ' ')
echo "OK floppy-bbs.sh ${bytes} bytes — run: bash floppy-bbs.sh"
if [[ "$bytes" -lt 5000 ]]; then
  echo "WARNING: file looks too small; parts may be incomplete." >&2
  exit 2
fi
