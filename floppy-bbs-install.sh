#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat bbs/part-*.b64 | tr -d '\n' | base64 -d > floppy-bbs.sh
chmod +x floppy-bbs.sh
echo "OK floppy-bbs.sh $(wc -c < floppy-bbs.sh) bytes — run: bash floppy-bbs.sh"
