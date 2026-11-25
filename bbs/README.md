# floppy-bbs parts

Rebuild the operator shell:

```bash
chmod +x ./floppy-bbs-install.sh
./floppy-bbs-install.sh
# expect: OK floppy-bbs.sh 25xxx bytes
bash floppy-bbs.sh --cmd health
```

Parts are plain text (`part-*.sh.txt`), concatenated in sorted order. Not base64.
