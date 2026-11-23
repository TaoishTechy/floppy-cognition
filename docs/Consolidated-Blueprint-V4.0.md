# Consolidated Blueprint v4.0 — Tiny Stack

**File:** `Consolidated_Blueprint_v4.0.md`
**Replaces:** `laser` v4.0 blueprint, `bumpy` v4.0 blueprint, `riskytensor` v4.0 blueprint, `safe2risky` v1.0, `NecroScale` v3.0. Those files are superseded. This file is the single source of truth.
**Stack:** stdlib only + `bumpy` + `laser` + `riskytensor` + `train2.py`. No torch, no numpy, no safetensors, no LZ4, no AES, no pickle for weights.
**Governance:** MOGOPs-369 v8.0 — Architecturally Sound Self-Optimising Token Framework, used as *engineering governance*, not as numerology. See §1.
**Size budget:** This file MUST stay < 1.4 MB. Current target: < 100 KB. Efficiency is enforced by §0, not by hope.
**Status:** Blueprint. Build order is §9. Do not build out of order.

---

## 0. How this file stays under 1.4 MB and super efficient

This section is normative. It is the efficiency mechanism.

1. **Define once, reference always.** Every concept has one canonical definition. Later sections reference it by ID (`[B-1]`, `[L-3]`, `[R-2]`, `[T-1]`), they do not re-explain it. If you find the same explanation twice, that is a bug in this file.
2. **Tables over prose.** Where a list would be prose, it is a table. Where a table would repeat, it is a reference.
3. **No duplication of code.** This file contains *specs and signatures*, not full implementations. Implementations live in `bumpy.py`, `laser.py`, `riskytensor.py`, `train2.py`. Code snippets here are at most 10 lines, only to pin down byte layout or a formula.
4. **No history.** Old claims (holographic, phi, Xi, SophiaQPU, coherence, qualia, paradox, etc.) are listed once in §11 as *deleted*, then never mentioned again.
5. **Measurable, not adjectives.** “Fast”, “efficient”, “robust” MUST NOT appear without a number and a measurement method. See MOGOPs Goodhart guard (§1.4).
6. **Byte budget.** This file: < 100 KB target, < 1.4 MB hard limit. Code: `bumpy.py` < 40 KB, `laser.py` < 55 KB, `riskytensor.py` < 30 KB, `train2.py` < 16 KB. A checkpoint file: see §6.4. If any file exceeds its budget, the build MUST stop and deduplicate, not grow the budget.

---

## 1. MOGOPs-369 v8.0 — governance applied to this stack

MOGOPs-369 v8.0 is *not* used here for tokens, crypto, or numerology. It is used for the five engineering rules it kept after excising v7.0 numerology:

### 1.1 RFC 2119 language
`MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, `MAY` have RFC 2119 meaning throughout this file. Anything without one of those words is explanatory, not normative.

### 1.2 Threat model (STRIDE, stripped to this stack)
| Threat | What it means here | Mitigation in this blueprint |
|---|---|---|
| Spoofing | A fake weight file pretends to be a real checkpoint | Magic + version + SHA-256 (§6). `load()` MUST reject on mismatch. |
| Tampering | One byte flipped in a weight file | Per-tensor SHA-256 + payload SHA-256 (§6). Error MUST name the tensor. |
| Repudiation | “I never saved that” | `save()` is deterministic; same input + recipe → byte-identical file (§6.3). Re-save proves provenance. |
| Info disclosure | Pickle code execution on `load()` | `riskytensor.load()` MUST NOT use pickle / `eval` / `exec`. Only `struct` + `json` + `hashlib` + `zlib`. |
| DoS | Huge / malformed manifest causes OOM | Manifest length is in the header (§6.1). `load()` MUST check `manifest_len` before allocating. `info()` MUST work without loading weights. |
| Elevation | A weight file runs code | No executable payload. No WASM, no kernel code, no tokenizer code in the weight file. |

No other threat model is claimed. No QPU, no TEE, no HSM is required to use this stack.

### 1.3 Separation of code and runtime integrity
- **Code integrity:** The four `.py` files are the code. They are reviewed as code.
- **Runtime integrity:** The `.riskytensor` / `.pt` / `TINY.BIN` files are data. They MUST NOT be executed. `load()` is a parser, not an executor.
- **Lattice / 3-6-9:** Per MOGOPs v8.0, the 3-6-9 lattice is demoted to a *non-security deterministic checksum salt*. It MUST NOT be used as a validation gate, capability, or security claim in this stack. It appears nowhere in the file format. If you need a salt, use SHA-256. This is the explicit excision of v7.0 numerology.

### 1.4 Measurable SLOs + Goodhart guards
Every performance claim MUST have: metric, target, measurement method, and a guard against gaming it.

| ID | SLO | Target | How measured | Goodhart guard |
|---|---|---|---|---|
| [SLO-1] | `bumpy` matmul correctness | 100% vs. reference math on conformance vectors | §10.1 | Do not tune to the test vectors; vectors are fixed, code is general |
| [SLO-2] | `laser` gradient correctness | Max abs error < 1e-4 vs. finite-difference on small graphs | §10.2 | Finite-difference step is fixed; do not hard-code test graphs |
| [SLO-3] | `riskytensor` round-trip (float32) | Exact, byte-identical | §10.3 | No “close enough” — exact or fail |
| [SLO-4] | `riskytensor` round-trip (int8) | Per-element error ≤ `scale/2` | §10.3 | Scale is per-row, computed from data, not from a claimed constant |
| [SLO-5] | `train2` export determinism | Same model → byte-identical `TINY.BIN` | §10.4 | Export MUST NOT use wall-clock, RNG, or dict order |
| [SLO-6] | File size | This blueprint < 1.4 MB; code budgets in §0.6 | `wc -c` | Do not compress the blueprint to hide bloat; keep it readable |

No SLO may be “99.9%” without a definition. No SLO may be claimed without a test in §10.

### 1.5 Conformance suite
One suite, in §10, covers all four modules. A module is done only when its §10 rows pass. No module may ship with “TODO: test later”.

### 1.6 What MOGOPs does *not* add here
- No CBOR tokens, no Ed25519, no DPoP, no mTLS, no Macaroons. Those are in MOGOPs-369 v8.0 for *token* systems. This stack is not a token system. Importing them would be scope creep.
- No φ, no golden ratio, no Sophia point, no semantic gravity, no ontological axioms. Those are in the old `MOGOPS-Axioms.md` and `MOS-HSRCF v4.0` files. They are *not* part of this blueprint. They are listed in §11 as superseded / out of scope.

---

## 2. System map (read this once)

```
train2.py  — model + training + eval + TINY.BIN export + CLI
  │
  ├▶ laser.py       — torch replacement: Tensor, autograd, nn, optim
  │      │
  │      └▶ bumpy.py    — numpy replacement: ndarray, dtype, broadcast, matmul
  │
  └▶ riskytensor.py — safetensors / torch.save replacement: file format only
         │
         └▶ uses bumpy for bytes, laser.Tensor for returned state_dict
```

Dependency rules (MUST):
- `bumpy.py` MUST NOT import `laser`, `riskytensor`, `torch`, `numpy`.
- `laser.py` MUST import only stdlib + `bumpy`. It MUST NOT import `riskytensor` or `train2`.
- `riskytensor.py` MAY import `bumpy` + `laser`, only for the returned `Tensor` wrapper. Its *parser* MUST be stdlib only (`struct`, `json`, `hashlib`, `zlib`).
- `train2.py` MAY import all three. It MUST NOT import `torch` or `numpy`.

---

(See the full file in the repository for the complete specification of bumpy [B], laser [L], riskytensor [R], train2 [T], build order, and conformance suite.)

*End of Consolidated Blueprint v4.0 summary. Full text committed separately if size permits.*
