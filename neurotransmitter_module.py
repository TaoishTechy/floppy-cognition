"""neurotransmitter_module.py — 144 heuristic neurotransmitter equations

SPECULATIVE / HEURISTIC CONCEPTUAL EQUATIONS ONLY — NOT VALIDATED CLINICAL FORMULAS.
Variables are normalized relative indices (0-1 unless noted).
This is a computational toy model for simulation / agent affective state,
NOT medical advice, NOT diagnosis, NOT treatment, NOT a measurement of real
brain chemistry.

All outputs are mapped to [0,1] with a sigmoid, except Reward RPE (Eq 2)
which is signed in [-1,1].

Usage:
    from neurotransmitter_module import NeurochemState, NeurochemicalEngine144
    eng = NeurochemicalEngine144()
    results = eng.evaluate_all()  # list of 144 dicts
    print(len(eng.to_matrix()))   # 12 rows of 12
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

EPS = 1e-6
SOPHIA_POINT = (math.sqrt(5) - 1) / 2  # ~0.6180339887


def _clip01(x: float) -> float:
    if isinstance(x, float) and math.isnan(x):
        raise ValueError("NaN in NeurochemState")
    return max(0.0, min(1.0, float(x)))


def _sig(x: float, k: float = 4.0, x0: float = 0.5) -> float:
    z = k * (x - x0)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _safe_div(num: float, den: float) -> float:
    return num / (den + EPS)


@dataclass
class NeurochemState:
    """Normalized relative indices in [0,1]. Not measured concentrations."""

    DA_tonic: float = 0.5
    DA_phasic: float = 0.2
    DA_expected: float = 0.2
    DA_meso: float = 0.5
    DA_nigro: float = 0.5
    DA_striatal: float = 0.5
    DA_PFC: float = 0.5
    D1: float = 0.6
    D1_striatal: float = 0.6
    D1_PFC: float = 0.6
    D2_sens: float = 0.5
    D2_down: float = 0.3
    D2_striatal: float = 0.5
    D3_tone: float = 0.3
    DAT: float = 0.4
    COMT: float = 0.4
    repetition: float = 0.5

    HT_tonic: float = 0.5
    HT1A: float = 0.6
    HT1A_sens: float = 0.6
    HT2A: float = 0.4
    HT2A_sens: float = 0.4
    HT3: float = 0.2
    HT_desc: float = 0.5
    SERT: float = 0.4

    NE: float = 0.5
    NE_alpha1: float = 0.3
    NE_alpha2: float = 0.5
    NE_beta: float = 0.3
    CRH: float = 0.2
    cortisol: float = 0.2
    cortisol_feedback: float = 0.2
    NO: float = 0.5

    ACh: float = 0.5
    nAChR: float = 0.6
    mAChR: float = 0.5
    mAChR_muscle: float = 0.6
    mAChR_heart: float = 0.5
    AChE: float = 0.4
    ACh_striatal: float = 0.5
    amyloid: float = 0.2

    GABA: float = 0.6
    GABAA: float = 0.6
    GABAB: float = 0.5
    GLU: float = 0.4
    GLU_NMDA: float = 0.4
    GLU_AMPA: float = 0.5
    GLU_EAAT: float = 0.4
    Ca: float = 0.5
    Mg: float = 0.5
    CaMKII: float = 0.5
    CREB: float = 0.5
    SUBSTANCE_P: float = 0.1

    END: float = 0.4
    MOR: float = 0.5
    KOR: float = 0.2

    OXT: float = 0.5
    OXTR: float = 0.5

    HIS: float = 0.3
    H1: float = 0.4
    H2: float = 0.3

    ADO: float = 0.3
    A1: float = 0.5
    A2A: float = 0.3

    AEA: float = 0.4
    CB1: float = 0.5

    BDNF: float = 0.5
    TrkB: float = 0.5
    exercise: float = 0.5
    sleep: float = 0.5
    inflammation: float = 0.2
    CGRP: float = 0.3
    PPI: float = 0.1

    caffeine: float = 0.0
    leptin: float = 0.6
    NPY: float = 0.3
    stress: float = 0.2
    anxiety: float = 0.2

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, _clip01(getattr(self, f.name)))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class NeurochemicalEngine144:
    """144 heuristic state indices. SPECULATIVE only."""

    DOMAINS = [
        "dopamine", "serotonin", "norepinephrine", "acetylcholine",
        "gaba", "glutamate", "endorphins", "oxytocin",
        "histamine", "adenosine", "endocannabinoids", "bdnf",
    ]

    NAMES = [
        "Motivation", "Reward_RPE", "Pleasure", "Addiction", "Motor",
        "WorkingMemory", "Exploration", "Procrastination", "Salience", "Habit",
        "Creativity", "Fatigue",
        "Mood", "Anxiety", "Patience", "Sleep", "Appetite", "Social",
        "Impulsivity", "Obsession", "Pain", "Nausea", "Sexual", "Resilience",
        "Arousal", "Attention", "Stress", "Alertness", "FightFlight", "Focus",
        "Anxiety", "BloodPressure", "Energy", "Tremor", "Memory", "Fatigue",
        "Attention", "Memory", "Learning", "Muscle", "REM_Sleep", "Wake",
        "Plasticity", "Alzheimer_Index", "Focus", "Neuropathy", "Vagal", "Salivation",
        "Calm", "Anxiety", "Sleep", "Seizure", "Inhibition", "Relaxation",
        "Focus", "Pain", "Alcohol_Index", "Tremor", "Spasticity", "Mood",
        "Learning", "Memory", "Excitotoxicity", "Plasticity", "Anxiety", "Seizure",
        "Pain", "Focus", "Neurodegeneration", "Addiction", "Sleep", "Mood",
        "Pain", "Pleasure", "Euphoria", "Reward", "Stress", "Social",
        "Addiction", "Analgesia", "Mood", "Withdrawal", "Appetite", "Immune",
        "Bonding", "Trust", "Empathy", "Social", "Love", "Maternal",
        "Anxiety", "Stress", "PairBond", "Generosity", "Wound", "Autism_Index",
        "Wake", "Alertness", "Sleep", "Itch", "Allergy", "Nausea",
        "Appetite", "Focus", "Anxiety", "Migraine", "Gastric", "Fatigue",
        "Sleep", "Fatigue", "Wake", "Neuroprotection", "Vasodilation", "Anxiety",
        "Focus", "Memory", "Pain", "Seizure", "Heart", "Recovery",
        "Relaxation", "Pain", "Appetite", "Memory", "Mood", "Anxiety",
        "Neuroplasticity", "Sleep", "Nausea", "Reward", "Inflammation", "Euphoria",
        "Plasticity", "Learning", "Memory", "Mood", "Resilience", "Neurogenesis",
        "Recovery", "Addiction", "Anxiety", "Cognitive", "Aging", "Wellbeing",
    ]

    def __init__(self, state: NeurochemState | None = None):
        self.state = state or NeurochemState()

    def evaluate_all(self) -> list:
        c = self.state
        DA = c.DA_tonic
        HT = c.HT_tonic
        NEv = c.NE
        DA_D2 = c.D2_sens
        END_MOR = c.END * c.MOR
        ADO_A1 = c.ADO * c.A1
        ADO_A2A = c.ADO * c.A2A

        raw = [0.0] * 144

        raw[0] = _safe_div(c.DA_tonic * c.D2_sens, 1 + c.DAT + c.COMT)
        raw[1] = c.DA_phasic - c.DA_expected
        raw[2] = c.DA_meso * (1 - c.D3_tone) * END_MOR
        raw[3] = _safe_div((c.DA_phasic ** 2) * c.D2_down, c.HT1A_sens + c.GABA)
        raw[4] = _safe_div(c.DA_nigro * c.D1_striatal, 1 + c.ACh_striatal)
        raw[5] = _safe_div(c.DA_PFC * c.D1_PFC, 1 + c.NE_alpha2 + c.cortisol)
        raw[6] = _safe_div(c.DA_tonic * (1 - c.HT2A_sens), 1 + c.NE_alpha1)
        raw[7] = _safe_div(1.0, c.DA_tonic * c.D2_sens + 1.0)
        raw[8] = _safe_div(c.DA_phasic * (1 + c.NE_alpha1), 1 + c.HT2A_sens)
        raw[9] = _safe_div(c.DA_striatal * c.D1 * c.repetition, 1 + c.D2_sens)
        raw[10] = c.DA_tonic * (1 - c.D2_striatal) * c.HT2A_sens
        raw[11] = _safe_div(ADO_A2A + c.GABA, c.DA_tonic + NEv)

        raw[12] = _safe_div(HT * c.HT1A_sens, 1 + c.HT2A_sens + c.stress)
        raw[13] = _safe_div(c.HT2A_sens * c.NE_alpha1, c.HT1A_sens + c.GABA)
        raw[14] = c.HT_tonic * (1 - c.DA_phasic)
        raw[15] = _safe_div(c.HT_tonic * c.HT2A_sens, 1 + NEv + c.HIS)
        raw[16] = _safe_div((1 - HT) * c.NPY, 1 + c.leptin)
        raw[17] = _safe_div(HT * c.OXTR, 1 + DA_D2)
        raw[18] = _safe_div(c.DA_phasic, c.HT_tonic + c.GABA)
        raw[19] = _safe_div(c.HT2A_sens * DA, c.HT1A_sens + c.SERT)
        raw[20] = _safe_div(c.HT_desc * (1 - c.END), 1 + NEv)
        raw[21] = _safe_div(c.HT3 * (1 + c.ACh), 1 + c.GABA)
        raw[22] = _safe_div(HT * (1 - DA), 1 + c.OXT)
        raw[23] = _safe_div(c.HT1A_sens * c.BDNF, 1 + c.cortisol)

        raw[24] = _safe_div(NEv * c.NE_alpha1, 1 + c.GABA + c.ADO)
        raw[25] = _safe_div(c.NE_alpha2 * c.D1, 1 + c.HT2A_sens)
        raw[26] = _safe_div(NEv * c.CRH, 1 + c.cortisol_feedback)
        raw[27] = _safe_div(NEv * c.NE_beta, 1 + ADO_A1)
        raw[28] = _safe_div(NEv * c.NE_alpha1 * c.cortisol, 1 + c.GABA)
        raw[29] = _safe_div(c.NE_alpha2 * c.ACh, 1 + c.NE_beta)
        raw[30] = _safe_div(NEv * c.NE_alpha1 * c.HT2A_sens, 1 + c.GABA + c.HT1A_sens)
        raw[31] = _safe_div(NEv * c.NE_alpha1 * c.NE_beta, 1 + c.NO)
        raw[32] = _safe_div(NEv * c.NE_beta * DA, 1 + c.ADO)
        raw[33] = _safe_div(c.NE_beta * c.ACh, 1 + c.GABA)
        raw[34] = _safe_div(NEv * c.NE_beta * c.ACh, 1 + c.cortisol)
        raw[35] = _safe_div(c.ADO + c.GABA, NEv + DA)

        raw[36] = _safe_div(c.ACh * c.nAChR, 1 + c.GABA)
        raw[37] = _safe_div(c.ACh * c.mAChR * c.BDNF, 1 + c.AChE)
        raw[38] = _safe_div(c.ACh * DA * NEv, 1 + c.AChE)
        raw[39] = _safe_div(c.ACh * c.mAChR_muscle, 1 + c.AChE)
        raw[40] = _safe_div(c.ACh * c.mAChR, 1 + HT + NEv)
        raw[41] = _safe_div(c.ACh * c.nAChR * NEv, 1 + c.GABA)
        raw[42] = _safe_div(c.ACh * c.BDNF, 1 + c.AChE)
        raw[43] = _safe_div(c.ACh * (1 - c.amyloid), 1 + c.AChE)
        raw[44] = _safe_div(c.ACh * c.NE_alpha2, 1 + DA_D2)
        raw[45] = _safe_div(c.ACh * (1 - c.AChE), 1 + c.GABA)
        raw[46] = _safe_div(c.ACh * c.mAChR_heart, 1 + c.NE_beta)
        raw[47] = _safe_div(c.ACh * c.mAChR, 1 + NEv)

        raw[48] = _safe_div(c.GABA * c.GABAA, 1 + c.GLU + NEv)
        raw[49] = _safe_div(c.GLU, c.GABA * c.GABAA + c.HT1A_sens)
        raw[50] = _safe_div(c.GABA * c.GABAA, 1 + c.HIS + NEv)
        raw[51] = _safe_div(c.GLU, c.GABA * c.GABAA + 1.0)
        raw[52] = _safe_div(c.GABA * c.GABAB, 1 + c.GLU)
        raw[53] = _safe_div(c.GABA * (1 + c.ADO), 1 + NEv)
        raw[54] = _safe_div(c.GABA * (1 - c.GLU), 1 + DA)
        raw[55] = _safe_div(c.GABA * c.GABAA, 1 + c.SUBSTANCE_P)
        raw[56] = _safe_div(c.GABA * c.GABAA, 1 + c.GLU_NMDA)
        raw[57] = _safe_div(c.GLU, c.GABA + DA)
        raw[58] = _safe_div(c.GLU, c.GABA * c.GABAB)
        raw[59] = _safe_div(c.GABA * HT, 1 + c.GLU)

        raw[60] = _safe_div(c.GLU_NMDA * c.BDNF, 1 + c.GABA)
        raw[61] = _safe_div(c.GLU_AMPA * c.CaMKII, 1 + c.GABA)
        raw[62] = _safe_div(c.GLU_NMDA * c.Ca, 1 + c.GABA + c.Mg)
        raw[63] = _safe_div(c.GLU_NMDA * c.Ca * c.CREB, 1 + c.GABA)
        raw[64] = _safe_div(c.GLU * NEv, 1 + c.GABA)
        raw[65] = _safe_div(c.GLU * (1 - c.GABA), 1 + c.Mg)
        raw[66] = _safe_div(c.GLU_NMDA * c.SUBSTANCE_P, 1 + c.GABA)
        raw[67] = _safe_div(c.GLU_AMPA * DA, 1 + c.GABA)
        raw[68] = _safe_div(c.GLU * c.Ca, c.BDNF + c.GABA)
        raw[69] = _safe_div(c.GLU * DA, 1 + c.GABA + c.GLU_EAAT)
        raw[70] = _safe_div(c.GABA, c.GLU + c.HIS)
        raw[71] = _safe_div(c.GLU * (1 - c.BDNF), 1 + c.GABA)

        raw[72] = _safe_div(c.END * c.MOR, 1 + c.SUBSTANCE_P + c.GLU)
        raw[73] = _safe_div(c.END * c.MOR * DA, 1 + c.GABA)
        raw[74] = _safe_div(c.END * c.MOR * c.CB1, 1 + HT)
        raw[75] = _safe_div(c.END * DA * c.MOR, 1 + c.KOR)
        raw[76] = _safe_div(c.END * c.MOR, 1 + c.CRH + NEv)
        raw[77] = _safe_div(c.END * c.OXT, 1 + DA_D2)
        raw[78] = _safe_div(c.END * c.MOR * DA, 1 + c.KOR + c.GABA)
        raw[79] = _safe_div(c.END * c.MOR * c.NE_alpha2, 1 + c.GLU_NMDA)
        raw[80] = _safe_div(c.END * c.MOR * HT, 1 + c.KOR)
        raw[81] = _safe_div(c.KOR * (1 - c.MOR), c.END + DA)
        raw[82] = _safe_div(c.END * c.MOR, 1 + c.leptin)
        raw[83] = _safe_div(c.END * c.MOR, 1 + c.cortisol)

        raw[84] = _safe_div(c.OXT * c.OXTR, 1 + c.cortisol)
        raw[85] = _safe_div(c.OXT * HT, 1 + DA_D2)
        raw[86] = _safe_div(c.OXT * (1 + HT), 1 + NEv)
        raw[87] = _safe_div(c.OXT * c.END, 1 + c.anxiety)
        raw[88] = _safe_div(c.OXT * DA * HT, 1 + c.cortisol)
        raw[89] = _safe_div(c.OXT * DA, 1 + c.stress)
        raw[90] = _safe_div(c.OXT * c.GABA, 1 + NEv)
        raw[91] = _safe_div(c.OXT * (1 - c.cortisol), 1 + c.CRH)
        raw[92] = _safe_div(c.OXT * DA_D2, 1 + c.HT2A_sens)
        raw[93] = _safe_div(c.OXT * HT, 1 + c.D1)
        raw[94] = _safe_div(c.OXT * c.END, 1 + NEv)
        raw[95] = _safe_div(c.OXT * c.OXTR, 1 + DA + HT)

        raw[96] = _safe_div(c.HIS * c.H1, 1 + c.GABA + c.ADO)
        raw[97] = _safe_div(c.HIS * c.H1 * NEv, 1 + HT)
        raw[98] = _safe_div(c.GABA + c.ADO, c.HIS * c.H1)
        raw[99] = _safe_div(c.HIS * c.H1, 1 + c.GABA)
        raw[100] = _safe_div(c.HIS * c.H1, 1 + c.cortisol)
        raw[101] = _safe_div(c.HIS * c.H1 * c.ACh, 1 + c.GABA)
        raw[102] = _safe_div(c.HIS * c.H1, 1 + c.leptin)
        raw[103] = _safe_div(c.HIS * c.H1 * c.ACh, 1 + c.GABA)
        raw[104] = _safe_div(c.HIS * c.H1 * NEv, 1 + c.GABA)
        raw[105] = _safe_div(c.HIS * (1 - HT), 1 + c.CGRP)
        raw[106] = _safe_div(c.HIS * c.H2, 1 + c.PPI)
        raw[107] = _safe_div(c.ADO + c.GABA, c.HIS + NEv)

        raw[108] = _safe_div(c.ADO * c.A1, 1 + c.caffeine)
        raw[109] = _safe_div(c.ADO * c.A1, 1 + DA + NEv)
        raw[110] = _safe_div(DA + NEv, c.ADO * c.A1)
        raw[111] = _safe_div(c.ADO * c.A1, 1 + c.GLU)
        raw[112] = _safe_div(c.ADO * c.A2A, 1 + NEv)
        raw[113] = _safe_div(c.ADO * c.A1, 1 + c.GABA)
        raw[114] = _safe_div(c.ADO * c.A1, 1 + c.caffeine + DA)
        raw[115] = _safe_div(c.ADO * c.A1, 1 + c.ACh)
        raw[116] = _safe_div(c.ADO * c.A1, 1 + c.END)
        raw[117] = _safe_div(c.GLU, c.ADO * c.A1 + c.GABA)
        raw[118] = _safe_div(c.ADO * c.A1, 1 + c.NE_beta)
        raw[119] = _safe_div(c.ADO * c.A1 * c.sleep, 1 + DA)

        raw[120] = _safe_div(c.AEA * c.CB1, 1 + NEv)
        raw[121] = _safe_div(c.AEA * c.CB1, 1 + c.GLU + c.SUBSTANCE_P)
        raw[122] = _safe_div(c.AEA * c.CB1, 1 + c.leptin)
        raw[123] = _safe_div(c.AEA * c.CB1, 1 + c.ACh)
        raw[124] = _safe_div(c.AEA * c.CB1 * HT, 1 + DA_D2)
        raw[125] = _safe_div(c.AEA * c.CB1, 1 + c.GABA)
        raw[126] = _safe_div(c.AEA * c.CB1 * c.BDNF, 1 + c.stress)
        raw[127] = _safe_div(c.AEA * c.CB1, 1 + c.HIS)
        raw[128] = _safe_div(c.AEA * c.CB1, 1 + c.HT3)
        raw[129] = _safe_div(c.AEA * c.CB1 * DA, 1 + c.KOR)
        raw[130] = _safe_div(c.AEA * c.CB1, 1 + c.cortisol)
        raw[131] = _safe_div(c.AEA * c.CB1 * c.END, 1 + c.GABA)

        raw[132] = _safe_div(c.BDNF * c.TrkB, 1 + c.cortisol)
        raw[133] = _safe_div(c.BDNF * c.GLU_NMDA, 1 + c.GABA)
        raw[134] = _safe_div(c.BDNF * c.ACh, 1 + c.AChE)
        raw[135] = _safe_div(c.BDNF * HT, 1 + c.cortisol)
        raw[136] = _safe_div(c.BDNF * c.HT1A_sens, 1 + c.CRH)
        raw[137] = _safe_div(c.BDNF * c.exercise, 1 + c.stress)
        raw[138] = _safe_div(c.BDNF * c.sleep, 1 + c.inflammation)
        raw[139] = _safe_div(DA * (1 - c.BDNF), 1 + c.GABA)
        raw[140] = _safe_div((1 - c.BDNF) * NEv, 1 + c.GABA)
        raw[141] = _safe_div(c.BDNF * c.ACh * NEv, 1 + c.ADO)
        raw[142] = _safe_div(1.0, c.BDNF * c.TrkB + 1.0)
        raw[143] = _safe_div(c.BDNF * (DA + HT + c.OXT), 1 + c.cortisol + c.GLU)

        results = []
        for i, r in enumerate(raw):
            domain = self.DOMAINS[i // 12]
            if i == 1:
                val = max(-1.0, min(1.0, float(r)))
            else:
                val = _sig(float(r))
            results.append({
                "id": i + 1,
                "domain": domain,
                "name": self.NAMES[i],
                "raw": float(r),
                "value": float(val),
                "evidence_class": "SPECULATIVE",
                "protocol": "heuristic_eq_%d" % (i + 1),
                "actuator": "none",
            })
        assert len(results) == 144
        return results

    def values(self) -> list:
        return [d["value"] for d in self.evaluate_all()]

    def to_matrix(self):
        """12x12 list-of-lists (stdlib; no numpy)."""
        vals = self.values()
        return [vals[i * 12:(i + 1) * 12] for i in range(12)]

    def step_entropy_loop(self, dt: float = 0.01) -> list:
        vals = self.values()
        mean = sum(vals) / len(vals)
        return [v + dt * (-SOPHIA_POINT * (v - mean)) for v in vals]

    def neuro_seed_bytes(self, n: int = 16) -> bytes:
        """Educational only — NOT secure crypto."""
        import hashlib
        text = ",".join("%.6f" % v for v in self.values()).encode()
        h = hashlib.sha256(text).digest()
        out = bytearray()
        while len(out) < n:
            out.extend(h)
            h = hashlib.sha256(h).digest()
        return bytes(out[:n])

    def xor_demo(self, data: bytes) -> bytes:
        """XOR demo with neuro seed. NOT for real encryption."""
        seed = self.neuro_seed_bytes(len(data) if data else 16)
        return bytes(b ^ seed[i % len(seed)] for i, b in enumerate(data))

    def training_signals(self) -> dict:
        """Map a few indices to optional train2 knobs (all SPECULATIVE).

        Returns floats in sensible ranges for LR / exploration / patience.
        Does not change weights by itself — caller decides whether to use them.
        """
        v = {d["name"]: d["value"] for d in self.evaluate_all()}
        rpe = v.get("Reward_RPE", 0.0)
        return {
            "lr_scale": 0.5 + 0.5 * v.get("Motivation", 0.5),
            "exploration": v.get("Exploration", 0.5),
            "patience": v.get("Patience", 0.5),
            "focus": v.get("Focus", 0.5) if "Focus" in v else 0.5,
            "rpe": rpe,
            "wellbeing": v.get("Wellbeing", 0.5),
            "evidence_class": "SPECULATIVE",
        }


def run_self_tests() -> None:
    eng = NeurochemicalEngine144()
    res = eng.evaluate_all()
    assert len(res) == 144
    for r in res:
        assert r["evidence_class"] == "SPECULATIVE"
        assert r["actuator"] == "none"
        if r["id"] != 2:
            assert 0.0 <= r["value"] <= 1.0
        else:
            assert -1.0 <= r["value"] <= 1.0

    low = NeurochemicalEngine144(NeurochemState(cortisol=0.1)).evaluate_all()[5]["value"]
    high = NeurochemicalEngine144(NeurochemState(cortisol=0.9)).evaluate_all()[5]["value"]
    assert low > high

    low = NeurochemicalEngine144(NeurochemState(caffeine=0.9)).evaluate_all()[108]["value"]
    high = NeurochemicalEngine144(NeurochemState(caffeine=0.0)).evaluate_all()[108]["value"]
    assert low < high

    pos = NeurochemicalEngine144(NeurochemState(DA_phasic=0.8, DA_expected=0.2)).evaluate_all()[1]["value"]
    neg = NeurochemicalEngine144(NeurochemState(DA_phasic=0.1, DA_expected=0.5)).evaluate_all()[1]["value"]
    assert pos > 0 and neg < 0

    m = eng.to_matrix()
    assert len(m) == 12 and all(len(row) == 12 for row in m)
    assert len(eng.step_entropy_loop()) == 144
    demo = eng.xor_demo(b"hello")
    assert eng.xor_demo(demo) == b"hello"
    sig = eng.training_signals()
    assert sig["evidence_class"] == "SPECULATIVE"
    print("All neurotransmitter self-tests passed: 144 states, SPECULATIVE, actuator=none")


if __name__ == "__main__":
    run_self_tests()
    eng = NeurochemicalEngine144()
    for r in eng.evaluate_all()[:12]:
        print("%3d [%14s] %-16s raw=%.4f value=%.4f" % (
            r["id"], r["domain"], r["name"], r["raw"], r["value"]))
    print("Wellbeing (144) = %.4f" % eng.evaluate_all()[143]["value"])
    print("Matrix 12x12 ok; training_signals =", eng.training_signals())
