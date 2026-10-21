# Vowel-harmony reference cohort (Exp 56)

Per-speaker pretonic-lowering descriptors for the 1,153 speakers (9 corpora,
~735k vowel tokens, 193 h) measured in the accents project's Exp 56, produced
by the same chain as `features.vowel_harmony`. Use it as a reference
distribution for placing new speakers, or to reproduce the regional contrast
without re-running the extraction.

```python
from pt_br_accent_toolbox.features.vowel_harmony import load_reference_cohort
ref = load_reference_cohort()                  # list[dict], one per speaker
ref = load_reference_cohort(gate_ok_only=True) # the 477 usable speakers
```

Region labels were joined **after** measurement; nothing in the pipeline saw
them.

## Headline numbers (gate_ok speakers)

| macro | n | OI mean | lowering cluster |
|-------|---|---------|------------------|
| NE | 168 | 0.646 | 82.1% |
| SE (incl. S) | 112 | 0.357 | 22.3% |
| unk | 196 | 0.432 | 51.5% |
| NEUTRAL (Globo) | 1 | 0.122 | 0% |

## Columns

| Column | Meaning |
|--------|---------|
| `corpus`, `speaker` | source corpus and speaker key used in Exp 56 |
| `region`, `macro`, `region_source` | state/region label, macro-region (NE / SE / NEUTRAL / unk), where it came from (`dir` = corpus folder layout, `csv` = corpus metadata, `none`) |
| `transcript_source` | `given` (corpus transcript) or `asr` (greedy XLSR CTC) |
| `n_tokens`, `n_T_low_both`, `n_T_mid_both` | measured vowel tokens; pretonic targets before an open / close-mid stressed vowel |
| `OI` | openness index: mean h before an open stressed vowel (primary score) |
| `HI` | harmony contrast: OI minus mean h before a close-mid stressed vowel |
| `HI_lowmid` | as HI, restricted to [ɛ ɔ] triggers (excludes /a/) |
| `open_rate_low`, `OR_delta` | share of T_low tokens with h > 0.5; minus the T_mid share |
| `grad_slope` | slope of h on the trigger's own normalized F1 (gradient vs categorical) |
| `RI` | raising index before a high stressed vowel (country-wide, weak regional marker) |
| `ashman_low` | Ashman's D of the T_low h distribution (bimodality diagnostic, not a feature) |
| `gap_e_eps`, `d_e_eps` | stressed [ɛ]−[e] F1 gap (Hz) and Cohen's d: the anchor QA |
| `anchor_ok` | anchors pass (gap ≥ 60 Hz, d ≥ 0.7, [a]−[e] ≥ 150 Hz) |
| `gate_ok` | anchor_ok and ≥ 15 T_low and ≥ 8 T_mid tokens; only these are trustworthy |
| `harmony_score` | OI standardized over the gate_ok cohort ((OI − median) / sd) |
| `p_lowering`, `cluster_gmm` | posterior / label of the forced 2-component GMM split |

The 2-way split is **imposed, not discovered**: BIC prefers one component
(ΔBIC = −42) and the OI distribution has a single peak. Treat OI as a continuous
rate; the cluster columns are there for comparison with the paper only.
