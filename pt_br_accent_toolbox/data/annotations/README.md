# Speaker annotations

Human labels for the three phonological markers the toolbox detects, on 115
natural Brazilian Portuguese speakers drawn from nine public corpora.
Regenerate with:

```bash
python tools/export_annotations.py --db /path/to/classifier_ui/annotations.db
```

The SQLite database written by the annotation UI is the source of truth; these
CSVs are a snapshot of it, versioned here so a fresh clone can reproduce the
labelled cohort without it.

## Files

| File | Grain | Rows |
|------|-------|------|
| `annotations.csv` | one per (dataset, speaker, annotator) — the raw labels | 118 |
| `speakers.csv` | one per (dataset, speaker) — majority label + agreement | 115 |
| `summary.csv` | counts per dataset × marker × value × source | 46 |
| `todo.csv` | speakers with at least one marker still unlabelled | 25 |

## Label vocabulary

| Marker | Values | Phenomenon |
|--------|--------|------------|
| `s_coda` | `sibilant`, `chiado`, `mixed` | alveolar [s] vs. postalveolar [ʃ] in coda |
| `r_coda` | `tap`, `carioca`, `caipira`, `mixed` | tap [ɾ] vs. fricative [x ʁ h] vs. retroflex [ɻ] |
| `dt_palat` | `palatalized`, `non-palatalized`, `mixed` | [tʃ dʒ] vs. [t d] before /i/ |

`mixed` means the speaker alternates between realizations within the sampled
audio — it is a label in its own right, not missing data. `unsure` (one row) is
the UI's explicit "could not tell". An empty cell means that marker was never
labelled for that speaker; `todo.csv` lists exactly those.

Current label counts: `s_coda` sibilant 75 / chiado 22 / mixed 19 ·
`r_coda` carioca 47 / tap 24 / caipira 21 / mixed 1 ·
`dt_palat` palatalized 66 / non-palatalized 23 / mixed 3 / unsure 1.

## Cohort

| Corpus | Speakers | | Corpus | Speakers |
|--------|---------:|-|--------|---------:|
| `common_voice` | 46 | | `brspeech_df` | 7 |
| `alcaim` | 22 | | `certas_palavras` | 6 |
| `coraa_ted` | 15 | | `cml_tts` | 6 |
| `colingpb` | 11 | | `tagarela` | 1 |
| | | | `thls` | 1 |

Speaker IDs are the directory names produced by this repo's `tools/download_*.py`
scripts, so they join directly against a local copy of each corpus — e.g.
`alcaim/Alcione_F018/`, `common_voice/{client_id}/`, `coraa_ted/{youtube_id}/`,
`certas_palavras/alvaro_malheiros/`. `thls` is the single-speaker
[1000-sentences-THLS](https://gitlab.com/lfelipesv/1000-sentences-thls-dataset)
corpus, which has no per-speaker directory.

## Caveats worth reading before you use this

**22 of the `alcaim` speakers carry only `s_coda`.** They come from the project's
two curated /s/-coda groups — the `subset_audio` (chiado) and
`subset_audio_sibilant` (sibilant) folders, 50 phonetically balanced sentences per
speaker. Those folders encode the sibilant distinction and nothing else, so
`r_coda` and `dt_palat` are deliberately blank rather than guessed. The `notes`
column records which folder each came from. The same applies to
`certas_palavras/alvaro_malheiros` and the `thls` speaker, which are in the
sibilant group. This is a real selection effect: the /s/-coda labels are not a
random sample of the corpora, they include a group chosen *because* of how its
speakers realize /s/.

**Agreement is measurable on three speakers only.** Nearly every speaker was
labelled once. Three `common_voice` speakers were labelled independently by two
annotators; of those, `r_coda` agreed 3/3, `s_coda` 1/3 and `dt_palat` 1/3. That
is far too small to quote as an inter-annotator agreement figure — it is here so
the disagreements are visible, not so they can be summarized.

**Ties are left empty.** In `speakers.csv`, when annotators split evenly on a
marker the value is blank and `agree_<marker>` is 0. With so few
multiply-annotated speakers, a tie-break rule would invent certainty that the
data does not have; resolve those by ear.

**Labels are per speaker, not per utterance.** Each one summarizes how that
speaker realizes the marker across the audio the annotator heard, which is a
handful of files, not the speaker's full contribution to the corpus.

**Four speaker IDs occur in two corpora.** `10107`, `12249`, `2961` and `4367`
appear under both `brspeech_df` and `cml_tts` — BRSpeech-DF's bonafide side is
drawn from CML-TTS, so these are the same speakers, and their labels match. That
is why 115 annotated (dataset, speaker) pairs collapse to 111 distinct speaker IDs
in `load_annotations()`, which is keyed by speaker alone. Use
`load_annotation_rows()` when you need the corpus kept apart.

**`source` is always `human` right now.** The column distinguishes a person's
judgement from a script's; the annotation UI can write `source=auto` rows, and
there are none at the moment.

**Two rows are empty.** `certas_palavras/alba_zaluar` and one `common_voice`
speaker have a row but no labels at all — they were opened in the UI and not
finished. They appear in `todo.csv` with all three markers missing.

**Annotators are named.** The `annotator` column holds the names of the authors
who did the labelling.
