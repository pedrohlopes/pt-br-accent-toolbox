"""Brazilian-Portuguese G2P + stress + pretonic-context labelling.

Used by `features.vowel_harmony` (ported from the accents project's Exp 56).
Needs the `espeak-ng` binary on PATH (``apt install espeak-ng``).

The job: given a word (orthography), decide
  1. how many syllable NUCLEI it has and where they are in the LETTER string,
  2. which nucleus carries lexical stress,
  3. the QUALITY of the stressed vowel (open ɛ/ɔ, low a, high i/u, close-mid e/o),
  4. which nuclei are pretonic single-letter <e>/<o> targets, and under which
     trigger context.

Two independent sources are cross-checked and a word is only accepted when they
agree:
  * ORTHOGRAPHY  — nucleus segmentation, nasality, diphthongs, and the classical
    Portuguese stress rule.  Orthography is authoritative for nucleus COUNT,
    nasality and diphthong status because espeak mangles all three (it renders
    "em" as `eɪ`, inserts epenthetic `ə` in "perto" -> `pˈɛɾətʊ`, etc.).
  * espeak-ng pt-br IPA — authoritative for the lexical QUALITY of the stressed
    vowel (ɛ vs e, ɔ vs o), which orthography cannot give: <colega> and
    <pequeno> look identical but are kɔlˈɛɡa vs pekˈenu.

CRITICAL: the pretonic vowel's own realization is NEVER read from espeak (it
always reports the canonical closed form) nor from ZIPA (measured unreliable:
per-speaker mean F1 of ZIPA-labelled [ɛ] minus [e] is only +13..20 Hz in the
exp43 cache).  It is measured acoustically downstream.  This module only says
WHERE to measure and WHAT the trigger is.
"""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass, asdict, field

# ── vowel/letter inventories ────────────────────────────────────────────────
VOWEL_LETTERS = set('aeiouáéêíóôúâãõàü')
STRONG_V      = set('aeoáéêóôâã')      # can be a syllable nucleus on their own
WEAK_V        = set('iuíú')            # glide candidates in a diphthong
CONSONANTS    = set('bcdfghjklmnpqrstvwxyz')

# orthographic falling diphthongs: strong + weak (one nucleus)
FALLING = {'ai', 'ei', 'oi', 'ui', 'au', 'eu', 'ou', 'iu', 'ao',
           'ái', 'éi', 'ói', 'áu', 'éu', 'ãe', 'ão', 'õe', 'ũi'}
# sites where orthography is genuinely ambiguous and espeak arbitrates:
#   falling  "ai" -> diphthong (b[ai]le) or hiatus (a-[i]nda)
#   rising   "ia" -> glide (histó-r[ja]) or hiatus (d[i]-[a])
AMBIG_FALLING = {'ai', 'ei', 'oi', 'ui', 'au', 'eu', 'ou', 'iu', 'ao'}

# espeak IPA vowel symbols -> quality class
IPA_QUALITY = {
    'a': 'a', 'ɐ': 'a', 'ɑ': 'a', 'æ': 'a',
    'ɛ': 'ɛ', 'e': 'e', 'ɘ': 'e', 'ə': 'e',
    'ɔ': 'ɔ', 'o': 'o',
    'i': 'i', 'ɪ': 'i', 'y': 'i',
    'u': 'u', 'ʊ': 'u', 'ʉ': 'u', 'ɯ': 'u',
}
IPA_VOWELS = set(IPA_QUALITY)
# glides espeak attaches to a nucleus; never nuclei on their own
IPA_GLIDES = set('jwɪʊ')

TRIGGER_OF_QUALITY = {'ɛ': 'T_low', 'ɔ': 'T_low', 'a': 'T_low',
                      'i': 'T_high', 'u': 'T_high',
                      'e': 'T_mid', 'o': 'T_mid'}

# unstressed function words / clitics: never carry a lexical pretonic target
CLITICS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos',
    'das', 'em', 'no', 'na', 'nos', 'nas', 'num', 'numa', 'por', 'pelo', 'pela',
    'pelos', 'pelas', 'para', 'pra', 'pro', 'com', 'sem', 'sob', 'sobre', 'que',
    'se', 'me', 'te', 'lhe', 'lhes', 'nos', 'vos', 'meu', 'teu', 'seu', 'e',
    'ou', 'mas', 'ao', 'aos', 'à', 'às', 'ele', 'ela', 'eles', 'elas', 'eu',
    'tu', 'ele', 'nós', 'vós', 'este', 'esse', 'isso', 'isto', 'ser', 'ter',
}

# heteronyms and known espeak failures: open/closed quality is unreliable
HETERONYMS = {
    'sede', 'colher', 'governo', 'gosto', 'jogo', 'olho', 'almoço', 'acordo',
    'corte', 'apoio', 'começo', 'choro', 'forma', 'sobre', 'para', 'este',
    'seca', 'molho', 'torre', 'posto', 'imposto', 'esforço', 'socorro',
}


@dataclass
class Nucleus:
    """One syllable nucleus, located in the word's LETTER string."""
    idx: int              # nucleus ordinal within the word
    letter_idx: int       # index of the nucleus letter in the word string
    letters: str          # the orthographic nucleus ("e", "ei", "ão")
    letter: str           # the measured vowel letter (first strong letter)
    nasal: bool
    diph: bool
    stressed: bool = False
    ipa: str = ''         # espeak quality of this nucleus, when matched
    role: str = 'other'   # pretonic | pre2 | anchor | other
    trigger: str = ''     # T_low | T_high | T_mid | T_nasal | T_diph | T_other
    stressed_ipa: str = ''
    dist: int = 0         # distance in nuclei to the stressed one (pretonic=1)
    flags: list = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Lowercase, NFC, drop punctuation/digits, split hyphenated clitics."""
    t = unicodedata.normalize('NFC', text.lower())
    t = t.replace('-', ' ')
    t = re.sub(r"[^a-záéêíóôúâãõàüç\s']", ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def ortho_nuclei(word: str, split_falling=(), keep_rising=()) -> list[Nucleus]:
    """Segment a word into syllable nuclei from ORTHOGRAPHY alone.

    Handles: falling diphthongs (one nucleus), silent <u> in qu/gu + e/i,
    nasal nuclei (V + m/n + consonant, word-final <m>/<n>, or tilde).
    Rising diphthongs (weak + strong, "história") are split into two nuclei
    only when the weak vowel is accented; otherwise the weak vowel is a glide.
    """
    nuc: list[Nucleus] = []
    i, n = 0, len(word)
    while i < n:
        c = word[i]
        if c not in VOWEL_LETTERS:
            i += 1
            continue

        # silent <u> of qu/gu before e/i
        if c == 'u' and i > 0 and word[i - 1] in 'qg' and i + 1 < n and word[i + 1] in 'eiéêí':
            i += 1
            continue

        start = i
        letters = c
        diph = False

        # falling diphthong: strong (or weak) + weak
        if (i + 1 < n and (c + word[i + 1]) in FALLING
                and start not in split_falling):
            letters = c + word[i + 1]
            diph = True
            i += 2
        # rising glide: unaccented weak + strong ("quase", "história" -> ia)
        elif (c in WEAK_V and c in 'iu' and i + 1 < n and word[i + 1] in STRONG_V
                and start not in keep_rising):
            # the weak vowel is a glide; the nucleus is the following strong one
            i += 1
            continue
        else:
            i += 1

        # nasality: nucleus followed by m/n that is not itself a syllable onset,
        # or a tilde on the nucleus
        nasal = any(unicodedata.normalize('NFD', ch)[1:2] == '̃' for ch in letters)
        if not nasal and i < n and word[i] in 'mn':
            # nasal if the m/n closes the syllable (word-final or followed by C)
            if i + 1 >= n or word[i + 1] not in VOWEL_LETTERS:
                nasal = True

        nuc.append(Nucleus(idx=len(nuc), letter_idx=start, letters=letters,
                           letter=letters[0], nasal=nasal, diph=diph))
    return nuc


def ortho_stress(word: str, nuc: list[Nucleus]) -> int | None:
    """Index of the stressed nucleus by the classical Portuguese rule."""
    if not nuc:
        return None
    if len(nuc) == 1:
        return 0
    # 1. a written accent wins
    for k, u in enumerate(nuc):
        for ch in u.letters:
            d = unicodedata.normalize('NFD', ch)
            if len(d) > 1 and d[1] in '́̂̀':   # ´ ^ `
                return k
    # 2. paroxytone endings — but a nasal-diphthong ending (ão/ãe/õe/ãos)
    #    is oxytone: "estão", "fogão", "vulcão"
    if re.search(r'[ãõ][eo]s?$', word):
        return len(nuc) - 1
    if re.search(r'(a|e|o|as|es|os|am|em|ens)$', word):
        return len(nuc) - 2
    # 3. otherwise oxytone
    return len(nuc) - 1


def _espeak_batch(words: list[str]) -> dict[str, str]:
    """Word -> IPA, one espeak-ng call for the whole batch."""
    if not words:
        return {}
    proc = subprocess.run(
        ['espeak-ng', '-v', 'pt-br', '-q', '--ipa'],
        input='\n'.join(words), capture_output=True, text=True)
    lines = [ln.strip() for ln in proc.stdout.split('\n')]
    lines = [ln for ln in lines if ln]
    if len(lines) != len(words):
        # fall back to one call per word when the batch desyncs
        out = {}
        for w in words:
            p = subprocess.run(['espeak-ng', '-v', 'pt-br', '-q', '--ipa'],
                               input=w, capture_output=True, text=True)
            out[w] = p.stdout.strip().replace('\n', ' ')
        return out
    return dict(zip(words, lines))


def ipa_nuclei(ipa: str) -> tuple[list[str], int | None]:
    """Parse espeak IPA into (nucleus qualities, stressed nucleus index).

    Glides (j w ɪ ʊ) directly adjacent to a nucleus are absorbed, epenthetic
    `ə` between consonants is dropped, and stress is `ˈ` when present, else a
    lone `ˌ` (espeak marks some initial-stress disyllables with `ˌ` only).
    """
    quals: list[str] = []
    stressed_idx = None
    pending_stress = None          # 'primary' | 'secondary'
    n_secondary = ipa.count('ˌ')
    has_primary = 'ˈ' in ipa

    i = 0
    prev_was_vowel = False
    while i < len(ipa):
        ch = ipa[i]
        if ch == 'ˈ':
            pending_stress = 'primary'
            prev_was_vowel = False
            i += 1
            continue
        if ch == 'ˌ':
            pending_stress = 'secondary'
            prev_was_vowel = False
            i += 1
            continue
        if ch in ('ː', '̃', '̃', ' '):
            i += 1
            continue
        if ch in IPA_VOWELS:
            # a glide right after a vowel is part of that nucleus
            if prev_was_vowel and ch in IPA_GLIDES:
                i += 1
                continue
            # epenthetic schwa: espeak inserts it between consonants
            if ch == 'ə' and not prev_was_vowel:
                nxt = ipa[i + 1] if i + 1 < len(ipa) else ''
                if nxt and nxt not in IPA_VOWELS and pending_stress is None:
                    i += 1
                    continue
            quals.append(IPA_QUALITY[ch])
            if pending_stress == 'primary':
                stressed_idx = len(quals) - 1
            elif pending_stress == 'secondary' and not has_primary and n_secondary == 1:
                stressed_idx = len(quals) - 1
            pending_stress = None
            prev_was_vowel = True
            i += 1
            continue
        prev_was_vowel = False
        i += 1
    return quals, stressed_idx


def _ambiguous_sites(word: str) -> tuple[list[int], list[int]]:
    """Letter positions where the nucleus segmentation is genuinely ambiguous.

    Returns (falling_sites, rising_sites): positions that could be read either
    as one diphthong nucleus or as two hiatus nuclei.  Orthography alone cannot
    decide ("b[ai]le" vs "a-[i]nda", "histó-r[ja]" vs "d[i]-[a]"), so
    analyze_word tries the combinations and lets espeak's nucleus count and
    stress position pick the winner.
    """
    falling, rising = [], []
    for i in range(len(word) - 1):
        pair = word[i:i + 2]
        if pair in AMBIG_FALLING:
            falling.append(i)
        elif word[i] in 'iu' and word[i + 1] in STRONG_V:
            if i == 0 or word[i - 1] not in 'qg':
                rising.append(i)
    return falling[:3], rising[:3]


def _segmentations(word: str):
    """Yield candidate nucleus segmentations, canonical reading first."""
    falling, rising = _ambiguous_sites(word)
    seen = set()
    # subsets of "split this falling diphthong" x "keep this rising hiatus"
    for fmask in range(1 << len(falling)):
        for rmask in range(1 << len(rising)):
            sf = tuple(s for k, s in enumerate(falling) if fmask >> k & 1)
            kr = tuple(s for k, s in enumerate(rising) if rmask >> k & 1)
            key = (sf, kr)
            if key in seen:
                continue
            seen.add(key)
            yield ortho_nuclei(word, split_falling=sf, keep_rising=kr)


def analyze_word(word: str, ipa: str) -> tuple[list[Nucleus] | None, str]:
    """Cross-check orthography against espeak; label nuclei. -> (nuclei, reason)."""
    if len(word) < 2:
        # espeak spells single letters out ("é" -> "ˌɛaɡˈudʊ" = "é agudo")
        return None, 'single_letter'
    quals, i_str = ipa_nuclei(ipa)
    if i_str is None:
        return None, 'no_ipa_stress'

    # pick the segmentation that agrees with espeak on BOTH nucleus count and
    # stress position; orthography still owns nasality and diphthong flags
    nuc = None
    best_reason = 'no_nuclei'
    for cand in _segmentations(word):
        if not cand:
            continue
        if len(cand) != len(quals):
            best_reason = f'nuc_mismatch:{len(cand)}vs{len(quals)}'
            continue
        o_str = ortho_stress(word, cand)
        if o_str != i_str:
            best_reason = f'stress_mismatch:{o_str}vs{i_str}'
            continue
        nuc = cand
        break
    if nuc is None:
        return None, best_reason
    o_str = i_str

    for k, u in enumerate(nuc):
        u.ipa = quals[k]
        u.stressed = (k == i_str)

    st = nuc[i_str]
    # trigger class of the stressed nucleus; orthography overrides espeak for
    # nasality and diphthongs, which espeak reports unreliably
    if st.nasal:
        trig = 'T_nasal'
    elif st.diph:
        trig = 'T_diph'
    else:
        trig = TRIGGER_OF_QUALITY.get(st.ipa, 'T_other')

    is_clitic = word in CLITICS
    is_hetero = word in HETERONYMS

    for k, u in enumerate(nuc):
        if u.stressed:
            # anchor: a stressed, plain, oral monophthong of known quality
            if (not u.nasal and not u.diph and u.ipa in 'aeɛoɔiu'
                    and len(u.letters) == 1 and not is_hetero):
                u.role = 'anchor'
                u.trigger = u.ipa
            continue
        d = i_str - k
        if d <= 0:                       # post-tonic
            u.role = 'other'
            continue
        # pretonic target: single-letter <e>/<o>, oral, no written accent
        if (u.letter in 'eo' and len(u.letters) == 1 and not u.nasal
                and not u.diph and not is_clitic and not is_hetero):
            u.role = 'pretonic' if d == 1 else ('pre2' if d == 2 else 'other')
            u.trigger = trig
            u.stressed_ipa = st.ipa
            u.dist = d
            if k == 0 and word[:2] in ('es', 'ex'):
                u.flags.append('esx')
            if u.letter_idx > 0 and word[u.letter_idx - 1] in 'mnh':
                u.flags.append('nasal_adj')
            nxt = u.letter_idx + len(u.letters)
            if nxt < len(word) and word[nxt] in 'mn':
                u.flags.append('nasal_adj')
        else:
            u.role = 'other'
    return nuc, 'ok'


class Lexicon:
    """Word -> analysis, cached to disk (espeak is the slow part)."""

    def __init__(self, path=None):
        self.path = path
        self.cache: dict[str, dict] = {}
        if path:
            try:
                with open(path) as fh:
                    self.cache = json.load(fh)
            except (FileNotFoundError, json.JSONDecodeError):
                self.cache = {}

    def analyze(self, words: list[str]) -> dict[str, dict]:
        missing = sorted({w for w in words if w and w not in self.cache})
        if missing:
            ipas = _espeak_batch(missing)
            for w in missing:
                nuc, reason = analyze_word(w, ipas.get(w, ''))
                self.cache[w] = {
                    'ipa': ipas.get(w, ''),
                    'ok': nuc is not None,
                    'reason': reason,
                    'nuclei': [asdict(u) for u in nuc] if nuc else [],
                }
        return {w: self.cache[w] for w in words if w in self.cache}

    def save(self):
        if self.path:
            with open(self.path, 'w') as fh:
                json.dump(self.cache, fh, ensure_ascii=False)
