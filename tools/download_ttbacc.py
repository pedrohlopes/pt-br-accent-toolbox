"""
Download and prepare the TTBAcc Brazilian Portuguese accent corpus.

    https://gitlab.com/thalesaguiar21/accent-dataset
    Paper: "Automatic Collection of Transcribed Speech for Low Resources Languages" (ICPRS 2023)
    Thesis: "A Methodology to Create Large Speech Datasets for Low-Resource Languages,
             and Improving Social Equity by Accent Conversion for PT-BR" (UFRN 2023)
    Authors: Thales Aguiar de Lima and Márjory Da Costa-Abreu

TTBAcc is a curated TEDx speech corpus across 21 Brazilian states (520 talks,
515 speakers, 16 kHz mono WAV) with demographic and geographic metadata:
  - speaker, state, city, birthplace, gender, age, ed_level, length, caption
  - manual (.hpt.txt) and automatic (.apt.txt) Portuguese transcriptions
  - MFA forced alignments (.TextGrid) for 64 talks in ptbr_aligned/

The dataset archive (ttbacc-br.zip, ~11.5 GB) is hosted on MEGA:
    https://mega.nz/file/JkF2HQzZ#pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c

Output layout under {ACCENTS_BASE}/ttbacc/:
    metadata.csv                           one row per talk (with speaker_id, region, state_name, ...)
    clips_metadata.csv                     one row per segmented clip (matches mupe/metadata.csv style)
    ttbacc-br-valid.tsv                    original TSV (+ ttbacc.tsv, ttbacc-descr.txt, ptbr_dictionary.txt)
    by_speaker/{speaker_id}/{fname}.wav    full talk (16 kHz mono WAV)
    by_speaker/{speaker_id}/{fname}.txt    preferred transcript (human if available, else auto)
    by_speaker/{speaker_id}/{fname}.TextGrid  MFA alignment (when present)
    clips/{speaker_id}/{fname}_{seg:04d}.wav  pause-segmented 5-15 s speech clips (for annotation UI / features)
    by_state/{UF}/{fname}.wav              hardlink tree (when --by-state is passed)

Usage:
    # Direct download from MEGA (or auto-detect ttbacc-br.zip in {ACCENTS_BASE}) and organize:
    python tools/download_ttbacc.py

    # From an already downloaded zip file + state tree:
    python tools/download_ttbacc.py --source /path/to/ttbacc-br.zip --by-state

    # Extract full talks only (skip speech clip segmentation):
    python tools/download_ttbacc.py --source /path/to/ttbacc-br.zip --no-clips

    # Segment all clips in every talk (no per-talk cap):
    python tools/download_ttbacc.py --source /path/to/ttbacc-br.zip --max-clips-per-talk 0

    # Using MEGA account credentials if anonymous IP bandwidth quota is hit:
    python tools/download_ttbacc.py --mega-email user@example.com --mega-password secret
"""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import unicodedata
import wave
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Progress, human, info, resolve_out, warn  # noqa: E402

DEFAULT_MEGA_URL = "https://mega.nz/file/JkF2HQzZ#pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c"
DEFAULT_ARCHIVE_NAME = "ttbacc-br.zip"

# ── Brazilian geography maps (IBGE standard, consistent with classifier_ui) ──

UF_TO_STATE: dict[str, str] = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondônia",
    "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}

UF_TO_REGION: dict[str, str] = {
    uf: reg
    for reg, ufs in {
        "N": ("AC", "AP", "AM", "PA", "RO", "RR", "TO"),
        "NE": ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"),
        "CO": ("DF", "GO", "MT", "MS"),
        "SE": ("ES", "MG", "RJ", "SP"),
        "S": ("PR", "RS", "SC"),
    }.items()
    for uf in ufs
}

AUX_FILES = (
    "ttbacc-br-valid.tsv",
    "ttbacc-br-valid-sorted.tsv",
    "ttbacc.tsv",
    "ttbacc-descr.txt",
    "ptbr_dictionary.txt",
)


def slugify_speaker(name: str) -> str:
    """Filesystem/URL-safe ASCII speaker folder id (preserves simple ASCII ids like spk_01)."""
    raw = (name or "").strip()
    if not raw:
        return "unknown"
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", folded).strip("_")
    return slug or "unknown"


# ── Self-contained MEGA crypto & download helpers (no mega.py required) ──────

def _b64url_decode(data: str) -> bytes:
    data += "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data.encode("ascii"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _str_to_a32(b: bytes) -> tuple[int, ...]:
    pad = (-len(b)) % 4
    if pad:
        b += b"\x00" * pad
    return struct.unpack(f">{len(b) // 4}I", b)


def _a32_to_str(a: tuple[int, ...]) -> bytes:
    return struct.pack(f">{len(a)}I", *a)


def _b64_to_a32(s: str) -> tuple[int, ...]:
    return _str_to_a32(_b64url_decode(s))


def _aes_cbc_encrypt(data: bytes, key_a32: tuple[int, ...]) -> bytes:
    from Crypto.Cipher import AES
    return AES.new(_a32_to_str(key_a32), AES.MODE_CBC, b"\x00" * 16).encrypt(data)


def _aes_cbc_decrypt(data: bytes, key_a32: tuple[int, ...]) -> bytes:
    from Crypto.Cipher import AES
    return AES.new(_a32_to_str(key_a32), AES.MODE_CBC, b"\x00" * 16).decrypt(data)


def _encrypt_key(a: tuple[int, ...], key: tuple[int, ...]) -> tuple[int, ...]:
    return sum((_str_to_a32(_aes_cbc_encrypt(_a32_to_str(a[i:i + 4]), key))
                for i in range(0, len(a), 4)), ())


def _decrypt_key(a: tuple[int, ...], key: tuple[int, ...]) -> tuple[int, ...]:
    return sum((_str_to_a32(_aes_cbc_decrypt(_a32_to_str(a[i:i + 4]), key))
                for i in range(0, len(a), 4)), ())


def _prepare_key(arr: tuple[int, ...]) -> tuple[int, ...]:
    pkey = (0x93C467E3, 0x7DB0C7A4, 0xD1BE3F81, 0x0152CB56)
    for _ in range(0x10000):
        for j in range(0, len(arr), 4):
            key = tuple(arr[j + i] if j + i < len(arr) else 0 for i in range(4))
            pkey = _encrypt_key(pkey, key)
    return pkey


def _stringhash(s: str, aeskey: tuple[int, ...]) -> str:
    s32 = _str_to_a32(s.encode("utf-8"))
    h32 = [0, 0, 0, 0]
    for i, v in enumerate(s32):
        h32[i & 3] ^= v
    h32_t = tuple(h32)
    for _ in range(0x4000):
        h32_t = _encrypt_key(h32_t, aeskey)
    return _b64url_encode(_a32_to_str((h32_t[0], h32_t[2])))


def _decrypt_attr(attr_bytes: bytes, key: tuple[int, ...]) -> dict | None:
    dec = _aes_cbc_decrypt(attr_bytes, key).decode("utf-8", errors="ignore").rstrip("\x00")
    if dec.startswith("MEGA"):
        return json.loads(dec[4:])
    return None


def _mpi_to_int(b: bytes) -> int:
    return int.from_bytes(b[2:], "big")


def parse_mega_url(url: str) -> tuple[str, str]:
    """Extract (file_id, file_key) from MEGA v1 or v2 URLs."""
    if "/file/" in url:
        clean = url.replace(" ", "")
        m = re.search(r"/file/([a-zA-Z0-9_-]+)#([a-zA-Z0-9_-]+)", clean)
        if m:
            return m.group(1), m.group(2)
    elif "!" in url:
        parts = url.split("!")
        if len(parts) >= 3:
            return parts[1], parts[2]
    raise ValueError(f"Could not parse file ID and key from MEGA URL: {url}")


def mega_login(email: str, password: str) -> str:
    """Authenticate with MEGA API and return session ID (sid)."""
    import requests
    from Crypto.PublicKey import RSA

    email = email.lower().strip()
    r0 = requests.post(
        "https://g.api.mega.co.nz/cs",
        json=[{"a": "us0", "user": email}],
        timeout=15,
    )
    resp0 = r0.json()[0]
    if isinstance(resp0, int):
        raise RuntimeError(f"MEGA user lookup error: {resp0}")

    if "s" in resp0:
        user_salt = _b64_to_a32(resp0["s"])
        pbkdf2_key = hashlib.pbkdf2_hmac(
            "sha512", password.encode(), _a32_to_str(user_salt), 100000, 32
        )
        password_aes = _str_to_a32(pbkdf2_key[:16])
        user_hash = _b64url_encode(pbkdf2_key[-16:])
    else:
        password_aes = _prepare_key(_str_to_a32(password.encode("utf-8")))
        user_hash = _stringhash(email, password_aes)

    r1 = requests.post(
        "https://g.api.mega.co.nz/cs",
        json=[{"a": "us", "user": email, "uh": user_hash}],
        timeout=15,
    )
    resp1 = r1.json()[0]
    if isinstance(resp1, int):
        if resp1 == -9:
            raise RuntimeError("Invalid MEGA email or password.")
        raise RuntimeError(f"MEGA login error code: {resp1}")

    master_key = _decrypt_key(_b64_to_a32(resp1["k"]), password_aes)

    if "tsid" in resp1:
        tsid = _b64url_decode(resp1["tsid"])
        key_encrypted = _a32_to_str(_encrypt_key(_str_to_a32(tsid[:16]), master_key))
        if key_encrypted == tsid[-16:]:
            return resp1["tsid"]

    if "csid" in resp1:
        rsa_private_key = _decrypt_key(_b64_to_a32(resp1["privk"]), master_key)
        private_key = _a32_to_str(rsa_private_key)
        rsa_components = [0, 0, 0, 0]
        for i in range(4):
            bitlength = (private_key[0] * 256) + private_key[1]
            bytelength = math.ceil(bitlength / 8) + 2
            rsa_components[i] = _mpi_to_int(private_key[:bytelength])
            private_key = private_key[bytelength:]
        p, q, d = rsa_components[:3]
        n = p * q
        phi = (p - 1) * (q - 1)
        e = pow(d, -1, phi)
        rsa = RSA.construct((n, e, d, p, q))
        encrypted_sid = _mpi_to_int(_b64url_decode(resp1["csid"]))
        sid_int = pow(encrypted_sid, rsa.d, rsa.n)
        sid_hex = f"{sid_int:x}"
        sid_bytes = binascii.unhexlify("0" + sid_hex if len(sid_hex) % 2 else sid_hex)
        return _b64url_encode(sid_bytes[:43])

    raise RuntimeError("Could not derive session ID from MEGA login response.")


def download_from_mega(
    url: str,
    dest_path: Path,
    email: str | None = None,
    password: str | None = None,
) -> Path:
    """Download a file from MEGA with streaming AES-CTR decryption (supports resume)."""
    try:
        import requests
        from Crypto.Cipher import AES
        from Crypto.Util import Counter as CryptoCounter
    except ImportError:
        raise RuntimeError(
            "MEGA download requires requests and pycryptodome: pip install requests pycryptodome"
        )

    file_id, file_key = parse_mega_url(url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    auth_sid = None
    if email and password:
        try:
            auth_sid = mega_login(email, password)
            info(f"Authenticated successfully as {email.strip()}")
        except Exception as e:
            warn(f"MEGA login failed ({e}), proceeding as anonymous.")

    info("Querying MEGA API for download URL ...")
    params = {"sid": auth_sid} if auth_sid else {}
    resp = requests.post(
        "https://g.api.mega.co.nz/cs",
        params=params,
        json=[{"a": "g", "g": 1, "p": file_id}],
        timeout=20,
    )
    data = resp.json()
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, int):
        if data == -2:
            raise RuntimeError("MEGA API error -2: Invalid file ID or key.")
        raise RuntimeError(f"MEGA API error code: {data}")
    if not isinstance(data, dict) or "g" not in data:
        raise RuntimeError(f"Unexpected MEGA API response: {data}")

    dl_url = data["g"]
    file_size = data.get("s", 0)

    key = _b64_to_a32(file_key)
    k = (key[0] ^ key[4], key[1] ^ key[5], key[2] ^ key[6], key[3] ^ key[7])
    iv = key[4:6] + (0, 0)
    k_bytes = _a32_to_str(k)

    if "at" in data:
        try:
            attrs = _decrypt_attr(_b64url_decode(data["at"]), k)
            if attrs and "n" in attrs:
                info(f"MEGA file: {attrs['n']} ({human(file_size)})")
        except Exception:
            pass

    if dest_path.exists() and file_size and dest_path.stat().st_size == file_size:
        info(f"{dest_path.name} already complete ({human(file_size)})")
        return dest_path

    # Resume on 16-byte AES block boundary if a partial file exists
    resume_bytes = 0
    if dest_path.exists():
        size_now = dest_path.stat().st_size
        if file_size and size_now < file_size:
            resume_bytes = (size_now // 16) * 16
            if resume_bytes != size_now:
                with open(dest_path, "r+b") as f:
                    f.truncate(resume_bytes)

    initial_ctr = (((iv[0] << 32) + iv[1]) << 64) + (resume_bytes // 16)
    ctr = CryptoCounter.new(128, initial_value=initial_ctr)
    cipher = AES.new(k_bytes, AES.MODE_CTR, counter=ctr)

    req_url = f"{dl_url}/{resume_bytes}-" if resume_bytes else dl_url
    if resume_bytes:
        info(f"Resuming {dest_path.name} at {human(resume_bytes)}/{human(file_size)} ...")
    else:
        info(f"Downloading {dest_path.name} ({human(file_size)}) ...")

    prog = Progress(total=file_size, label=dest_path.name)
    prog.done = resume_bytes
    chunk_size = 1 << 20
    try:
        with requests.get(req_url, stream=True, timeout=30) as r:
            if r.status_code == 509:
                retry_after = r.headers.get("X-MEGA-Time-Left", "unknown")
                raise PermissionError(
                    f"MEGA Bandwidth Limit Exceeded (HTTP 509). Cooldown: {retry_after}s.\n"
                    f"Tip: Set --mega-email and --mega-password, or download via browser "
                    f"and pass --source /path/to/{DEFAULT_ARCHIVE_NAME}."
                )
            r.raise_for_status()
            mode = "ab" if resume_bytes else "wb"
            with open(dest_path, mode) as f:
                for raw_chunk in r.iter_content(chunk_size=chunk_size):
                    if raw_chunk:
                        f.write(cipher.decrypt(raw_chunk))
                        prog.add(len(raw_chunk))
        prog.close()
        info(f"Download complete: {dest_path} ({human(dest_path.stat().st_size)})")
        return dest_path
    except Exception:
        prog.close()
        if dest_path.exists() and dest_path.stat().st_size == 0:
            dest_path.unlink(missing_ok=True)
        raise


# ── Pause-aware speech clip segmentation ─────────────────────────────────────

def _read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file (PCM_16 or IEEE_FLOAT / format 3) as float32 mono in [-1, 1]."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.clip(audio, -1.0, 1.0), int(sr)
    except ImportError:
        pass

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            audio = data.astype(np.float32) / max_val
        else:
            audio = data.astype(np.float32)
        return np.clip(audio, -1.0, 1.0), int(sr)
    except ImportError:
        pass

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV (install soundfile for float WAVs): {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return np.clip(pcm, -1.0, 1.0), sr


def _write_pcm16_wav(path: Path, audio_f32: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = path.with_suffix(".tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    tmp.replace(path)


def segment_talk_wav(
    wav_path: Path,
    clip_dir: Path,
    stem: str,
    clip_min_s: float = 5.0,
    clip_max_s: float = 15.0,
    max_clips: int = 24,
    edge_skip_s: float = 15.0,
) -> list[dict]:
    """
    Slice a long TEDx talk WAV into clean, pause-bounded speech clips (5-15 s, PCM_16).

    - Handles both PCM_16 and IEEE_FLOAT (format 3) source WAVs, always writing
      browser-safe 16-bit PCM mono WAV clips.
    - Skips TEDx intro/outro music and applause (edge_skip_s at start/end when
      the talk is long enough).
    - Splits on low-energy pauses using 50 ms frame RMS relative to the talk's
      speech level, so cuts land cleanly between phrases rather than mid-word.
    - Discards silent/near-silent segments.
    - When max_clips > 0 and more candidate clips exist, samples evenly across
      the full timeline of the talk.
    """
    audio, sr = _read_wav_float32(wav_path)
    total_s = len(audio) / sr
    if total_s < clip_min_s:
        return []

    # Trim intro/outro margins only when there is plenty of audio
    trim = edge_skip_s if total_s > (2 * edge_skip_s + 2 * clip_max_s) else 0.0
    start_idx = int(trim * sr)
    end_idx = len(audio) - int(trim * sr)
    body = audio[start_idx:end_idx]

    frame_ms = 50
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(body) // frame_len
    if n_frames < 4:
        return []

    frames = body[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)

    # Adaptive speech/pause threshold from the talk's own energy distribution:
    # always at least 14 dB below speech level (p90), and lifted above noisy floors (p10).
    p90 = float(np.percentile(db, 90))
    p10 = float(np.percentile(db, 10))
    if p90 < -55.0:
        return []
    pause_db = min(p90 - 14.0, max(p10 + 6.0, p90 - 24.0))
    min_speech_db = p90 - 20.0
    is_pause = db < pause_db

    # Find pause runs >= 200 ms (4 frames) and collect their quietest frame as split points
    min_pause_frames = 4
    split_frames = [0]
    i = 0
    while i < n_frames:
        if is_pause[i]:
            j = i
            while j < n_frames and is_pause[j]:
                j += 1
            if (j - i) >= min_pause_frames:
                best = i + int(np.argmin(rms[i:j]))
                if best > split_frames[-1]:
                    split_frames.append(best)
            i = j
        else:
            i += 1
    if split_frames[-1] < n_frames:
        split_frames.append(n_frames)

    # Greedily group inter-pause spans into [clip_min_s, clip_max_s] windows
    min_f = int(round(clip_min_s * 1000 / frame_ms))
    max_f = int(round(clip_max_s * 1000 / frame_ms))
    target_f = int(round(0.65 * (min_f + max_f)))

    candidates: list[tuple[int, int]] = []
    cur_start = split_frames[0]
    for idx in range(1, len(split_frames)):
        f_end = split_frames[idx]
        span = f_end - cur_start
        if span < min_f:
            continue
        if span <= max_f:
            # Prefer slightly fuller clips (~8-12 s) when the next pause still fits in max_f
            if idx + 1 < len(split_frames) and (split_frames[idx + 1] - cur_start) <= target_f:
                continue
            candidates.append((cur_start, f_end))
            cur_start = f_end
        else:
            # Single long span without a clear pause: cut at quietest frame inside [min_f, max_f]
            pos = cur_start
            while f_end - pos > max_f:
                lo = pos + min_f
                hi = min(pos + max_f, f_end)
                cut = lo + int(np.argmin(rms[lo:hi]))
                candidates.append((pos, cut))
                pos = cut
            if f_end - pos >= min_f:
                candidates.append((pos, f_end))
            cur_start = f_end

    # Trim leading/trailing silence inside each candidate and keep only speech-bearing clips
    clean: list[tuple[int, int]] = []
    for f0, f1 in candidates:
        active = np.flatnonzero(~is_pause[f0:f1])
        if len(active) == 0:
            continue
        a0 = max(f0, f0 + int(active[0]) - 2)
        a1 = min(f1, f0 + int(active[-1]) + 3)
        dur_s = (a1 - a0) * frame_ms / 1000.0
        if dur_s < clip_min_s or dur_s > clip_max_s + 0.5:
            continue
        seg_db = db[a0:a1]
        speech_ratio = float(np.mean(seg_db > min_speech_db))
        if speech_ratio < 0.35:
            continue
        clean.append((a0, a1))

    if not clean:
        return []

    if max_clips and max_clips > 0 and len(clean) > max_clips:
        idxs = np.linspace(0, len(clean) - 1, max_clips, dtype=int)
        clean = [clean[k] for k in sorted(set(idxs.tolist()))]

    clip_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for seq, (a0, a1) in enumerate(clean):
        s0 = start_idx + a0 * frame_len
        s1 = min(len(audio), start_idx + a1 * frame_len)
        clip_path = clip_dir / f"{stem}_{seq:04d}.wav"
        if not clip_path.exists():
            _write_pcm16_wav(clip_path, audio[s0:s1], sr)
        results.append({
            "start_s": round(s0 / sr, 3),
            "end_s": round(s1 / sr, 3),
            "duration": round((s1 - s0) / sr, 3),
            "clip_path": clip_path,
        })
    return results


# ── Archive extraction & layout ──────────────────────────────────────────────

def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink if possible, copy otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_valid_tsv(extract_dir: Path) -> Path:
    exact = list(extract_dir.rglob("ttbacc-br-valid.tsv"))
    if exact:
        return sorted(exact, key=lambda p: len(p.parts))[0]
    fallback = list(extract_dir.rglob("*valid*.tsv")) or list(extract_dir.rglob("*.tsv"))
    if fallback:
        return sorted(fallback, key=lambda p: (len(p.parts), p.name))[0]
    raise RuntimeError(f"Could not find valid TSV file inside {extract_dir}")


def process_ttbacc(
    archive_path: Path,
    out_root: Path,
    by_state: bool = False,
    min_per_state: int = 0,
    max_speakers: int | None = None,
    delete_archive: bool = False,
    make_clips: bool = True,
    clip_min_s: float = 5.0,
    clip_max_s: float = 15.0,
    max_clips_per_talk: int = 24,
) -> None:
    """
    Unpack ttbacc-br.zip and organize into the standard accents data lake layout:
        ttbacc/
            metadata.csv
            clips_metadata.csv                        (if make_clips=True)
            ttbacc-br-valid.tsv (+ aux TSV/dict/descr files)
            by_speaker/{speaker_id}/{fname}.wav
            by_speaker/{speaker_id}/{fname}.txt
            by_speaker/{speaker_id}/{fname}.TextGrid  (when available)
            clips/{speaker_id}/{fname}_{seg:04d}.wav  (if make_clips=True)
            by_state/{UF}/{fname}.wav                 (if by_state=True)
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    info(f"Extracting {archive_path.name} ...")
    extract_dir = out_root / "_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    tsv_path = _find_valid_tsv(extract_dir)
    info(f"Reading annotations from {tsv_path.relative_to(extract_dir)} ...")

    records: list[dict] = []
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            records.append(row)
    info(f"Found {len(records)} entries in TSV.")

    # Single-pass file index over extracted tree (O(N) instead of O(N^2) rglob)
    wav_index: dict[str, Path] = {}
    tg_index: dict[str, Path] = {}
    for p in extract_dir.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(".wav"):
            wav_index.setdefault(p.stem, p)
        elif low.endswith(".textgrid"):
            tg_index.setdefault(p.stem, p)

    # Preserve top-level auxiliary corpus files (TSVs, description, pronunciation dictionary)
    for aux_name in AUX_FILES:
        matches = list(extract_dir.rglob(aux_name))
        if matches:
            shutil.copy2(matches[0], out_root / aux_name)

    by_speaker_dir = out_root / "by_speaker"
    by_speaker_dir.mkdir(parents=True, exist_ok=True)
    clips_root = out_root / "clips"

    state_counts = Counter((r.get("state") or "").strip().upper() for r in records)
    speakers_seen: set[str] = set()
    kept_records: list[dict] = []
    clip_records: list[dict] = []

    saved_wavs = 0
    saved_txts = 0
    saved_tgs = 0
    saved_clips = 0

    for idx, row in enumerate(records, 1):
        spk_raw = (row.get("speaker") or "unknown").strip()
        spk_id = slugify_speaker(spk_raw)
        fname = (row.get("fname") or "").strip()
        state = (row.get("state") or "").strip().upper()
        state_name = UF_TO_STATE.get(state, state)
        region = UF_TO_REGION.get(state, "")

        if not fname:
            continue

        if max_speakers and spk_id not in speakers_seen and len(speakers_seen) >= max_speakers:
            continue

        wav_file = wav_index.get(fname)
        if not wav_file or not wav_file.exists():
            continue

        speakers_seen.add(spk_id)

        dest_spk_dir = by_speaker_dir / spk_id
        dest_wav = dest_spk_dir / f"{fname}.wav"
        link_or_copy(wav_file, dest_wav)
        saved_wavs += 1

        # Transcriptions: prefer human (.hpt.txt), then automatic (.apt.txt), then plain (.txt)
        hpt_txt = wav_file.parent / f"{fname}.hpt.txt"
        apt_txt = wav_file.parent / f"{fname}.apt.txt"
        raw_txt = wav_file.parent / f"{fname}.txt"
        dest_txt = dest_spk_dir / f"{fname}.txt"

        transcript = ""
        trans_type = "none"
        if hpt_txt.exists():
            transcript = hpt_txt.read_text(encoding="utf-8", errors="replace").strip()
            dest_txt.write_text(transcript, encoding="utf-8")
            trans_type = "human"
            saved_txts += 1
        elif apt_txt.exists():
            transcript = apt_txt.read_text(encoding="utf-8", errors="replace").strip()
            dest_txt.write_text(transcript, encoding="utf-8")
            trans_type = "auto"
            saved_txts += 1
        elif raw_txt.exists():
            transcript = raw_txt.read_text(encoding="utf-8", errors="replace").strip()
            dest_txt.write_text(transcript, encoding="utf-8")
            trans_type = "plain"
            saved_txts += 1

        # Preserve MFA forced alignment TextGrid alongside the talk WAV
        tg_rel = ""
        tg_file = tg_index.get(fname)
        if tg_file and tg_file.exists():
            dest_tg = dest_spk_dir / f"{fname}.TextGrid"
            link_or_copy(tg_file, dest_tg)
            tg_rel = str(dest_tg.relative_to(out_root))
            saved_tgs += 1

        # Optional by_state/{UF}/*.wav tree for accent classifiers
        if by_state and state:
            if not min_per_state or state_counts[state] >= min_per_state:
                dest_state_wav = out_root / "by_state" / state / f"{fname}.wav"
                link_or_copy(dest_wav, dest_state_wav)

        # Pause-aware speech clips for annotation UI and fast feature extraction
        n_clips = 0
        if make_clips:
            try:
                segs = segment_talk_wav(
                    dest_wav,
                    clips_root / spk_id,
                    fname,
                    clip_min_s=clip_min_s,
                    clip_max_s=clip_max_s,
                    max_clips=max_clips_per_talk,
                )
            except Exception as e:
                warn(f"Clip segmentation skipped for {fname}: {e}")
                segs = []
            n_clips = len(segs)
            saved_clips += n_clips
            for seg in segs:
                clip_records.append({
                    "speaker_id": spk_id,
                    "speaker": spk_raw,
                    "fname": fname,
                    "state": state,
                    "state_name": state_name,
                    "region": region,
                    "city": (row.get("city") or "").strip(),
                    "birthplace": (row.get("birthplace") or "").strip(),
                    "gender": (row.get("gender") or "").strip(),
                    "age": (row.get("age") or "").strip(),
                    "ed_level": (row.get("ed_level") or "").strip(),
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "duration": seg["duration"],
                    "wav": str(seg["clip_path"].relative_to(out_root)),
                })

        row_meta = dict(row)
        row_meta["speaker_id"] = spk_id
        row_meta["state"] = state
        row_meta["state_name"] = state_name
        row_meta["region"] = region
        row_meta["wav_path"] = str(dest_wav.relative_to(out_root))
        row_meta["textgrid_path"] = tg_rel
        row_meta["transcript"] = transcript
        row_meta["transcript_type"] = trans_type
        row_meta["n_clips"] = n_clips
        kept_records.append(row_meta)

        if idx % 50 == 0 or idx == len(records):
            print(
                f"\r  processed {idx}/{len(records)} talks "
                f"({saved_clips} clips, {saved_tgs} TextGrids)   ",
                end="",
                flush=True,
            )
    print(flush=True)

    meta_csv = out_root / "metadata.csv"
    if kept_records:
        fieldnames = list(kept_records[0].keys())
        with open(meta_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_records)
        info(f"Saved metadata: {meta_csv} ({len(kept_records)} talks)")

    if make_clips and clip_records:
        clips_csv = out_root / "clips_metadata.csv"
        with open(clips_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(clip_records[0].keys()))
            writer.writeheader()
            writer.writerows(clip_records)
        info(f"Saved clip metadata: {clips_csv} ({len(clip_records)} clips)")

    shutil.rmtree(extract_dir, ignore_errors=True)

    if delete_archive:
        archive_path.unlink(missing_ok=True)
        info(f"Deleted archive: {archive_path.name}")

    info(
        f"Done: {saved_wavs} talks, {saved_txts} transcripts, {saved_tgs} TextGrids, "
        f"{saved_clips} clips across {len(speakers_seen)} speakers."
    )
    if by_state:
        state_dir = out_root / "by_state"
        if state_dir.exists():
            states_created = [d.name for d in state_dir.iterdir() if d.is_dir()]
            info(f"Created state tree for {len(states_created)} states in {state_dir}")


def segment_existing_ttbacc(
    out_root: Path,
    clip_min_s: float = 5.0,
    clip_max_s: float = 15.0,
    max_clips_per_talk: int = 24,
) -> None:
    """Segment already-extracted talks under {out_root}/by_speaker/ into {out_root}/clips/."""
    meta_csv = out_root / "metadata.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found in {out_root}; run full extraction first.")

    with open(meta_csv, "r", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))

    clips_root = out_root / "clips"
    clip_records: list[dict] = []
    saved_clips = 0

    for idx, row in enumerate(records, 1):
        spk_id = (row.get("speaker_id") or slugify_speaker(row.get("speaker") or "")).strip()
        fname = (row.get("fname") or "").strip()
        wav_path = out_root / row["wav_path"]
        if not wav_path.exists():
            continue
        try:
            segs = segment_talk_wav(
                wav_path,
                clips_root / spk_id,
                fname,
                clip_min_s=clip_min_s,
                clip_max_s=clip_max_s,
                max_clips=max_clips_per_talk,
            )
        except Exception as e:
            warn(f"Clip segmentation skipped for {fname}: {e}")
            segs = []
        row["n_clips"] = len(segs)
        saved_clips += len(segs)
        for seg in segs:
            clip_records.append({
                "speaker_id": spk_id,
                "speaker": (row.get("speaker") or "").strip(),
                "fname": fname,
                "state": (row.get("state") or "").strip().upper(),
                "state_name": (row.get("state_name") or "").strip(),
                "region": (row.get("region") or "").strip(),
                "city": (row.get("city") or "").strip(),
                "birthplace": (row.get("birthplace") or "").strip(),
                "gender": (row.get("gender") or "").strip(),
                "age": (row.get("age") or "").strip(),
                "ed_level": (row.get("ed_level") or "").strip(),
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "duration": seg["duration"],
                "wav": str(seg["clip_path"].relative_to(out_root)),
            })
        if idx % 50 == 0 or idx == len(records):
            print(f"\r  segmented {idx}/{len(records)} talks ({saved_clips} clips)   ", end="", flush=True)
    print(flush=True)

    if records:
        with open(meta_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        info(f"Updated metadata: {meta_csv} ({len(records)} talks)")

    if clip_records:
        clips_csv = out_root / "clips_metadata.csv"
        with open(clips_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(clip_records[0].keys()))
            writer.writeheader()
            writer.writerows(clip_records)
        info(f"Saved clip metadata: {clips_csv} ({len(clip_records)} clips)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _locate_existing_archive(out_root: Path) -> Path | None:
    """Check {out_root}/ttbacc-br.zip and {out_root.parent}/ttbacc-br.zip."""
    for cand in (out_root / DEFAULT_ARCHIVE_NAME, out_root.parent / DEFAULT_ARCHIVE_NAME):
        if cand.exists() and cand.stat().st_size > 10_000_000_000:
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and prepare the TTBAcc speech corpus")
    ap.add_argument("--out", default=None,
                    help="output directory (default: {ACCENTS_BASE}/ttbacc)")
    ap.add_argument("--source", default=None,
                    help="path to local ttbacc-br.zip if already downloaded")
    ap.add_argument("--url", default=DEFAULT_MEGA_URL,
                    help="MEGA URL to download from")
    ap.add_argument("--mega-email", default=os.environ.get("MEGA_EMAIL"),
                    help="MEGA account email for quota extension")
    ap.add_argument("--mega-password", default=os.environ.get("MEGA_PASSWORD"),
                    help="MEGA account password")
    ap.add_argument("--by-state", action="store_true",
                    help="create by_state/{UF}/*.wav tree for accent classification")
    ap.add_argument("--min-per-state", type=int, default=0,
                    help="minimum talks required per state for by_state tree")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="limit number of speakers to process")
    ap.add_argument("--no-clips", action="store_true",
                    help="skip pause-aware 5-15 s speech clip segmentation")
    ap.add_argument("--clips-only", action="store_true",
                    help="only (re-)segment already extracted talks in {out}/by_speaker/")
    ap.add_argument("--clip-min", type=float, default=5.0,
                    help="minimum clip duration in seconds (default: 5.0)")
    ap.add_argument("--clip-max", type=float, default=15.0,
                    help="maximum clip duration in seconds (default: 15.0)")
    ap.add_argument("--max-clips-per-talk", type=int, default=24,
                    help="cap clips per talk evenly across timeline; 0 = unlimited (default: 24)")
    ap.add_argument("--delete-archive", action="store_true",
                    help="delete zip archive after extraction")
    args = ap.parse_args()

    out_root = resolve_out(args.out, "ttbacc")
    print("=== TTBAcc PT-BR Accent Corpus ===")
    print(f"  Target directory: {out_root}")

    if args.clips_only:
        segment_existing_ttbacc(
            out_root=out_root,
            clip_min_s=args.clip_min,
            clip_max_s=args.clip_max,
            max_clips_per_talk=args.max_clips_per_talk,
        )
        return

    if args.source:
        archive_path = Path(args.source).resolve()
        if not archive_path.exists():
            sys.exit(f"Source archive not found: {archive_path}")
        info(f"Using provided source archive: {archive_path}")
    else:
        found = _locate_existing_archive(out_root)
        if found:
            archive_path = found
            info(f"Found existing complete archive: {archive_path}")
        else:
            archive_path = out_root / DEFAULT_ARCHIVE_NAME
            try:
                download_from_mega(
                    url=args.url,
                    dest_path=archive_path,
                    email=args.mega_email,
                    password=args.mega_password,
                )
            except PermissionError as e:
                warn(str(e))
                sys.exit(1)
            except Exception as e:
                warn(f"Failed to download from MEGA: {e}")
                sys.exit(1)

    process_ttbacc(
        archive_path=archive_path,
        out_root=out_root,
        by_state=args.by_state,
        min_per_state=args.min_per_state,
        max_speakers=args.max_speakers,
        delete_archive=args.delete_archive,
        make_clips=not args.no_clips,
        clip_min_s=args.clip_min,
        clip_max_s=args.clip_max,
        max_clips_per_talk=args.max_clips_per_talk,
    )


if __name__ == "__main__":
    main()
