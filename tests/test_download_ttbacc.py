"""
Tests for tools/download_ttbacc.py.
"""
from __future__ import annotations

import csv
import io
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from download_ttbacc import parse_mega_url, process_ttbacc, slugify_speaker  # noqa: E402


def _make_speech_like_wav(duration_s: float = 24.0, sr: int = 16000, float32: bool = False) -> bytes:
    """Synthesize alternating 6.5 s speech bursts and 0.6 s silences (16 kHz PCM16 or IEEE float32)."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sig = np.zeros(n, dtype=np.float32)
    # 3 bursts separated by clear pauses
    for start_s, end_s in ((0.5, 7.0), (7.6, 14.2), (14.8, 22.0)):
        i0, i1 = int(start_s * sr), min(n, int(end_s * sr))
        if i0 < i1:
            sig[i0:i1] = 0.25 * np.sin(2 * np.pi * 220.0 * t[i0:i1])
    buf = io.BytesIO()
    if float32:
        import soundfile as sf
        sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    else:
        pcm = (sig * 32767.0).astype("<i2")
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_parse_mega_url():
    url_v2 = "https://mega.nz/file/JkF2HQzZ#pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c"
    file_id, key = parse_mega_url(url_v2)
    assert file_id == "JkF2HQzZ"
    assert key == "pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c"

    url_v1 = "https://mega.co.nz/#!JkF2HQzZ!pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c"
    file_id_v1, key_v1 = parse_mega_url(url_v1)
    assert file_id_v1 == "JkF2HQzZ"
    assert key_v1 == "pqwru_4MKtnljqqsS9LbMsCSk5bR6ETLPm6IA0eTY2c"

    with pytest.raises(ValueError):
        parse_mega_url("https://invalid-url.com/something")


def test_slugify_speaker():
    assert slugify_speaker("spk_01") == "spk_01"
    assert slugify_speaker("mila d'oliveira") == "mila_d_oliveira"
    assert slugify_speaker("néli pereira") == "neli_pereira"
    assert slugify_speaker("   ") == "unknown"


def test_process_ttbacc_pipeline(tmp_path: Path):
    mock_zip_path = tmp_path / "mock_ttbacc.zip"
    out_dir = tmp_path / "processed"

    tsv_data = (
        "url\ttitle\tspeaker\tage\tgender\tfname\ted_level\tstate\tcity\tbirthplace\tvalid\tcaption\tlength\n"
        "https://yt/1\ttalk 1\tmila d'oliveira\t30\tF\tted-talk-0001\tSUP\tBA\tSalvador\tTrue\tTrue\tpt-BR\t24\n"
        "https://yt/2\ttalk 2\tmila d'oliveira\t30\tF\tted-talk-0002\tSUP\tBA\tSalvador\tTrue\tTrue\ta.pt\t24\n"
        "https://yt/3\ttalk 3\tnéli pereira\t40\tF\tted-talk-0003\tSUP\tPR\tCuritiba\tFalse\tTrue\ta.pt\t24\n"
    )
    wav_bytes = _make_speech_like_wav(24.0)
    wav_float_bytes = _make_speech_like_wav(24.0, float32=True)

    with zipfile.ZipFile(mock_zip_path, "w") as zf:
        zf.writestr("ttbacc-br-valid.tsv", tsv_data)
        zf.writestr("ttbacc-descr.txt", "summary")
        zf.writestr("ptbr_dictionary.txt", "ola o l a\n")
        zf.writestr("ptbr/mila d'oliveira/ted-talk-0001.wav", wav_bytes)
        zf.writestr("ptbr/mila d'oliveira/ted-talk-0001.hpt.txt", "texto transcrito humano 1")
        zf.writestr("ptbr/mila d'oliveira/ted-talk-0002.wav", wav_bytes)
        zf.writestr("ptbr/mila d'oliveira/ted-talk-0002.apt.txt", "texto automatico 2")
        zf.writestr("ptbr/néli pereira/ted-talk-0003.wav", wav_float_bytes)
        zf.writestr("ptbr_aligned/mila d'oliveira/ted-talk-0001.TextGrid", 'File type = "ooTextFile"\n')

    process_ttbacc(
        archive_path=mock_zip_path,
        out_root=out_dir,
        by_state=True,
        make_clips=True,
        clip_min_s=5.0,
        clip_max_s=10.0,
        max_clips_per_talk=10,
    )

    # Check by_speaker layout (slugified speaker dirs + TextGrid preserved)
    assert (out_dir / "by_speaker" / "mila_d_oliveira" / "ted-talk-0001.wav").exists()
    assert (out_dir / "by_speaker" / "mila_d_oliveira" / "ted-talk-0001.txt").read_text() == "texto transcrito humano 1"
    assert (out_dir / "by_speaker" / "mila_d_oliveira" / "ted-talk-0001.TextGrid").exists()
    assert (out_dir / "by_speaker" / "mila_d_oliveira" / "ted-talk-0002.txt").read_text() == "texto automatico 2"
    assert (out_dir / "by_speaker" / "neli_pereira" / "ted-talk-0003.wav").exists()

    # Check auxiliary files preserved
    assert (out_dir / "ttbacc-br-valid.tsv").exists()
    assert (out_dir / "ttbacc-descr.txt").exists()
    assert (out_dir / "ptbr_dictionary.txt").exists()

    # Check by_state
    assert (out_dir / "by_state" / "BA" / "ted-talk-0001.wav").exists()
    assert (out_dir / "by_state" / "BA" / "ted-talk-0002.wav").exists()
    assert (out_dir / "by_state" / "PR" / "ted-talk-0003.wav").exists()

    # Check metadata.csv
    meta_csv = out_dir / "metadata.csv"
    assert meta_csv.exists()
    with open(meta_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0]["speaker_id"] == "mila_d_oliveira"
        assert rows[0]["state"] == "BA"
        assert rows[0]["state_name"] == "Bahia"
        assert rows[0]["region"] == "NE"
        assert rows[0]["transcript_type"] == "human"
        assert rows[0]["textgrid_path"] == "by_speaker/mila_d_oliveira/ted-talk-0001.TextGrid"
        assert int(rows[0]["n_clips"]) >= 2
        assert rows[1]["transcript_type"] == "auto"
        assert rows[2]["region"] == "S"
        assert rows[2]["state_name"] == "Paraná"

    # Check speech clips and clips_metadata.csv
    clips_csv = out_dir / "clips_metadata.csv"
    assert clips_csv.exists()
    with open(clips_csv, "r", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
        assert len(clips) >= 6
        for c in clips:
            assert (out_dir / c["wav"]).exists()
            assert 5.0 <= float(c["duration"]) <= 10.5
