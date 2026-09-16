"""
Shared helpers for the tools/download_*.py scripts.

Nothing here is specific to a single corpus: resumable HTTP downloads, streaming
tar readers, output-directory resolution (honouring ACCENTS_BASE) and a couple of
tiny progress/formatting helpers.
"""
from __future__ import annotations

import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Iterator

UA = {'User-Agent': 'Mozilla/5.0 (pt-br-accent-toolbox)'}
CHUNK = 1 << 20  # 1 MiB


# ── formatting ───────────────────────────────────────────────────────────────

def human(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def info(msg: str):
    print(f'  ✓ {msg}', flush=True)


def warn(msg: str):
    print(f'  ⚠ {msg}', flush=True)


class Progress:
    """Single-line byte counter, throttled to one repaint per second."""

    def __init__(self, total: int | None = None, label: str = ''):
        self.total = total or 0
        self.label = label
        self.done = 0
        self.t0 = time.time()
        self._last = 0.0

    def add(self, n: int):
        self.done += n
        now = time.time()
        if now - self._last < 1.0:
            return
        self._last = now
        self.paint()

    def paint(self, end: str = ''):
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = self.done / elapsed
        if self.total:
            pct = 100.0 * self.done / self.total
            msg = (f'\r  {self.label} {human(self.done)}/{human(self.total)} '
                   f'({pct:5.1f}%) {human(rate)}/s   ')
        else:
            msg = f'\r  {self.label} {human(self.done)} {human(rate)}/s   '
        sys.stdout.write(msg + end)
        sys.stdout.flush()

    def close(self):
        self.paint(end='\n')


# ── paths ────────────────────────────────────────────────────────────────────

def accents_base() -> Path:
    """Root data directory — matches pt_br_accent_toolbox.config.BASE."""
    return Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents'))


def resolve_out(out: str | None, default_name: str) -> Path:
    """`--out` if given, else {ACCENTS_BASE}/{default_name}."""
    path = Path(out) if out else accents_base() / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── HTTP ─────────────────────────────────────────────────────────────────────

def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, headers=UA, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get('Content-Length')
            return int(n) if n else None
    except Exception:
        return None


def http_download(url: str, dest: Path, label: str = '') -> Path:
    """Download `url` to `dest`, resuming a partial file if one is there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        info(f'{dest.name} already complete ({human(total)})')
        return dest

    headers = dict(UA)
    mode = 'wb'
    if have:
        headers['Range'] = f'bytes={have}-'
        mode = 'ab'
        print(f'  resuming {dest.name} at {human(have)}', flush=True)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:   # server ignored the Range header
            have, mode = 0, 'wb'
        prog = Progress(total, label or dest.name)
        prog.done = have
        with open(dest, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                prog.add(len(buf))
        prog.close()
    return dest


def stream_tar(url: str, compression: str = 'gz') -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo]]:
    """
    Iterate a remote tar archive without storing it.

    Yields (tarfile, member) pairs; call `tf.extractfile(member)` on the ones you
    want. The stream is sequential — members must be consumed in archive order and
    each file object is only valid until the next iteration step.
    """
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        with tarfile.open(fileobj=r, mode=f'r|{compression}') as tf:
            for member in tf:
                yield tf, member


def fetch_text(url: str, encoding: str = 'utf-8') -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode(encoding)
