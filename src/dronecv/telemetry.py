"""Parse DJI SRT telemetry sidecars into per-frame flight state.

DJI writes one subtitle cue per video frame. Cue payload looks like:

    FrameCnt: 1, DiffTime: 33ms
    2026-08-19 21:07:02.430
    [iso: 100] [shutter: 1/640.0] [fnum: 1.7] [ev: 0] [color_md: default]
    [focal_len: 24.00] [latitude: 53.530183] [longitude: -113.546103]
    [rel_alt: 29.100 abs_alt: 681.444] [ct: 5544]

Field presence varies by airframe and firmware, so every field is optional and
the parser never raises on an unknown or missing key.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# [key: value] pairs, where value runs to the next ']'. rel_alt/abs_alt share a
# bracket, so values are re-split on whitespace-delimited "key: value" runs.
_BRACKET = re.compile(r"\[([^\]]+)\]")
_PAIR = re.compile(r"([A-Za-z_]+)\s*:\s*([^\s\]]+)")
_FRAMECNT = re.compile(r"FrameCnt\s*:\s*(\d+)")
_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


@dataclass
class FrameState:
    """Flight + camera state at a single video frame."""

    frame: int
    t: float  # seconds from clip start
    lat: float | None = None
    lon: float | None = None
    rel_alt: float | None = None
    abs_alt: float | None = None
    iso: float | None = None
    shutter: float | None = None
    fnum: float | None = None
    focal_len: float | None = None
    ev: float | None = None
    color_temp: float | None = None
    stamp: datetime | None = None

    @property
    def has_fix(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass
class Telemetry:
    """All per-frame states for one clip, plus derived local-ENU geometry."""

    path: Path
    frames: list[FrameState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fixed(self) -> list[FrameState]:
        """Frames that carry a usable GPS fix."""
        return [f for f in self.frames if f.has_fix]

    @property
    def duration(self) -> float:
        return self.frames[-1].t if self.frames else 0.0

    def origin(self) -> tuple[float, float]:
        """Centroid of the GPS track, used as the local ENU origin."""
        pts = self.fixed
        if not pts:
            raise ValueError(f"{self.path.name}: no GPS fixes")
        return (
            sum(p.lat for p in pts) / len(pts),
            sum(p.lon for p in pts) / len(pts),
        )

    def enu(self, stride: int = 1) -> list[tuple[float, float, float]]:
        """Track as local East/North/Up metres about the centroid.

        Uses a first-order equirectangular projection with latitude-corrected
        metres-per-degree. Drone tracks span tens of metres, where the error
        against a full geodetic solution is well under a centimetre.
        """
        pts = self.fixed[::stride]
        if not pts:
            return []
        lat0, lon0 = self.origin()
        m_lat, m_lon = meters_per_degree(lat0)
        base = pts[0].rel_alt or 0.0
        return [
            (
                (p.lon - lon0) * m_lon,
                (p.lat - lat0) * m_lat,
                (p.rel_alt or base) - base,
            )
            for p in pts
        ]

    def intrinsics_hint(self) -> dict[str, float]:
        """Median camera settings, for seeding SfM with a sane prior."""
        out: dict[str, float] = {}
        for key in ("focal_len", "fnum"):
            vals = [getattr(f, key) for f in self.frames if getattr(f, key) is not None]
            if vals:
                vals.sort()
                out[key] = vals[len(vals) // 2]
        return out


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of latitude and longitude on the WGS-84 ellipsoid."""
    lat = math.radians(lat_deg)
    s = math.sin(lat)
    w = math.sqrt(1.0 - WGS84_E2 * s * s)
    m_lat = math.pi * WGS84_A * (1.0 - WGS84_E2) / (180.0 * w**3)
    m_lon = math.pi * WGS84_A * math.cos(lat) / (180.0 * w)
    return m_lat, m_lon


def _to_float(raw: str) -> float | None:
    """Coerce a DJI field value to float, handling '1/640.0' shutter notation."""
    raw = raw.strip().rstrip(",")
    if not raw or raw.lower() in {"none", "n/a", "default"}:
        return None
    if "/" in raw:
        num, _, den = raw.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else None
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_srt(path: str | Path) -> Telemetry:
    """Parse a DJI .SRT sidecar. Malformed cues are skipped, not fatal."""
    path = Path(path)
    text = path.read_text(errors="ignore")
    tel = Telemetry(path=path)

    # Cues are blank-line separated; strip the HTML DJI wraps the payload in.
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"<[^>]+>", " ", block)
        if "FrameCnt" not in block and "latitude" not in block:
            continue

        m = _FRAMECNT.search(block)
        frame = int(m.group(1)) if m else len(tel.frames) + 1

        t = _cue_start_seconds(block)
        st = FrameState(frame=frame, t=t if t is not None else (frame - 1) / 30.0)

        m = _TIMESTAMP.search(block)
        if m:
            raw = m.group(1).replace(",", ".")
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    st.stamp = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue

        for group in _BRACKET.findall(block):
            for key, val in _PAIR.findall(group):
                v = _to_float(val)
                if v is None:
                    continue
                k = key.lower()
                if k == "latitude":
                    st.lat = v
                elif k == "longitude":
                    st.lon = v
                elif k == "rel_alt":
                    st.rel_alt = v
                elif k == "abs_alt":
                    st.abs_alt = v
                elif k == "iso":
                    st.iso = v
                elif k == "shutter":
                    st.shutter = v
                elif k == "fnum":
                    st.fnum = v
                elif k == "focal_len":
                    st.focal_len = v
                elif k == "ev":
                    st.ev = v
                elif k == "ct":
                    st.color_temp = v

        tel.frames.append(st)

    tel.frames.sort(key=lambda f: f.frame)
    return tel


def _cue_start_seconds(block: str) -> float | None:
    """Start time of an SRT cue ('00:00:01,234 --> ...') in seconds."""
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->", block)
    if not m:
        return None
    h, mnt, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mnt * 60 + s + ms / 1000.0
