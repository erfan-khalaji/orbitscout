"""Score DJI flights for 3D-reconstruction suitability from telemetry alone.

Photogrammetry and Gaussian splatting want the camera to travel *around* the
subject. A flight that dollies past a subject in a straight line yields parallax
along one axis only: you recover a facade, never a volume. Reading that off the
GPS track costs milliseconds, so a large archive can be triaged before a single
frame is decoded.

Degenerate case worth naming, because the obvious metric gets it wrong:
accumulated bearing change about the track centroid looks like the natural
"did it go around?" statistic, but an out-and-back straight line passes through
its own centroid, so the bearing flips ~180 degrees at each end and the total
lands near 360 -- scoring a pure dolly as a perfect orbit. Isotropy of the
track's spatial covariance disambiguates: a line has one dominant eigenvalue,
a circle has two equal ones. Sweep is retained only as a supporting signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

from .telemetry import Telemetry, parse_srt


@dataclass
class FlightScore:
    name: str
    path: str
    when: str
    duration_s: float
    n_fixes: int
    isotropy: float      # 0 = straight line, 1 = perfectly circular footprint
    radius_cv: float     # coefficient of variation of radius; low = steady orbit
    sweep_deg: float     # unsigned accumulated bearing change about centroid
    path_m: float        # total ground distance travelled
    net_m: float         # start-to-end displacement
    radius_m: float      # mean distance from centroid
    alt_m: float         # mean relative altitude
    alt_cv: float        # altitude stability
    hour: int            # local hour of capture, for a daylight guess
    score: float         # 0..1 reconstruction suitability
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def _covariance_isotropy(xy: list[tuple[float, float]]) -> float:
    """sqrt(lambda_min / lambda_max) of the 2-D spatial covariance.

    Returns ~0.0 for a collinear track and ~1.0 for a circular one. This is the
    statistic that separates a true orbit from an out-and-back dolly.
    """
    n = len(xy)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in xy) / n
    my = sum(p[1] for p in xy) / n
    sxx = sum((p[0] - mx) ** 2 for p in xy) / n
    syy = sum((p[1] - my) ** 2 for p in xy) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in xy) / n
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l_max, l_min = tr / 2.0 + disc, tr / 2.0 - disc
    if l_max <= 1e-9:
        return 0.0
    return math.sqrt(max(l_min, 0.0) / l_max)


def _sweep_degrees(xy: list[tuple[float, float]]) -> float:
    """Unsigned accumulated bearing change about the centroid, in degrees."""
    n = len(xy)
    if n < 3:
        return 0.0
    mx = sum(p[0] for p in xy) / n
    my = sum(p[1] for p in xy) / n
    ang = [math.atan2(y - my, x - mx) for x, y in xy]
    total = 0.0
    for i in range(len(ang) - 1):
        d = ang[i + 1] - ang[i]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        total += d
    return abs(math.degrees(total))


def _cv(vals: list[float]) -> float:
    """Coefficient of variation; 9.0 sentinel when the mean is degenerate."""
    if not vals:
        return 9.0
    m = sum(vals) / len(vals)
    if abs(m) < 1e-6:
        return 9.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return sd / abs(m)


def score_flight(tel: Telemetry, target_hz: float = 3.0, fps: float = 30.0) -> FlightScore | None:
    """Score one parsed flight. Returns None if telemetry is too thin to judge."""
    if len(tel.fixed) < 60:
        return None

    stride = max(1, int(round(fps / target_hz)))
    enu = tel.enu(stride=stride)
    if len(enu) < 20:
        return None

    xy = [(e, n) for e, n, _ in enu]
    alts = [u for _, _, u in enu]
    n = len(xy)

    mx = sum(p[0] for p in xy) / n
    my = sum(p[1] for p in xy) / n
    radii = [math.hypot(x - mx, y - my) for x, y in xy]
    radius = sum(radii) / n

    path = sum(math.dist(xy[i], xy[i + 1]) for i in range(n - 1))
    net = math.dist(xy[0], xy[-1])

    iso = _covariance_isotropy(xy)
    sweep = _sweep_degrees(xy)
    r_cv = _cv(radii)

    abs_alts = [f.rel_alt for f in tel.fixed if f.rel_alt is not None]
    alt_mean = sum(abs_alts) / len(abs_alts) if abs_alts else 0.0

    stamp = next((f.stamp for f in tel.frames if f.stamp), None)
    hour = stamp.hour if stamp else -1
    when = stamp.strftime("%Y-%m-%d %H:%M") if stamp else "unknown"

    # Suitability: an isotropic footprint at a steady radius, with enough
    # travel to give real baseline. Each factor saturates rather than
    # dominating, so one strong term cannot rescue a fatally weak one.
    f_iso = iso
    f_steady = max(0.0, 1.0 - min(r_cv, 1.0))
    f_travel = min(path / 150.0, 1.0)
    f_baseline = min(radius / 15.0, 1.0)
    score = f_iso * f_steady * f_travel * f_baseline

    if score >= 0.30:
        verdict = "STRONG - full orbit, reconstructable"
    elif score >= 0.15:
        verdict = "USABLE - partial arc, expect gaps"
    elif iso < 0.25 and path > 40:
        verdict = "DOLLY - straight pass, 2.5D only"
    else:
        verdict = "WEAK - insufficient parallax"

    return FlightScore(
        name=tel.path.stem,
        path=str(tel.path),
        when=when,
        duration_s=round(tel.duration, 1),
        n_fixes=len(tel.fixed),
        isotropy=round(iso, 3),
        radius_cv=round(r_cv, 3),
        sweep_deg=round(sweep, 1),
        path_m=round(path, 1),
        net_m=round(net, 1),
        radius_m=round(radius, 1),
        alt_m=round(alt_mean, 1),
        alt_cv=round(_cv(alts) if alts else 9.0, 3),
        hour=hour,
        score=round(score, 4),
        verdict=verdict,
    )


def scan(roots: list[str | Path], target_hz: float = 3.0) -> list[FlightScore]:
    """Score every .SRT found under the given roots, best first."""
    out: list[FlightScore] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        srts = [root] if root.is_file() else sorted(root.rglob("*.SRT")) + sorted(root.rglob("*.srt"))
        for srt in srts:
            rp = srt.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                fs = score_flight(parse_srt(srt), target_hz=target_hz)
            except Exception:
                continue
            if fs:
                out.append(fs)
    out.sort(key=lambda f: f.score, reverse=True)
    return out
