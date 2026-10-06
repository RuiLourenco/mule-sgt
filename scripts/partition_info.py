#!/usr/bin/env python3
"""Read and analyse the partition info files written by MSGTEncoder and MSGTDecoder.

The encoder writes ``info.json`` next to the bitstream (``--partition-info full|winner|off``).
The decoder writes ``decoded_info.json`` into its output directory (``--partition-info winner|off``).

Use it as a command line tool:

    partition_info.py summary info.json
    partition_info.py map info.json --field bpp --channel Y --view 4 4 -o bpp.png
    partition_info.py grid info.json --image decoded/004_004.ppm --view 4 4 -o grid.png
    partition_info.py candidates info.json --pixel 120 64 --view 4 4 -o costs.png
    partition_info.py compare info.json decoded/decoded_info.json
    partition_info.py gap info.json --method structure_tensor
    partition_info.py coarse-grid info.json --step 10

or as a library:

    import partition_info as pinfo
    info = pinfo.load("info.json")
    image, origin = pinfo.rasterize(info, channel="Y", view=(4, 4), field="psnr")

Only numpy is required; matplotlib is needed for the plotting commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

FORMAT_NAME = "mule-sgt-partition-info"
SUPPORTED_VERSION = 2
CHANNEL_NAMES = ("Y", "Cb", "Cr")
DEFAULT_PEAK = 1024.0  # same convention as the encoder's "Predicted PSNR" printout
SPLIT_FLAG_NAMES = {"T": "leaf", "S": "spatial split", "V": "view split"}
# The encoder assigns ~1e100*lambda to numerically unstable transforms; anything above
# this is a rejection marker rather than a real cost.
REJECTED_COST = 1e50

# Methods that belong to one estimator family (for the gap analysis).
METHOD_FAMILIES = {
    "structure_tensor": ("structure_tensor_h", "structure_tensor_v", "structure_tensor_avg",
                         "structure_tensor_pooled", "structure_tensor_per_direction",
                         "structure_tensor_eigen4d"),
    "logdet": ("logdet_h", "logdet_v", "logdet_avg"),
    "covariance": ("covariance_h", "covariance_v", "covariance_avg"),
    "zero": ("zero",),
    "grid_search": ("grid_search",),
    "rho_search": ("rho_search",),
    "least_squares_rho": ("least_squares_rho",),
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """One configuration evaluated by the RD search. ``cost`` is J = D + lambda*R."""

    method: str
    angleV: float
    angleH: float
    rhoS: float
    rhoT: float
    rhoU: float
    rhoV: float
    cost: float


@dataclass
class Unit:
    """A leaf of the partition tree (one transformed and coded block)."""

    position: tuple  # (t, s, v, u) in light field coordinates
    size: tuple      # (t, s, v, u)
    bits: float
    sse: Optional[float]
    ssi: Optional[dict]
    method: Optional[str]
    search_cost: Optional[float]
    candidates: list
    chosen_index: Optional[int]
    probes: list = field(default_factory=list)          # diagnostic evaluations (--probe-st-estimators)
    structure_tensor: Optional[np.ndarray] = None       # 4x4, axes (t, s, v, u)

    @property
    def samples(self) -> int:
        return int(np.prod(self.size))

    @property
    def bpp(self) -> float:
        """Bits per sample (per pixel of every view the unit spans)."""
        return self.bits / self.samples if self.samples else math.nan

    @property
    def mse(self) -> Optional[float]:
        return None if self.sse is None or not self.samples else self.sse / self.samples

    def psnr(self, peak: float = DEFAULT_PEAK) -> Optional[float]:
        mse = self.mse
        if mse is None:
            return None
        return math.inf if mse <= 0 else 10 * math.log10(peak * peak / mse)

    @property
    def chosen(self) -> Optional[Candidate]:
        return None if self.chosen_index is None else self.candidates[self.chosen_index]

    def covers_view(self, t: int, s: int) -> bool:
        return (self.position[0] <= t < self.position[0] + self.size[0]
                and self.position[1] <= s < self.position[1] + self.size[1])

    def covers_pixel(self, row: int, col: int) -> bool:
        return (self.position[2] <= row < self.position[2] + self.size[2]
                and self.position[3] <= col < self.position[3] + self.size[3])

    def best(self, methods: Iterable[str]) -> Optional[Candidate]:
        methods = set(methods)
        pool = [c for c in self.candidates if c.method in methods]
        return min(pool, key=lambda c: c.cost) if pool else None


@dataclass
class Partition:
    """One maximum-size block of one channel; its units are in coding order."""

    channel: int
    position: tuple
    size: tuple
    split_code: str
    bits: float
    sse: Optional[float]
    units: list

    @property
    def channel_name(self) -> str:
        return channel_name(self.channel)


@dataclass
class InfoFile:
    producer: str
    level: str
    metadata: dict
    partitions: list = field(default_factory=list)

    def units(self, channel=None) -> Iterator[tuple]:
        """Yields (partition, unit) pairs, optionally for one channel only."""
        ch = None if channel is None else parse_channel(channel)
        for p in self.partitions:
            if ch is None or p.channel == ch:
                for u in p.units:
                    yield p, u


def channel_name(channel: int) -> str:
    return CHANNEL_NAMES[channel] if 0 <= channel < len(CHANNEL_NAMES) else str(channel)


def parse_channel(channel) -> int:
    if isinstance(channel, int):
        return channel
    text = str(channel)
    if text.isdigit():
        return int(text)
    aliases = {"y": 0, "cb": 1, "co": 1, "cr": 2, "cg": 2}
    if text.lower() not in aliases:
        raise ValueError(f"unknown channel {channel!r}; use Y, Cb, Cr or 0, 1, 2")
    return aliases[text.lower()]


def _candidates_from_json(rows: list, fields: list) -> list:
    out = []
    for row in rows:
        values = dict(zip(fields, row))
        cost = values["cost"]
        out.append(Candidate(
            method=values["method"],
            angleV=values["angleV"], angleH=values["angleH"],
            rhoS=values["rhoS"], rhoT=values["rhoT"], rhoU=values["rhoU"], rhoV=values["rhoV"],
            cost=math.nan if cost is None else cost,
        ))
    return out


def _unit_from_json(j: dict, fields: list) -> Unit:
    tensor = j.get("structureTensor")
    return Unit(
        position=tuple(j["lightFieldPosition"]),
        size=tuple(j["size"]),
        bits=j.get("bits", 0.0),
        sse=j.get("sse"),
        ssi=j.get("ssi"),
        method=j.get("method"),
        search_cost=j.get("searchCost"),
        candidates=_candidates_from_json(j.get("candidates", []), fields),
        chosen_index=j.get("chosenCandidate"),
        probes=_candidates_from_json(j.get("probes", []), fields),
        structure_tensor=None if tensor is None else np.array(tensor, dtype=float).reshape(4, 4),
    )


def load(path: str) -> InfoFile:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise ValueError(f"{path}: not a partition info file (files written before format version "
                         f"{SUPPORTED_VERSION} are not supported)")
    if data.get("version") != SUPPORTED_VERSION:
        raise ValueError(f"{path}: unsupported version {data.get('version')}")
    fields = data["candidateFields"]
    info = InfoFile(producer=data.get("producer", ""), level=data.get("level", ""),
                    metadata=data.get("metadata", {}))
    for p in data["partitions"]:
        info.partitions.append(Partition(
            channel=p["channel"],
            position=tuple(p["lightFieldPosition"]),
            size=tuple(p["size"]),
            split_code=p.get("splitCode", ""),
            bits=p.get("bits", 0.0),
            sse=p.get("sse"),
            units=[_unit_from_json(u, fields) for u in p["codingUnits"]],
        ))
    return info


# --------------------------------------------------------------------------- #
# Per-unit fields and rasterization
# --------------------------------------------------------------------------- #

def _ssi_field(key: str) -> Callable[[Unit], Optional[float]]:
    return lambda u: None if u.ssi is None else u.ssi[key]


FIELDS: dict = {
    "bits": lambda u: u.bits,
    "bpp": lambda u: u.bpp,
    "sse": lambda u: u.sse,
    "mse": lambda u: u.mse,
    "psnr": lambda u: u.psnr(),
    "searchCost": lambda u: u.search_cost,
    "angleH": _ssi_field("angleH"),
    "angleV": _ssi_field("angleV"),
    "rhoS": _ssi_field("rhoS"),
    "rhoT": _ssi_field("rhoT"),
    "rhoU": _ssi_field("rhoU"),
    "rhoV": _ssi_field("rhoV"),
    "spatialSize": lambda u: u.size[2],
    "viewSize": lambda u: u.size[0],
    "candidates": lambda u: len(u.candidates),
}
CATEGORICAL_FIELDS = ("method",)


def bounding_box(info: InfoFile, channel) -> tuple:
    """(row0, col0, rows, cols) spanned by all partitions of a channel."""
    ch = parse_channel(channel)
    parts = [p for p in info.partitions if p.channel == ch]
    if not parts:
        raise ValueError(f"no partitions for channel {channel_name(ch)}")
    r0 = min(p.position[2] for p in parts)
    c0 = min(p.position[3] for p in parts)
    r1 = max(p.position[2] + p.size[2] for p in parts)
    c1 = max(p.position[3] + p.size[3] for p in parts)
    return r0, c0, r1 - r0, c1 - c0


def units_at_view(info: InfoFile, channel, view) -> list:
    return [u for _, u in info.units(channel) if u.covers_view(*view)]


def rasterize(info: InfoFile, channel, view, field) -> tuple:
    """Paints a per-unit value onto one view. Returns (image, (row0, col0)).

    ``field`` is a name from FIELDS, "method" (painted as category indices) or a
    callable taking a Unit. Pixels with no value are NaN.
    """
    r0, c0, rows, cols = bounding_box(info, channel)
    image = np.full((rows, cols), np.nan)
    if field in CATEGORICAL_FIELDS:
        names = method_names(info)
        getter = lambda u: None if u.method is None else names.index(u.method)
    elif callable(field):
        getter = field
    else:
        getter = FIELDS[field]
    for unit in units_at_view(info, channel, view):
        value = getter(unit)
        if value is None:
            continue
        r, c = unit.position[2] - r0, unit.position[3] - c0
        image[r:r + unit.size[2], c:c + unit.size[3]] = value
    return image, (r0, c0)


def method_names(info: InfoFile) -> list:
    return sorted({u.method for _, u in info.units() if u.method is not None})


def unit_rectangles(info: InfoFile, channel, view) -> list:
    """(row, col, height, width) of every unit at a view, relative to the bounding box."""
    r0, c0, _, _ = bounding_box(info, channel)
    return [(u.position[2] - r0, u.position[3] - c0, u.size[2], u.size[3])
            for u in units_at_view(info, channel, view)]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _fmt(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def summarize(info: InfoFile, peak: float = DEFAULT_PEAK) -> str:
    lines = [f"producer: {info.producer}   level: {info.level}"]
    for key in ("searchMethod", "lambda", "lightFieldSize", "maxPartitionSize", "minPartitionSize"):
        if key in info.metadata:
            lines.append(f"{key}: {info.metadata[key]}")

    lf_size = info.metadata.get("lightFieldSize")
    lf_samples = int(np.prod(lf_size[:4])) if lf_size else None
    psnr_by_channel = {}
    total_bits = 0.0
    for ch in sorted({p.channel for p in info.partitions}):
        parts = [p for p in info.partitions if p.channel == ch]
        units = [u for p in parts for u in p.units]
        bits = sum(p.bits for p in parts)
        total_bits += bits
        coded_samples = sum(u.samples for u in units)
        samples = lf_samples or coded_samples
        sses = [u.sse for u in units]
        psnr = None
        if units and all(s is not None for s in sses):
            mse = sum(sses) / samples
            psnr = math.inf if mse <= 0 else 10 * math.log10(peak * peak / mse)
            psnr_by_channel[ch] = psnr
        flags = Counter("".join(p.split_code for p in parts))
        lines.append("")
        lines.append(f"[{channel_name(ch)}] partitions {len(parts)}  units {len(units)}  "
                     f"bits {bits:.0f}  bpp {bits / samples:.5f}  PSNR {_fmt(psnr)} dB")
        lines.append("  split flags: " + ", ".join(f"{SPLIT_FLAG_NAMES.get(k, k)} {v}"
                                                   for k, v in sorted(flags.items())))
        sizes = Counter(f"{u.size[0]}x{u.size[1]}x{u.size[2]}x{u.size[3]}" for u in units)
        lines.append("  unit sizes: " + ", ".join(f"{k}: {v}" for k, v in sizes.most_common()))
        methods = Counter(u.method or "unrecorded" for u in units)
        lines.append("  chosen by: " + ", ".join(f"{k} {v}" for k, v in methods.most_common()))
    lines.append("")
    lines.append(f"total bits {total_bits:.0f} ({total_bits / 8:.0f} bytes, excluding the file header)")
    if len(psnr_by_channel) == 3 and all(math.isfinite(v) for v in psnr_by_channel.values()):
        yuv = (6 * psnr_by_channel[0] + psnr_by_channel[1] + psnr_by_channel[2]) / 8
        lines.append(f"PSNR-YUV (6:1:1) {yuv:.3f} dB")
    return "\n".join(lines)


def compare(encoder: InfoFile, decoder: InfoFile, max_report: int = 20, bits_tolerance: float = 1e-6) -> list:
    """Checks that a decoder info file describes the same tree and side information as the
    encoder's. Returns human readable mismatches (empty when the files agree)."""
    problems = []

    def report(msg):
        problems.append(msg)

    if len(encoder.partitions) != len(decoder.partitions):
        report(f"partition count differs: encoder {len(encoder.partitions)}, decoder {len(decoder.partitions)}")
    for index, (pe, pd) in enumerate(zip(encoder.partitions, decoder.partitions)):
        where = f"partition {index} ({channel_name(pe.channel)} at {pe.position})"
        if pe.channel != pd.channel:
            report(f"{where}: channel differs ({pe.channel} vs {pd.channel})")
        if pe.split_code != pd.split_code:
            report(f"{where}: split code differs ({pe.split_code!r} vs {pd.split_code!r})")
        if abs(pe.bits - pd.bits) > bits_tolerance * max(1.0, pe.bits):
            report(f"{where}: bits differ ({pe.bits:.3f} vs {pd.bits:.3f})")
        if len(pe.units) != len(pd.units):
            report(f"{where}: unit count differs ({len(pe.units)} vs {len(pd.units)})")
        for k, (ue, ud) in enumerate(zip(pe.units, pd.units)):
            uwhere = f"{where}, unit {k}"
            rel_e = tuple(a - b for a, b in zip(ue.position, pe.position))
            rel_d = tuple(a - b for a, b in zip(ud.position, pd.position))
            if rel_e != rel_d or ue.size != ud.size:
                report(f"{uwhere}: geometry differs (offset {rel_e} size {ue.size} vs offset {rel_d} size {ud.size})")
            if ue.ssi and ud.ssi:
                diffs = [key for key in ue.ssi if abs(ue.ssi[key] - ud.ssi.get(key, math.nan)) > 1e-9]
                if diffs:
                    report(f"{uwhere}: side information differs in {', '.join(diffs)}")
            if abs(ue.bits - ud.bits) > bits_tolerance * max(1.0, ue.bits):
                report(f"{uwhere}: bits differ ({ue.bits:.3f} vs {ud.bits:.3f})")
        if len(problems) >= max_report:
            break
    return problems[:max_report]


def gap_analysis(info: InfoFile, family: str, threshold: float, channel=None) -> str:
    """How far a heuristic's best candidate is from the grid-search optimum, per unit size.

    Needs files encoded at level "full" with a search that ran both the heuristic and a
    grid search (e.g. --all-heuristics or --refine-structure-tensor)."""
    methods = METHOD_FAMILIES[family]
    by_size = defaultdict(list)
    for _, unit in info.units(channel):
        heuristic = unit.best(methods)
        grid = unit.best(("grid_search",))
        if heuristic is None or grid is None or not grid.cost:
            continue
        angle_error = abs(heuristic.angleH - grid.angleH)
        cost_gap = (heuristic.cost - grid.cost) / grid.cost
        by_size[unit.size[2]].append((angle_error, cost_gap))
    if not by_size:
        return f"no units have both {family} and grid_search candidates (needs level 'full')"
    lines = [f"{family} vs grid search (angle error in degrees, relative cost gap)"]
    lines.append(f"{'size':>6} {'units':>6} {'median err':>11} {'mean err':>9} "
                 f"{'within ' + format(threshold, 'g') + 'deg':>12} {'median gap':>11} {'max gap':>9}")
    for size in sorted(by_size):
        errors = np.array([e for e, _ in by_size[size]])
        gaps = np.array([g for _, g in by_size[size]])
        lines.append(f"{size:>6} {len(errors):>6} {np.median(errors):>11.2f} {errors.mean():>9.2f} "
                     f"{(errors <= threshold).mean() * 100:>11.1f}% {np.median(gaps):>11.4f} {gaps.max():>9.4f}")
    return "\n".join(lines)


def coarse_grid_analysis(info: InfoFile, step: float, channel=None) -> str:
    """Would a coarse grid followed by a +/- step refinement find the fine-grid optimum?"""
    by_size = defaultdict(list)
    for _, unit in info.units(channel):
        grid = [c for c in unit.candidates if c.method == "grid_search"]
        if not grid:
            continue
        # The grid starts at the edge of the angle range, so align the coarse grid to it.
        start = min(c.angleH for c in grid)
        coarse = [c for c in grid if abs((c.angleH - start) / step - round((c.angleH - start) / step)) < 1e-6]
        if not coarse:
            continue
        best = min(grid, key=lambda c: c.cost)
        best_coarse = min(coarse, key=lambda c: c.cost)
        local = [c for c in grid if abs(c.angleH - best_coarse.angleH) <= step]
        best_local = min(local, key=lambda c: c.cost)
        found = abs(best.angleH - best_coarse.angleH) <= step
        rel_gap = (best_local.cost - best.cost) / best_local.cost if best_local.cost else 0.0
        by_size[unit.size[2]].append((found, rel_gap))
    if not by_size:
        return "no units have grid_search candidates (needs level 'full')"
    lines = [f"coarse grid every {step:g} deg, refined within +/-{step:g} deg"]
    lines.append(f"{'size':>6} {'units':>6} {'optimum found':>14} {'median gap':>11} {'max gap':>9}")
    for size in sorted(by_size):
        found = np.array([f for f, _ in by_size[size]])
        gaps = np.array([g for _, g in by_size[size]])
        lines.append(f"{size:>6} {len(found):>6} {found.mean() * 100:>13.1f}% {np.median(gaps):>11.4f} {gaps.max():>9.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_rectangles(ax, rectangles, color="white"):
    from matplotlib.patches import Rectangle
    for r, c, h, w in rectangles:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), w, h, fill=False, edgecolor=color, linewidth=0.6))


def read_ppm(path: str) -> np.ndarray:
    """Reads a binary PPM/PGM (8 or 16 bit) into a float array scaled to [0, 1]."""
    with open(path, "rb") as f:
        data = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    pos += 1
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    channels = 3 if magic == b"P6" else 1
    dtype = ">u2" if maxval > 255 else np.uint8
    image = np.frombuffer(data, dtype=dtype, count=width * height * channels, offset=pos)
    image = image.reshape(height, width, channels).astype(float) / maxval
    return image[..., 0] if channels == 1 else image


def plot_map(info, channel, view, field_name, output, draw_units=True, value_range=None):
    plt = _pyplot()
    image, _ = rasterize(info, channel, view, field_name)
    fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / max(image.shape[1], 1) + 0.5))
    if field_name in CATEGORICAL_FIELDS:
        names = method_names(info)
        cmap = plt.get_cmap("tab20", max(len(names), 1))
        im = ax.imshow(image, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
        bar = fig.colorbar(im, ax=ax, ticks=range(len(names)))
        bar.ax.set_yticklabels(names)
    else:
        vmin, vmax = value_range if value_range else (None, None)
        im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, label=field_name)
    if draw_units:
        _draw_rectangles(ax, unit_rectangles(info, channel, view))
    ax.set_title(f"{field_name}, channel {channel_name(parse_channel(channel))}, view {tuple(view)}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def plot_grid(info, channel, view, image_path, output):
    plt = _pyplot()
    image = read_ppm(image_path)
    r0, c0, rows, cols = bounding_box(info, channel)
    if image.shape[0] >= r0 + rows and image.shape[1] >= c0 + cols and (r0 or c0):
        image = image[r0:r0 + rows, c0:c0 + cols]  # full view, crop to the coded region
    fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / max(image.shape[1], 1)))
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
    _draw_rectangles(ax, unit_rectangles(info, channel, view), color="red")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def find_unit(info, channel, view, pixel) -> tuple:
    for partition, unit in info.units(channel):
        if unit.covers_view(*view) and unit.covers_pixel(*pixel):
            return partition, unit
    raise ValueError(f"no unit covers pixel {tuple(pixel)} at view {tuple(view)}")


def plot_candidates(info, channel, view, pixel, output):
    plt = _pyplot()
    _, unit = find_unit(info, channel, view, pixel)
    if not unit.candidates:
        raise ValueError("this unit has no recorded candidates (encode with --partition-info full)")
    angle_methods = defaultdict(list)
    rho_candidates = []
    rejected = [c for c in unit.candidates if not c.cost < REJECTED_COST]
    for c in unit.candidates:
        if c in rejected:
            continue
        (rho_candidates if c.method in ("rho_search", "least_squares_rho") else angle_methods[c.method]).append(c)
    panels = 2 if rho_candidates else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5), squeeze=False)
    ax = axes[0][0]
    for method, cands in sorted(angle_methods.items()):
        cands = sorted(cands, key=lambda c: c.angleH)
        style = "-o" if len(cands) > 3 else "s"
        ax.plot([c.angleH for c in cands], [c.cost for c in cands], style, markersize=3, label=method)
    chosen = unit.chosen
    if chosen is not None:
        ax.axvline(chosen.angleH, color="black", linestyle="--", linewidth=0.8, label=f"chosen ({chosen.method})")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("J = D + lambda R")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if rho_candidates:
        ax2 = axes[0][1]
        rho_candidates.sort(key=lambda c: c.rhoT)
        ax2.semilogx([1 - c.rhoT for c in rho_candidates], [c.cost for c in rho_candidates], "-o", markersize=3)
        if chosen is not None and chosen.method in ("rho_search", "least_squares_rho"):
            ax2.axvline(1 - chosen.rhoT, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("1 - angular rho")
        ax2.set_ylabel("J")
        ax2.grid(True, alpha=0.3)
    title = f"unit at {unit.position}, size {unit.size}, channel {channel_name(parse_channel(channel))}"
    if rejected:
        title += f"\n{len(rejected)} numerically unstable candidates not shown"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _add_view_args(parser):
    parser.add_argument("--channel", default="Y", help="Y, Cb, Cr or 0, 1, 2 (default Y)")
    parser.add_argument("--view", nargs=2, type=int, default=(0, 0), metavar=("T", "S"),
                        help="view index (default 0 0)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summary", help="rate, PSNR, split statistics and chosen methods per channel")
    p.add_argument("info")
    p.add_argument("--peak", type=float, default=DEFAULT_PEAK)

    p = sub.add_parser("map", help="heat map of a per-unit field over one view")
    p.add_argument("info")
    p.add_argument("--field", default="bpp", choices=sorted(FIELDS) + list(CATEGORICAL_FIELDS))
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--no-units", action="store_true", help="do not draw unit boundaries")
    p.add_argument("-o", "--output", required=True)
    _add_view_args(p)

    p = sub.add_parser("grid", help="draw unit boundaries over a view image")
    p.add_argument("info")
    p.add_argument("--image", required=True, help="PPM/PGM of the view (original or decoded)")
    p.add_argument("-o", "--output", required=True)
    _add_view_args(p)

    p = sub.add_parser("candidates", help="plot the search costs of the unit covering a pixel")
    p.add_argument("info")
    p.add_argument("--pixel", nargs=2, type=int, required=True, metavar=("ROW", "COL"),
                   help="light field coordinates, as stored in the file")
    p.add_argument("-o", "--output", required=True)
    _add_view_args(p)

    p = sub.add_parser("compare", help="check an encoder info file against a decoder info file")
    p.add_argument("encoder_info")
    p.add_argument("decoder_info")
    p.add_argument("--max-report", type=int, default=20)

    p = sub.add_parser("gap", help="how close a heuristic gets to the grid-search optimum")
    p.add_argument("info")
    p.add_argument("--method", default="structure_tensor", choices=sorted(METHOD_FAMILIES))
    p.add_argument("--threshold", type=float, default=10.0, help="angle tolerance in degrees")
    p.add_argument("--channel", default=None)

    p = sub.add_parser("coarse-grid", help="would a coarse grid plus local refinement find the optimum")
    p.add_argument("info")
    p.add_argument("--step", type=float, default=10.0)
    p.add_argument("--channel", default=None)

    args = parser.parse_args(argv)

    if args.command == "summary":
        print(summarize(load(args.info), args.peak))
    elif args.command == "map":
        plot_map(load(args.info), args.channel, args.view, args.field, args.output,
                 draw_units=not args.no_units, value_range=args.range)
    elif args.command == "grid":
        plot_grid(load(args.info), args.channel, args.view, args.image, args.output)
    elif args.command == "candidates":
        plot_candidates(load(args.info), args.channel, args.view, args.pixel, args.output)
    elif args.command == "compare":
        problems = compare(load(args.encoder_info), load(args.decoder_info), args.max_report)
        if problems:
            print("\n".join(problems))
            return 1
        print("encoder and decoder agree on every partition, unit, side information and bit count")
    elif args.command == "gap":
        print(gap_analysis(load(args.info), args.method, args.threshold, args.channel))
    elif args.command == "coarse-grid":
        print(coarse_grid_analysis(load(args.info), args.step, args.channel))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
