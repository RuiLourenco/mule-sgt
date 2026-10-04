#!/usr/bin/env python3
"""Orientation-estimation ablation: pooled vs per-direction vs 4D-eigenvector structure tensor.

Each subcommand produces the numbers for one part of the ablation:

  synthetic  Controlled experiment on synthetic Lambertian light fields
             L(t,s,v,u) = f(u - d*s, v - d*t), isotropic and one-directional textures.
  validate   Checks that this script's estimators (and gradients) match the codec, using
             an info.json recorded with --probe-st-estimators.
  accuracy   Accuracy on real light fields: angular error and relative RD excess of each
             estimator against the grid-search optimum, binned by coherence and |theta*|.
  run        Runs the encodes for one light field: the grid-search reference (with probes)
             and the structure-tensor search with each estimator, at several lambdas.
  bdrate     BD-rate and relative encoding time of each estimator against the reference.

Typical use for one light field:

  orientation_ablation.py run --light-field LF_DIR --out runs/greek --lambdas 27 672 10140 117590
  orientation_ablation.py accuracy runs/greek/reference_*/info.json -o greek_accuracy
  orientation_ablation.py bdrate runs/greek
  orientation_ablation.py synthetic -o synthetic

Conventions follow the codec: axes (t, s, v, u), angles are atan(disparity) in degrees,
PSNR uses peak 1024 and the 6:1:1 YUV weighting printed by the encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partition_info as pinfo  # noqa: E402

ESTIMATORS = ("pooled", "per_direction", "eigen4d")
PROBE_METHOD = {
    "pooled": "structure_tensor_pooled",
    "per_direction": "structure_tensor_per_direction",
    "eigen4d": "structure_tensor_eigen4d",
}
ENCODER_FLAG = {"pooled": "pooled", "per_direction": "per-direction", "eigen4d": "eigen4d", "legacy": "legacy"}
LABEL = {"pooled": "Pooled", "per_direction": "Per-direction", "eigen4d": "4D eigenvector", "legacy": "Legacy"}
REJECTED_COST = 1e50  # the encoder's numerically-unstable marker

# --------------------------------------------------------------------------- #
# Estimators: mirror Block4D_::angleFromStructureTensor (axes 0=t, 1=s, 2=v, 3=u)
# --------------------------------------------------------------------------- #


def epi_orientation(jaa: float, jas: float, jss: float) -> float:
    """Direction along the EPI lines of the 2D tensor [[jaa, jas], [jas, jss]] over
    (angular, spatial): atan(d) in degrees for f(spatial - d*angular). NaN for a zero tensor."""
    if jaa + jss <= 0:
        return math.nan
    return 0.5 * math.degrees(math.atan2(-2.0 * jas, jss - jaa))


def _mean_finite(a: float, b: float) -> float:
    values = [x for x in (a, b) if math.isfinite(x)]
    return sum(values) / len(values) if values else math.nan


def estimate(T: np.ndarray, estimator: str) -> dict:
    """Returns {"angle", "h", "v"} in degrees (unclamped; NaN when undefined)."""
    T = np.asarray(T, dtype=float)
    if estimator == "pooled":
        angle = epi_orientation(T[1, 1] + T[0, 0], T[1, 3] + T[0, 2], T[3, 3] + T[2, 2])
        return {"angle": angle, "h": math.nan, "v": math.nan}
    if estimator == "per_direction":
        h = epi_orientation(T[1, 1], T[1, 3], T[3, 3])
        v = epi_orientation(T[0, 0], T[0, 2], T[2, 2])
        return {"angle": _mean_finite(h, v), "h": h, "v": v}
    if estimator == "eigen4d":
        if np.abs(T).sum() <= 0:
            return {"angle": math.nan, "h": math.nan, "v": math.nan}
        _, Q = np.linalg.eigh(T)  # ascending eigenvalues, like at::linalg_eigh
        e = Q[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -math.degrees(math.atan(e[1] / e[3])) if e[3] != 0 or e[1] != 0 else math.nan
            v = -math.degrees(math.atan(e[0] / e[2])) if e[2] != 0 or e[0] != 0 else math.nan
        return {"angle": _mean_finite(h, v), "h": h, "v": v}
    raise ValueError(f"unknown estimator {estimator}")


def angle_range(disparity_range) -> tuple:
    """Mirror of SgtSideInfo::angleRangeFromDispRange."""
    lo = math.floor(math.degrees(math.atan(disparity_range[0])))
    hi = math.degrees(math.atan(disparity_range[1]))
    n = math.ceil((hi - lo) / 0.1)
    return lo, lo + n * 0.1


def codec_angle(T: np.ndarray, estimator: str, disparity_range) -> float:
    """The angle the codec would test: NaN mapped to 0, clamped to the angle range."""
    angle = estimate(T, estimator)["angle"]
    if not math.isfinite(angle):
        return 0.0
    lo, hi = angle_range(disparity_range)
    return min(max(angle, lo), hi)


def coherence(T: np.ndarray) -> float:
    """((l1 - l2) / (l1 + l2))^2 of the pooled 2x2 EPI tensor; 0 for a zero tensor."""
    A = T[1, 1] + T[0, 0]
    B = T[1, 3] + T[0, 2]
    C = T[3, 3] + T[2, 2]
    trace = A + C
    if trace <= 0:
        return 0.0
    return ((A - C) ** 2 + 4 * B ** 2) / trace ** 2


# --------------------------------------------------------------------------- #
# Gradients: mirror compute_first_order_derivatives_separable (LightField.cpp)
# --------------------------------------------------------------------------- #


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_derivative_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = -x * np.exp(-(x * x) / (2 * sigma * sigma)) / (sigma * sigma)
    scale = max(k[k > 0].sum(), -k[k < 0].sum())
    return k / scale if scale > 0 else k


def _correlate_same(data: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """torch conv1d with padding='same' and zero padding: a cross-correlation."""
    c = len(kernel) // 2
    pad = [(0, 0)] * data.ndim
    pad[axis] = (c, c)
    padded = np.pad(data, pad)
    out = np.zeros_like(data, dtype=float)
    n = data.shape[axis]
    for i, w in enumerate(kernel):
        out += w * np.take(padded, np.arange(i, i + n), axis=axis)
    return out


def gradients(lf: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-derivative gradients of a 4D array (t, s, v, u). Returns shape (..., 4)."""
    g = gaussian_kernel(size, sigma)
    dg = gaussian_derivative_kernel(size, sigma)
    out = np.empty(lf.shape + (4,))
    for dim in range(4):
        temp = lf.astype(float)
        for other in range(4):
            temp = _correlate_same(temp, dg if other == dim else g, other)
        out[..., dim] = temp
    return out


def block_tensor(grads: np.ndarray, position, size, lf_spatial, angular_border: int = 2) -> np.ndarray:
    """Mirror of Block4D_::computeGradientSum for all 16 entries (no invalid corners)."""
    t0, s0, v0, u0 = position
    nt, ns, nv, nu = size
    spatial_border = 2 if (u0 == 0 or u0 == lf_spatial[1] - 1 or v0 == 0 or v0 == lf_spatial[0] - 1) else 0
    block = grads[t0 + angular_border:t0 + nt - angular_border,
                  s0 + angular_border:s0 + ns - angular_border,
                  v0 + spatial_border:v0 + nv - spatial_border,
                  u0 + spatial_border:u0 + nu - spatial_border]
    flat = block.reshape(-1, 4)
    return flat.T @ flat


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #


def synthetic_light_field(d: float, texture: str, rng: np.random.Generator, views: int = 9,
                          n: int = 128, cutoff: float = 0.15, mean: float = 512.0,
                          std: float = 100.0, noise: float = 2.0) -> np.ndarray:
    """L(t, s, v, u) = f(u - d*(s - c), v - d*(t - c)) + noise, with f band-limited and
    periodic so sub-pixel shifts are exact (applied as Fourier phase ramps)."""
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    if texture == "isotropic":
        spectrum = np.fft.fft2(rng.standard_normal((n, n)))
        spectrum *= np.exp(-(fx ** 2 + fy ** 2) / (2 * cutoff ** 2))
    elif texture == "one_directional":  # varies along u only
        row = np.fft.fft(rng.standard_normal(n)) * np.exp(-(np.fft.fftfreq(n) ** 2) / (2 * cutoff ** 2))
        spectrum = np.zeros((n, n), dtype=complex)
        spectrum[0, :] = row * n
    else:
        raise ValueError(texture)
    base = np.real(np.fft.ifft2(spectrum))
    scale = std / base.std() if base.std() > 0 else 1.0
    spectrum *= scale
    c = (views - 1) / 2
    lf = np.empty((views, views, n, n))
    for t in range(views):
        for s in range(views):
            du, dv = d * (s - c), d * (t - c)
            phase = np.exp(-2j * np.pi * (fx * du + fy * dv))
            lf[t, s] = mean + np.real(np.fft.ifft2(spectrum * phase))
    return lf + rng.normal(0.0, noise, lf.shape)


def run_synthetic(args) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    disparities = np.round(np.arange(args.d_min, args.d_max + 1e-9, args.d_step), 6)
    rng = np.random.default_rng(args.seed)
    block = args.block
    rows = []
    for texture in ("isotropic", "one_directional"):
        for d in disparities:
            for trial in range(args.trials):
                lf = synthetic_light_field(d, texture, rng, n=args.size, cutoff=args.cutoff, noise=args.noise)
                # Only the block plus the kernel margin matters: zero padding affects just
                # the outer 2 samples, which the block excludes.
                start = (args.size - block) // 2 - 2
                region = lf[:, :, start:start + block + 4, start:start + block + 4]
                grads = gradients(region)
                T = block_tensor(grads, (0, 0, 2, 2), (9, 9, block, block), (block + 4, block + 4))
                row = {"texture": texture, "d": float(d), "trial": trial,
                       "true": math.degrees(math.atan(d)), "coherence": coherence(T)}
                for name in ESTIMATORS:
                    est = estimate(T, name)
                    row[name] = est["angle"]
                    row[name + "_h"] = est["h"]
                    row[name + "_v"] = est["v"]
                rows.append(row)
    _write_csv(out / "synthetic.csv", rows)

    lines = [f"synthetic: d in [{args.d_min}, {args.d_max}] step {args.d_step}, {args.trials} trials, "
             f"noise sigma {args.noise} (texture std 100), block 9x9x{block}x{block}"]
    for texture in ("isotropic", "one_directional"):
        sub = [r for r in rows if r["texture"] == texture]
        lines.append(f"\n[{texture}]  absolute error vs atan(d), degrees")
        lines.append(f"  {'estimator':<22}{'median':>9}{'p95':>9}{'max':>9}")
        columns = list(ESTIMATORS)
        if texture == "one_directional":
            columns += ["per_direction_v", "eigen4d_v"]
        for name in columns:
            err = np.array([abs(r[name] - r["true"]) if math.isfinite(r[name]) else math.nan for r in sub])
            finite = err[np.isfinite(err)]
            undefined = len(err) - len(finite)
            if len(finite):
                lines.append(f"  {name:<22}{np.median(finite):>9.3f}{np.percentile(finite, 95):>9.3f}"
                             f"{finite.max():>9.3f}" + (f"   ({undefined} undefined)" if undefined else ""))
            else:
                lines.append(f"  {name:<22}  undefined in all {undefined} cases")
    summary = "\n".join(lines)
    (out / "synthetic_summary.txt").write_text(summary + "\n")
    print(summary)

    plt = _pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, texture in zip(axes, ("isotropic", "one_directional")):
        sub = [r for r in rows if r["texture"] == texture]
        truth = sorted({r["true"] for r in sub})
        ax.plot(truth, truth, color="0.6", linewidth=1, label="true")
        for name, marker in zip(ESTIMATORS, ("o", "s", "^")):
            med = [np.nanmedian([r[name] for r in sub if r["true"] == x]) for x in truth]
            ax.plot(truth, med, marker=marker, markersize=3, linewidth=1, label=LABEL[name])
        ax.set_title(texture.replace("_", "-") + " texture")
        ax.set_xlabel("true orientation atan(d) (degrees)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("estimated orientation (degrees)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "synthetic.pdf")
    fig.savefig(out / "synthetic.png", dpi=150)


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def read_light_field(directory: str, views: int = 9) -> np.ndarray:
    """Channel 0 (as the codec uses for gradients) of AAA_BBB.ppm views, as (t, s, v, u).

    The codec maps the first number of the file name to the s axis (it pairs with u in
    the horizontal EPI) and the second to t; validate confirms this against recorded tensors."""
    files = sorted(f for f in os.listdir(directory) if f.endswith(".ppm") and f[3] == "_")
    s_ids = sorted({f[:3] for f in files})[:views]
    t_ids = sorted({f[4:7] for f in files})[:views]
    lf = None
    for si, s in enumerate(s_ids):
        for ti, t in enumerate(t_ids):
            image = _read_ppm_raw(os.path.join(directory, f"{s}_{t}.ppm"))
            if lf is None:
                lf = np.empty((len(t_ids), len(s_ids)) + image.shape[:2])
            lf[ti, si] = image[..., 0]
    return lf


def _read_ppm_raw(path: str) -> np.ndarray:
    image = pinfo.read_ppm(path)
    with open(path, "rb") as f:
        header = f.read(64).split()
    return image * int(header[3])


def run_validate(args) -> int:
    info = pinfo.load(args.info)
    disparity_range = info.metadata.get("disparityRange", [-3.1, 3.5])
    units = [u for _, u in info.units(args.channel) if u.structure_tensor is not None]
    if not units:
        print("no units with a recorded structure tensor (encode with --probe-st-estimators)")
        return 1

    # 1. Estimator formulas: recompute each probe's angle from the recorded tensor.
    worst = 0.0
    for unit in units:
        for probe in unit.probes:
            name = next(k for k, v in PROBE_METHOD.items() if v == probe.method)
            expected = codec_angle(unit.structure_tensor, name, disparity_range)
            worst = max(worst, abs(expected - probe.angleH))
    print(f"estimators: max |python - codec| angle over {len(units)} units = {worst:.4f} deg "
          f"(angles are coded at 0.1 deg, so <= 0.05 means identical)")
    status = 0 if worst <= 0.05 + 1e-9 else 1

    # 2. Cost consistency: probes run on the main encoder, grid candidates on the thread
    #    pool. Where both evaluated the same coded configuration, the costs must agree,
    #    otherwise the relative RD excess would compare unlike quantities.
    pairs = 0
    worst_cost = 0.0
    for unit in units:
        grid = {(round(c.angleH, 6), round(c.rhoT, 9), round(c.rhoU, 9)): c.cost
                for c in unit.candidates if c.method == "grid_search"}
        for probe in unit.probes:
            key = (round(probe.angleH, 6), round(probe.rhoT, 9), round(probe.rhoU, 9))
            if key in grid and grid[key] < REJECTED_COST:
                pairs += 1
                worst_cost = max(worst_cost, abs(grid[key] - probe.cost) / max(abs(grid[key]), 1e-12))
    if pairs:
        print(f"costs: {pairs} probe/grid pairs at the same configuration, max relative difference {worst_cost:.2e}")
        if worst_cost > 1e-5:  # float noise between threaded and main-thread evaluations
            status = 1
    else:
        print("costs: no probe landed on a grid angle (needs a grid-search reference)")

    # 3. Gradients: recompute tensors of interior blocks from the light field.
    if args.light_field:
        lf = read_light_field(args.light_field)
        grads = gradients(lf)
        spatial = lf.shape[2:]
        errors = []
        for unit in units:
            p, s = unit.position, unit.size
            interior = (p[2] >= 2 and p[3] >= 2 and p[2] + s[2] <= spatial[0] - 2 and p[3] + s[3] <= spatial[1] - 2)
            if not interior or min(s[0], s[1]) <= 4:
                continue
            T = block_tensor(grads, p, s, spatial)
            ref = unit.structure_tensor
            errors.append(np.abs(T - ref).max() / max(np.abs(ref).max(), 1e-12))
        if errors:
            errors = np.array(errors)
            print(f"gradients: relative max tensor difference over {len(errors)} interior units: "
                  f"median {np.median(errors):.2e}, max {errors.max():.2e}")
            if np.median(errors) > 1e-3:
                status = 1
        else:
            print("gradients: no interior units to compare")
    return status


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #


def accuracy_rows(info: pinfo.InfoFile, channel, source: str) -> list:
    rows = []
    for partition, unit in info.units(channel):
        if unit.structure_tensor is None or not unit.probes:
            continue
        grid = [c for c in unit.candidates if c.method == "grid_search" and c.cost < REJECTED_COST]
        if not grid:
            continue
        best = min(grid, key=lambda c: c.cost)
        row = {"source": source, "channel": partition.channel_name,
               "position": "x".join(map(str, unit.position)), "size": "x".join(map(str, unit.size)),
               "spatial_size": unit.size[2], "theta_star": best.angleH, "J_star": best.cost,
               "coherence": coherence(unit.structure_tensor)}
        for probe in unit.probes:
            name = next(k for k, v in PROBE_METHOD.items() if v == probe.method)
            row[name] = probe.angleH
            row[name + "_error"] = abs(probe.angleH - best.angleH)
            row[name + "_excess"] = ((probe.cost - best.cost) / best.cost
                                     if probe.cost < REJECTED_COST and best.cost else math.nan)
        rows.append(row)
    return rows


def _binned(rows, key, edges, value):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r[value] for r in rows if lo <= r[key] < hi and math.isfinite(r[value])]
        out.append((lo, hi, len(sel), float(np.median(sel)) if sel else math.nan))
    return out


def run_accuracy(args) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.info:
        rows += accuracy_rows(pinfo.load(path), None if args.channel == "all" else args.channel, path)
    if not rows:
        print("no probed units found (encode the reference with --probe-st-estimators at level full)")
        return
    _write_csv(out / "accuracy_units.csv", rows)

    coherence_edges = np.array(args.coherence_bins)
    theta_edges = np.array(args.theta_bins + [1e9])
    for r in rows:
        r["abs_theta_star"] = abs(r["theta_star"])

    lines = [f"accuracy: {len(rows)} coded units from {len(args.info)} file(s), channel {args.channel}"]
    lines.append(f"  {'estimator':<16}{'median err (deg)':>18}{'mean err':>10}{'median excess':>15}{'p90 excess':>12}")
    for name in ESTIMATORS:
        err = np.array([r[name + "_error"] for r in rows])
        exc = np.array([r[name + "_excess"] for r in rows])
        exc = exc[np.isfinite(exc)]
        lines.append(f"  {LABEL[name]:<16}{np.median(err):>18.3f}{err.mean():>10.3f}"
                     f"{np.median(exc):>15.5f}{np.percentile(exc, 90):>12.5f}")
    for key, edges, title in (("coherence", coherence_edges, "coherence"),
                              ("abs_theta_star", theta_edges, "|theta*| (deg)")):
        lines.append(f"\nby {title}: units | median angular error | median relative RD excess")
        for name in ESTIMATORS:
            errs = _binned(rows, key, edges, name + "_error")
            excs = _binned(rows, key, edges, name + "_excess")
            cells = [f"[{lo:g},{hi:g}): n={n} {e:.2f}deg {x:.4f}" for (lo, hi, n, e), (_, _, _, x) in zip(errs, excs)]
            lines.append(f"  {LABEL[name]}: " + "; ".join(cells))
    summary = "\n".join(lines)
    (out / "accuracy_summary.txt").write_text(summary + "\n")
    print(summary)

    plt = _pyplot()
    if plt is None:
        return
    centers = (coherence_edges[:-1] + coherence_edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, marker in zip(ESTIMATORS, ("o", "s", "^")):
        errs = [e for _, _, _, e in _binned(rows, "coherence", coherence_edges, name + "_error")]
        excs = [x for _, _, _, x in _binned(rows, "coherence", coherence_edges, name + "_excess")]
        axes[0].plot(centers, errs, marker=marker, label=LABEL[name])
        axes[1].plot(centers, excs, marker=marker, label=LABEL[name])
    axes[0].set_ylabel("median |theta_hat - theta*| (degrees)")
    axes[1].set_ylabel("median relative RD excess")
    for ax in axes:
        ax.set_xlabel("coherence")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_coherence.pdf")
    fig.savefig(out / "accuracy_vs_coherence.png", dpi=150)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def run_encodes(args) -> None:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ["-d", str(Path(args.light_field).resolve()) + "/",
              "-l", *map(str, args.max_partition), "-m", *map(str, args.min_partition),
              "-v", *map(str, args.views), "-r", *map(str, args.disparity_range),
              "--extension-repeat", "--bt601", "--pre-slant-tan", str(args.pre_slant)]
    configs = []
    if args.reference:
        configs.append(("reference", ["--refine-grid-search", *map(str, args.grid), "--probe-st-estimators"]))
    if args.reference and args.timing_reference:
        configs.append(("reference_timing", ["--refine-grid-search", *map(str, args.grid), "--partition-info", "winner"]))
    for name in args.estimators:
        configs.append((name, ["--refine-structure-tensor", *map(str, args.st_refine),
                               "--st-estimator", ENCODER_FLAG[name], "--partition-info", "winner"]))
    for lam in args.lambdas:
        for name, extra in configs:
            run_dir = out / f"{name}_{lam:g}"
            if (run_dir / "run.json").exists():
                print(f"skip {run_dir.name} (done)")
                continue
            if run_dir.exists():
                for f in run_dir.iterdir():
                    f.unlink()
            cmd = [args.encoder, *common, "--lambda", str(lam), "-o", str(run_dir / "lf.comp"), *extra]
            print(f"run {run_dir.name}", flush=True)
            start = time.time()
            with open(out / f"{name}_{lam:g}.log", "w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            if result.returncode != 0:
                raise SystemExit(f"{run_dir.name} failed, see {out / (run_dir.name + '.log')}")
            (run_dir / "run.json").write_text(json.dumps(
                {"name": name, "lambda": lam, "seconds": elapsed, "command": cmd}, indent=1))


# --------------------------------------------------------------------------- #
# bdrate
# --------------------------------------------------------------------------- #


def rd_point(run_dir: Path) -> dict:
    info = pinfo.load(str(run_dir / "info.json"))
    run = json.loads((run_dir / "run.json").read_text())
    size = info.metadata["lightFieldSize"]
    samples = float(np.prod(size[:4]))
    bits = sum(p.bits for p in info.partitions)
    psnr = {}
    for ch in (0, 1, 2):
        sse = sum(u.sse for p in info.partitions if p.channel == ch for u in p.units)
        psnr[ch] = 10 * math.log10(1024.0 ** 2 / (sse / samples))
    file_bits = os.path.getsize(run_dir / "lf.comp") * 8
    return {"bpp": file_bits / samples, "coded_bpp": bits / samples,
            "psnr_y": psnr[0], "psnr_yuv": (6 * psnr[0] + psnr[1] + psnr[2]) / 8,
            "seconds": run["seconds"]}


def bd_rate(anchor_rate, anchor_psnr, test_rate, test_psnr) -> float:
    """Bjontegaard delta rate (%) with cubic fits of log10(rate) against PSNR."""
    degree = min(3, len(anchor_rate) - 1, len(test_rate) - 1)
    pa = np.polyfit(anchor_psnr, np.log10(anchor_rate), degree)
    pt = np.polyfit(test_psnr, np.log10(test_rate), degree)
    lo = max(min(anchor_psnr), min(test_psnr))
    hi = min(max(anchor_psnr), max(test_psnr))
    if lo >= hi:
        return math.nan
    ia, it = np.polyint(pa), np.polyint(pt)
    avg = ((np.polyval(it, hi) - np.polyval(it, lo)) - (np.polyval(ia, hi) - np.polyval(ia, lo))) / (hi - lo)
    return (10 ** avg - 1) * 100


def run_bdrate(args) -> None:
    root = Path(args.runs)
    points = {}
    for run_dir in sorted(p for p in root.iterdir() if (p / "run.json").exists()):
        run = json.loads((run_dir / "run.json").read_text())
        points.setdefault(run["name"], []).append(rd_point(run_dir))
    if "reference" not in points:
        raise SystemExit("no reference runs found")
    timing_ref = points.get("reference_timing", points["reference"])
    ref_time = sum(p["seconds"] for p in timing_ref)
    anchor = sorted(points["reference"], key=lambda p: p["bpp"])

    lines = [f"{root.name}: anchor = grid search with full refinement ({len(anchor)} points), "
             f"metric = PSNR-{args.metric.upper()}, rate = file bits per sample",
             f"timing reference: {'reference_timing' if 'reference_timing' in points else 'reference (includes probe overhead)'}"]
    lines.append(f"  {'method':<16}{'BD-rate (%)':>12}{'rel. time':>11}")
    key = "psnr_" + args.metric
    rows = []
    for name in [n for n in ("pooled", "per_direction", "eigen4d", "legacy") if n in points]:
        test = sorted(points[name], key=lambda p: p["bpp"])
        bd = bd_rate([p["bpp"] for p in anchor], [p[key] for p in anchor],
                     [p["bpp"] for p in test], [p[key] for p in test])
        rel_time = sum(p["seconds"] for p in test) / ref_time
        lines.append(f"  {LABEL[name]:<16}{bd:>12.2f}{rel_time:>11.3f}")
        rows.append({"light_field": root.name, "method": name, "bd_rate": bd, "rel_time": rel_time})
    lines.append("\nRD points (bpp, PSNR-Y, PSNR-YUV, seconds):")
    for name, pts in points.items():
        for p in sorted(pts, key=lambda p: p["bpp"]):
            lines.append(f"  {name:<18}{p['bpp']:.5f}  {p['psnr_y']:.3f}  {p['psnr_yuv']:.3f}  {p['seconds']:.0f}")
    summary = "\n".join(lines)
    (root / "bdrate_summary.txt").write_text(summary + "\n")
    _write_csv(root / "bdrate.csv", rows)
    print(summary)


# --------------------------------------------------------------------------- #
# helpers and CLI
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available, skipping figure")
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthetic", help="controlled experiment on synthetic Lambertian light fields")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--d-min", type=float, default=-3.0)
    p.add_argument("--d-max", type=float, default=3.0)
    p.add_argument("--d-step", type=float, default=0.25)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--noise", type=float, default=2.0, help="noise sigma (texture std is 100)")
    p.add_argument("--cutoff", type=float, default=0.15, help="texture low-pass sigma, cycles/pixel")
    p.add_argument("--size", type=int, default=128, help="spatial size of the synthetic views")
    p.add_argument("--block", type=int, default=64, help="spatial size of the analysed block")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("validate", help="check this script's estimators and gradients against the codec")
    p.add_argument("info")
    p.add_argument("--light-field", help="directory of the encoded views, to also check gradients")
    p.add_argument("--channel", default="Y")

    p = sub.add_parser("accuracy", help="estimator accuracy against the grid-search optimum")
    p.add_argument("info", nargs="+", help="info.json files of reference encodes with probes")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--channel", default="Y", help="Y, Cb, Cr or all")
    p.add_argument("--coherence-bins", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001])
    p.add_argument("--theta-bins", type=float, nargs="+", default=[0, 1, 2, 5, 10, 20, 45])

    p = sub.add_parser("run", help="run the encodes for one light field")
    p.add_argument("--light-field", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default=str(Path(__file__).resolve().parents[1] / "build/new/bin/MSGTEncoder"))
    p.add_argument("--lambdas", type=float, nargs="+", default=[27, 672, 10140, 117590])
    p.add_argument("--estimators", nargs="*", default=list(ESTIMATORS), choices=list(ENCODER_FLAG),
                   help="structure-tensor searches to run (empty for none)")
    p.add_argument("--no-reference", dest="reference", action="store_false",
                   help="skip the grid-search reference runs (e.g. to split a long job)")
    p.add_argument("--grid", type=float, nargs=3, default=[1, 0.9, 0.1], help="refine-grid-search initStep range step")
    p.add_argument("--st-refine", type=float, nargs=2, default=[10, 0.5], help="refine-structure-tensor range step")
    p.add_argument("--max-partition", type=int, nargs=4, default=[9, 9, 64, 64])
    p.add_argument("--min-partition", type=int, nargs=4, default=[4, 4, 4, 4])
    p.add_argument("--views", type=int, nargs=2, default=[9, 9])
    p.add_argument("--disparity-range", type=float, nargs=2, default=[-3.1, 3.5])
    p.add_argument("--pre-slant", type=int, default=0)
    p.add_argument("--no-timing-reference", dest="timing_reference", action="store_false",
                   help="skip the extra un-probed reference run; relative times then include probe overhead")

    p = sub.add_parser("bdrate", help="BD-rate and relative time per estimator")
    p.add_argument("runs", help="output directory of the run command")
    p.add_argument("--metric", default="yuv", choices=["y", "yuv"])

    args = parser.parse_args(argv)
    if args.command == "synthetic":
        run_synthetic(args)
    elif args.command == "validate":
        return run_validate(args)
    elif args.command == "accuracy":
        run_accuracy(args)
    elif args.command == "run":
        run_encodes(args)
    elif args.command == "bdrate":
        run_bdrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
