# mule-sgt

A 4D light-field codec. A light field (a 2D grid of `T×S` views, each a `V×U` image)
is compressed as a hierarchical partition tree of 4D blocks: each block is either
coded directly, split spatially ("intra-view", along the `V,U` pixel axes), or split
angularly ("inter-view", along the `T,S` view axes), with the winning choice picked by
rate-distortion optimization. Leaf blocks are decorrelated with a slant/generalized
transform (SGT) and entropy-coded with an adaptive binary arithmetic coder.

Two command-line tools are built: **`MSGTEncoder`** (light field → compressed
`.comp` bitstream) and **`MSGTDecoder`** (`.comp` bitstream → PPM views).

## Table of contents

- [How to build](#how-to-build)
  - [Dependencies](#dependencies)
  - [Build steps](#build-steps)
  - [Known machine-specific caveats](#known-machine-specific-caveats)
- [Quick start](#quick-start)
- [Encoding: `MSGTEncoder`](#encoding-msgtencoder)
  - [Command-line flags vs. config-file syntax](#command-line-flags-vs-config-file-syntax-encoder)
  - [Flag reference](#encoder-flag-reference)
  - [Config-file token reference](#encoder-config-file-token-reference)
- [Decoding: `MSGTDecoder`](#decoding-msgtdecoder)
  - [Flag reference](#decoder-flag-reference)
  - [Config-file token reference](#decoder-config-file-token-reference)
- [What must match between encode and decode](#what-must-match-between-encode-and-decode)
- [Example config files](#example-config-files)
- [Repository layout](#repository-layout)
- [Releases / CI](#releases--ci)

## How to build

### Dependencies

The build is C++20 and uses CMake with the Ninja generator. It links against:

| Dependency | Used for | Notes |
|---|---|---|
| CMake ≥ 3.19 | build system | |
| Ninja | build generator | |
| A C++20 compiler | | Tested with GCC 10.4 (conda-forge `gxx_linux-64`). |
| [LibTorch](https://pytorch.org/) (PyTorch's C++ library) | tensor storage/ops throughout `LightField`, `Encoder`, `Decoder` | Found via `find_package(Torch REQUIRED)`; the CMake `CMAKE_PREFIX_PATH` must point at the `torch/share/cmake` directory of a PyTorch install (CPU build is sufficient — nothing here uses CUDA). The easiest way to get this is `pip install torch` or `conda install pytorch-cpu`. |
| LAPACK | linear algebra (`find_package(LAPACK REQUIRED)`) | e.g. `liblapack-dev` on Debian/Ubuntu, or conda-forge `lapack`. |
| [OSQP](https://osqp.org/) | quadratic-programming solver used by `LightField` | No apt package on Debian/Ubuntu; use conda-forge (`osqp`/`libosqp`) or build from source. |
| Boost | `program_options` (CLI parsing), `iostreams`, `filesystem`, header-only components | conda-forge's `boost-cpp`/`libboost-devel`, or `libboost-program-options-dev libboost-iostreams-dev libboost-filesystem-dev` on Debian/Ubuntu. |
| OpenMP | parallel RD-cost search in `Encoder` | Usually ships with the compiler (`libgomp`). |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON output (`info.json` per encode run) | Fetched automatically by CMake `FetchContent` — no manual install needed. |
| [GoogleTest](https://github.com/google/googletest) | unit tests under `tests/` | Fetched automatically by CMake `FetchContent` — no manual install needed. |

This exact set of C++ dependencies (LibTorch, LAPACK, OSQP, Boost, plus a matching
GCC 10 toolchain) is readily available as a single conda-forge environment. A
known-working recipe:

```bash
conda create -n mule-sgt -c conda-forge \
    python=3.10 pytorch-cpu lapack osqp libboost-devel boost-cpp \
    cxx-compiler gxx_linux-64 gcc_linux-64 cmake ninja
conda activate mule-sgt
```

(OpenCV was previously a dependency, used only to write optional debug PNGs from
`Block4D_::write_tensor`; it has been removed — that function is now a no-op, and
none of the executables link OpenCV any more.)

### Build steps

The repo ships CMake presets (`CMakePresets.json` / `CMakeUserPresets.json`):

```bash
cmake --preset default-profile
cmake --build --preset default-profile-build
```

`CMakeUserPresets.json` currently hardcodes an absolute compiler path from the
original development machine
(`/nfs/home/.../miniconda3/envs/slantKLT/bin/x86_64-conda-linux-gnu-{gcc,c++}`) —
**edit `CMakeUserPresets.json` to point at your own compiler** (or delete the
`cacheVariables` compiler overrides and let CMake pick up `CC`/`CXX` from your
environment) before using the preset. Alternatively, configure directly without the
user preset:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.dirname(os.path.dirname(torch.__file__))+"/torch/share/cmake")')"
cmake --build build
```

Either way, the two executables land at `<build-dir>/bin/MSGTEncoder` and
`<build-dir>/bin/MSGTDecoder` (`build/new/bin/...` when using the preset above).

The first CMake configure also copies `01_TemplateFolder/` into `results/` at the
repo root (see `CMakeLists.txt`) — this only happens once per build directory.

### Known machine-specific caveats

- Presets aside, nothing in the build depends on a specific filesystem layout beyond
  the compiler-path point above — dependency discovery otherwise goes through normal
  CMake `find_package` calls.

## Quick start

Using one of the shipped example config pairs (adjust `-d`/`-i`/`-o` paths for your
light field and where you want output written):

```bash
ENCODER=build/new/bin/MSGTEncoder
DECODER=build/new/bin/MSGTDecoder

$ENCODER -c 02_ResultsTemplate/Greek/GridSearchTight/GREEK_0.02_encode.conf
$DECODER -c 02_ResultsTemplate/Greek/GridSearchTight/GREEK_0.02_decode.conf
```

The encoder reads a directory of `TTT_SSS.ppm` view images (one per `T,S` view
position — see `-d`), writes one `.comp` bitstream, and also writes an `info.json`
next to it describing the chosen partition tree. The decoder reads that `.comp` file
and writes one `U_V.ppm` per decoded view into its output directory.

Light fields are expected as directories of 16-bit PPM images named
`<row>_<col>.ppm` (e.g. `000_000.ppm`, `000_001.ppm`, ...), one file per view.

## Encoding: `MSGTEncoder`

### Command-line flags vs. config-file syntax (encoder)

`MSGTEncoder` accepts options two ways, and **they are not the same vocabulary**:

1. **Real command-line flags**, parsed by Boost `program_options`
   (`-h/--help`, `-l/--maximum-partition-size`, `--bt601`, etc. — GNU-style, `--` for
   long names, `-` for short aliases). Use these when invoking the binary directly.
2. **A config file**, passed via `-c`/`--config-file <path>` (on the real command
   line, or as the *only* thing you pass — e.g. `MSGTEncoder -c my.conf`). The file
   itself uses **its own, different set of single-dash tokens** (e.g. `-lambda`,
   `-preSlantTan`, `-extension_repeat` with an **underscore**, `-VV` instead of
   `-V`/`--verbosity`), read whitespace-separated from the file — see the
   [config-file token reference](#encoder-config-file-token-reference) below. Every
   example config under `01_TemplateFolder/` and `02_ResultsTemplate/` uses this
   config-file syntax exclusively.

Unrecognized config-file tokens are silently ignored (no warning, no error) — a typo
or the wrong dialect (e.g. writing `-extension-repeat` with a hyphen, which the
config-file parser does not recognize; it only accepts the underscore form
`-extension_repeat`) will not fail loudly, it will just leave that setting at its
default. Double-check flag spelling against the tables below if a setting doesn't
seem to take effect.

### Encoder flag reference

| Flag (`--long`, `-short`) | Args | Default | Meaning |
|---|---|---|---|
| `--help`, `-h` | — | | Print usage and exit. |
| `--config-file`, `-c` | 1 path | | Read all other options from a config file (see below) instead of the command line. |
| `--light-field-dir`, `-d` | 1 path | `./ExampleLightField/` | Directory containing the input light field's `TTT_SSS.ppm` view images. |
| `--output-dir`, `-o` | 1 path | `out.comp` | **Output file path** for the compressed bitstream (despite the flag name, this is a file, not a directory — if the target already exists, the encoder auto-suffixes the parent directory with `-1`, `-2`, ... to avoid overwriting). |
| `--lambda` | 1 float | `1.0` | The Lagrange multiplier λ for rate-distortion optimization (`cost = distortion + λ·rate`). Higher λ → fewer bits, lower quality; lower λ → more bits, higher quality. This is the primary rate control knob — see the example configs for λ values at several target rates. |
| `--maximum-partition-size`, `-l` | 4 ints (`t s v u`) | `4 4 4 4` | Size of the top-level 4D block the partition search starts from, in (angular-T, angular-S, spatial-V, spatial-U) order. |
| `--minimum-partition-size`, `-m` | 4 ints (`t s v u`) | `13 13 15 15` | Smallest block size the RD search is allowed to split down to, same axis order. Setting `min == max` on an axis pair disables splitting on that axis entirely (e.g. `-m 9 9 4 4` with `-l 9 9 64 64` allows spatial splits but no angular/inter-view split). |
| `--disp-range`, `-r` | 2 floats (`lo hi`) | `-3.5 3.5` | Expected disparity range of the light field, used to bound the slant-angle search. |
| `--pre-slant-tan` | 1 int | `0` | Tangent of the global pre-slant shear applied to the light field before encoding (for lenslet/plenoptic content with non-axis-aligned epipolar structure). `0` disables pre-slanting. |
| `--transform-gain`, `-g` | 1 float | `1` | Scalar gain folded into the transform (and into `--lambda`, internally). Leave at the default unless you know you need to change it — it must match what the decoder is told via `--t_gain`. |
| `--num-views`, `-v` | 2 ints (`t s`) | `13 13` | Number of views to read from the light field directory, in each angular axis. |
| `--view-offset`, `-b` | 2 ints (`t s`) | `0 0` | Index of the first view to read (lets you encode a sub-grid of a larger light field). |
| `--isLenslet13x13` | flag | off | Applies a symmetric brightness correction to the edge sub-aperture views, for 13×13 plenoptic-camera captures. Must be matched by `--lenslet13x13` on the decoder. |
| `--extension-repeat` | flag | on (this is the built-in default) | Pad partial blocks at the light-field boundary by repeating the last row/column. |
| `--extension-cyclic` | flag | off | Pad partial boundary blocks cyclically (wrap around). |
| `--extension-none` | flag | off | Do not pad; boundary blocks are used at their natural (possibly non-power-of-two) size. |
| `--bt601` | flag | on (default) | Use BT.601 RGB→YCbCr color transform. Mutually exclusive with `--ycocg` (specifying both is an error). |
| `--ycocg` | flag | off | Use YCoCg color transform instead of BT.601. |
| `--verbosity`, `-V` | flag | off | Print detailed per-block progress/diagnostics to stdout while encoding. |

**Rate-distortion search method** — exactly one of the following may be given (giving
more than one raises `std::logic_error: Conflicting options: Multiple search methods
specified.` and aborts the process); if none is given, the default is
`REFINE_STRUCTURE_TENSOR`:

| Flag | Args | Meaning |
|---|---|---|
| `--structure-tensor` | flag | Estimate the optimal transform angle from the local structure tensor (fast, no search). |
| `--logdet [angleStep]` | flag, optional 1 float | Search transform angles on a grid, minimizing log-determinant of the model covariance; optional step size (default `1.0`). |
| `--grid-search [angleStep]` | flag, optional 1 float | Exhaustive grid search over candidate transform angles by actual RD cost; optional step size (default `1.0`). This is the method used by every example config in this repo. |
| `--covariance` | flag | Estimate the transform directly from the empirical covariance. |
| `--all-heuristics` | flag | Try every non-refining heuristic above and keep the best by RD cost. |
| `--zero` | flag | Force a zero transform angle (no slant) — useful as a baseline/ablation. |
| `--refine-structure-tensor [range step]` | flag, optional 2 floats | Structure-tensor estimate, then a local refinement search around it; optional `range`/`step` (defaults `10.0`/`0.5`). |
| `--refine-grid-search [initStep range step]` | flag, optional 3 floats | Grid search with a coarse-to-fine refinement pass; optional `initStep`/`range`/`step` (defaults `1.0`/`0.9`/`0.1`). |

### Encoder config-file token reference

One token (and its arguments) at a time, whitespace/newline-separated. This is a
**different vocabulary** from the CLI flags above — see the note in the previous
section.

| Token | Args | Sets |
|---|---|---|
| `-d` | 1 path | Light field directory |
| `-o` | 1 path | Output `.comp` file path |
| `-lambda` | 1 float | λ |
| `-l` | 4 ints | Maximum partition size (`t s v u`) |
| `-m` | 4 ints | Minimum partition size (`t s v u`) |
| `-t` / `-s` / `-v` / `-u` | 1 int each | Maximum partition size for just that one axis (`t`,`s`,`v`,`u` respectively) — note `-v` here means the angular-S maximum size, **not** number of views (see `-nv`/`-nh` below) |
| `-min_t` / `-min_s` / `-min_v` / `-min_u` | 1 int each | Minimum partition size for just that one axis |
| `-r` | 2 floats | Disparity range |
| `-preSlantTan` | 1 int | Pre-slant tangent |
| `-t_gain` | 1 float | Transform gain |
| `-nv` | 1 int | Number of views, T axis |
| `-nh` | 1 int | Number of views, S axis |
| `-off_v` | 1 int | First view index, T axis |
| `-off_h` | 1 int | First view index, S axis |
| `-lenslet13x13` | — | Enable 13×13 lenslet edge-view correction |
| `-extension_repeat` | — | Boundary extension: repeat |
| `-extension_cyclic` | — | Boundary extension: cyclic |
| `-extension_none` | — | Boundary extension: none |
| `-bt601` | — | BT.601 color transform |
| `-ycocg` | — | YCoCg color transform |
| `-VV` | — | Verbose output |
| `-structure_tensor` | — | Search method: structure tensor |
| `-logdet [angleStep]` | optional 1 float | Search method: logdet |
| `-grid_search [angleStep]` | optional 1 float | Search method: grid search |
| `-covariance` | — | Search method: covariance |
| `-all_heuristics` | — | Search method: all heuristics |
| `-zero` | — | Search method: zero angle |
| `-refine_structure_tensor [range step]` | optional 2 floats | Search method: refine structure tensor |
| `-refine_grid_search [initStep range step]` | optional 3 floats | Search method: refine grid search |

Hyphens and underscores are **not** interchangeable in this file — e.g. `-grid_search`
and `-grid-search` are both accepted for that particular token, but
`-extension_repeat` (accepted) and `-extension-repeat` (silently ignored, not
accepted) are not; when unsure, copy the form used in an
[example config](#example-config-files).

## Decoding: `MSGTDecoder`

`MSGTDecoder` takes the same two-syntax approach (real CLI flags vs. a `-c` config
file with its own token vocabulary) as the encoder.

### Decoder flag reference

| Flag (`--long`, `-short`) | Args | Default | Meaning |
|---|---|---|---|
| `--help`, `-h` | — | | Print usage and exit. |
| `--config-file`, `-c` | 1 path | | Read options from a config file instead of the command line. |
| `--input-file`, `-i` | 1 path | | Path to the `.comp` bitstream to decode. |
| `--output-dir`, `-o` | 1 path | | Directory to write decoded `U_V.ppm` view images into. |
| `--num-views`, `-v` | 2 ints | | Accepted for symmetry with the encoder but **currently has no effect on decoding** — the light field's actual size is always read from the bitstream header. |
| `--view-offset`, `-b` | 2 ints | | Offsets used only when naming the output `U_V.ppm` files — set to match the encoder's `--view-offset` so filenames line up with the originals; does not affect pixel reconstruction. |
| `--view-stride`, `-s` | 2 ints | `1 1` | Stride used only when naming output files (`index*stride + offset`) — leave at `1 1` unless you have a specific renaming need. |
| `--lenslet13x13` | flag | off | **Must match** the encoder's `--isLenslet13x13` (same underlying correction applied in reverse). |
| `--extension-repeat` / `--extension-cyclic` / `--extension-none` | flag | *(none — see warning)* | **Must match** whichever the encoder used; controls how boundary blocks are reconstructed. |
| `--bt601` / `--ycocg` | flag | *(none — see warning)* | **Must match** the encoder's color-transform choice; not stored in the bitstream. |
| `--t_gain` | 1 float | *(none — see warning)* | **Must match** the encoder's `--transform-gain`/`-g`. |
| `--verbosity`, `-V` | flag | off | Verbose per-block decode diagnostics. |

> **Always pass an extension flag, a color-transform flag, and `--t_gain` explicitly.**
> Unlike the encoder, `DecoderParameters` has no compiled-in default for
> `extensionMethod` or `transformGain` — if you omit them (and they're not set by a
> config file either), they're left uninitialized. `--lenslet13x13`, `--verbosity`,
> and `--bt601`/`--ycocg` (which does default to BT.601) are safe to omit since those
> are always explicitly assigned from their (defaulted) switches.

### Decoder config-file token reference

| Token | Args | Sets |
|---|---|---|
| `-i` | 1 path | Input `.comp` file |
| `-o` | 1 path | Output directory |
| `-nv` | 1 int | Number of views, T axis (parsed, unused — see above) |
| `-nh` | 1 int | Number of views, S axis (parsed, unused — see above) |
| `-off_v` | 1 int | View-offset, T axis (output filenames only) |
| `-off_h` | 1 int | View-offset, S axis (output filenames only) |
| `-s_v` | 1 int | View-stride, T axis (output filenames only) |
| `-s_h` | 1 int | View-stride, S axis (output filenames only) |
| `-lenslet13x13` | — | Enable 13×13 lenslet correction (must match encoder) |
| `-extension_repeat` / `-extension_cyclic` / `-extension_none` | — | Boundary extension (must match encoder) |
| `-t_gain` | 1 float | Transform gain (must match encoder) |
| `-bt601` / `-ycocg` | — | Color transform (must match encoder) |
| `-VV` | — | Verbose output |

Unlike the encoder's config parser, the decoder's tokens above are used consistently
(hyphen vs. underscore) across every shipped example config — copy them as-is.

## What must match between encode and decode

The bitstream header is self-describing for some settings but not others.

**Read automatically from the bitstream header** (no decoder flag exists for these —
they cannot be overridden even if you tried): superior bit-plane, maximum partition
size (`-l`), total light-field size, disparity range (`-r`), pre-slant tangent.

**Must be supplied again, identically, on the decoder** because they are *not* in the
header and have no safe default:

- Boundary extension method (`--extension-repeat/-cyclic/-none`)
- Transform gain (`--t_gain`, matching the encoder's `-g/--transform-gain`)
- Color transform (`--bt601`/`--ycocg`)
- Lenslet 13×13 correction (`--lenslet13x13`, matching the encoder's `--isLenslet13x13`)

Getting any of these wrong will not necessarily produce an error — it can silently
decode to visibly wrong colors or brightness. See
[`02_ResultsTemplate/Bikes/AllHeuristics`](02_ResultsTemplate/Bikes/AllHeuristics)
for an example where `-lenslet13x13` and view offsets are kept consistent between the
encode and decode configs.

Flags that only affect output file naming/location, not reconstruction correctness:
`--input-file`/`-i` (decoder), `--output-dir`/`-o` (both — but note it means a *file*
path for the encoder and a *directory* for the decoder), `--view-offset`/`-b` and
`--view-stride`/`-s` (decoder).

## Example config files

| Pair | What it demonstrates |
|---|---|
| [`02_ResultsTemplate/Greek/GridSearchTight/GREEK_0.02_encode.conf`](02_ResultsTemplate/Greek/GridSearchTight/GREEK_0.02_encode.conf) + [`..._decode.conf`](02_ResultsTemplate/Greek/GridSearchTight/GREEK_0.02_decode.conf) | Baseline: grid-search RD method, `-m 9 9 4 4` disables angular splitting (`min t,s == 9 == num views`), only spatial splitting from 64×64 down to 4×4. |
| [`02_ResultsTemplate/Greek/GridSearchTightViewSplit/GREEK_0.02_encode.conf`](02_ResultsTemplate/Greek/GridSearchTightViewSplit/GREEK_0.02_encode.conf) + [`..._decode.conf`](02_ResultsTemplate/Greek/GridSearchTightViewSplit/GREEK_0.02_decode.conf) | Same as above but `-m 4 4 4 4` — angular (inter-view) splitting is now allowed too. The decode config is unchanged, since partition sizes are read from the header. |
| [`02_ResultsTemplate/Bikes/AllHeuristics/BIKES_0.02_encode.conf`](02_ResultsTemplate/Bikes/AllHeuristics/BIKES_0.02_encode.conf) + [`..._decode.conf`](02_ResultsTemplate/Bikes/AllHeuristics/BIKES_0.02_decode.conf) | A 13×13 lenslet capture: `-lenslet13x13`, non-zero `-off_v`/`-off_h`, and `-all_heuristics` search, with the lenslet flag and view offsets mirrored on the decode side. |
| [`01_TemplateFolder/Greek/GREEK_0.02_encode.conf`](01_TemplateFolder/Greek/GREEK_0.02_encode.conf) + [`run_decoder.sh`](01_TemplateFolder/Greek/run_decoder.sh) | Encoding via a config file, decoding via real CLI flags (`$DECODER -v 9 9 -b 0 0 --extension-repeat --t_gain 1 --bt601 -V ...`) — showing both syntaxes side by side. |

Each `02_ResultsTemplate/<LightField>/<Sweep>/` directory also has a matching
[`run_experiments.sh`](02_ResultsTemplate/Greek/GridSearchTight/run_experiments.sh)
that runs the encoder then the decoder across a set of rate points (`0.005`, `0.02`,
`0.1`, `0.75`), and a sibling under `02_ResultsTemplate/<LightField>/eval/<Sweep>/`
with a `metrics_QM.py` script for computing PSNR/SSIM against the original views.
`.conf` files across a sweep only differ in `--lambda`/`-lambda` (and, for
`GridSearchTightViewSplit`, `-m`) — compare a whole sweep's files to see how λ maps to
rate/quality for a given light field.

## Repository layout

```
src/
  MSGTEncoder/    Encoder CLI entry point (MSGTEncoder.cpp) and argument parsing
  MSGTDecoder/    Decoder CLI entry point (MSGTDecoder.cpp) and argument parsing
  Encoder/        RD-optimized partition search + bitstream writing (TransformPartition, Hierarchical4DEncoder)
  Decoder/        Partition-tree and bitstream reading (PartitionDecoder, Hierarchical4DDecoder, ABADecoder)
  LightField/     4D light-field/block representation, SGT transform, PPM I/O, pre-slant/padding
  ProbabilityModel/  Adaptive binary probability model used by the arithmetic coder
  IO/             Low-level file/collection I/O helpers
  DebugTools/     JSON reporting for per-run partition info (info.json)
  OldDCT/         Legacy transform code retained for comparison
01_TemplateFolder/    Template experiment layout, copied into results/ on first CMake configure
02_ResultsTemplate/   Per-light-field experiment sweeps (config files + eval scripts)
tests/                GoogleTest unit tests (fetched automatically)
```

## Releases / CI

Pushing a tag matching `v*` (e.g. `v1.0.0`) triggers
[`.github/workflows/release.yml`](.github/workflows/release.yml), which builds
`MSGTEncoder`/`MSGTDecoder` in Release mode and publishes them as a GitHub Release
attached to that tag.

The release archive is a portable, extract-and-run bundle, produced by
[`scripts/package_release.sh`](scripts/package_release.sh):

```
mule-sgt-<version>-linux-x86_64/
├── bin/
│   ├── MSGTEncoder
│   └── MSGTDecoder
├── lib/           # every shared library the two binaries need, resolved via ldd
│                   # (plus Intel MKL's runtime-dispatched CPU backends, which
│                   # ldd can't see, since they're dlopen()'d rather than linked)
└── README.md
```

The two binaries have their `RPATH` rewritten (via `patchelf`) to `$ORIGIN/../lib`,
so run them directly from wherever the archive is extracted — no `LD_LIBRARY_PATH`,
conda environment, or install step needed:

```bash
tar xzf mule-sgt-v1.0.0-linux-x86_64.tar.gz
cd mule-sgt-v1.0.0-linux-x86_64
./bin/MSGTEncoder --help
```

This still assumes a Linux x86-64 target with a glibc version at least as new as the
build machine's (the standard native-binary portability floor — glibc itself is
deliberately *not* bundled, since doing so is fragile and can break more than it
fixes).
