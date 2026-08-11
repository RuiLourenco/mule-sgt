#!/usr/bin/env bash
# Packages MSGTEncoder/MSGTDecoder into a self-contained, extract-and-run bundle:
#   <staging>/bin/MSGTEncoder
#   <staging>/bin/MSGTDecoder
#   <staging>/lib/*.so           (every shared library they resolve to via ldd,
#                                 plus Intel MKL's full set of runtime-dispatched
#                                 CPU backends, which ldd can't see)
#   <staging>/README.md
#
# The two binaries have their RPATH rewritten (via patchelf) to a path relative to
# their own location ($ORIGIN/../lib), so the bundle runs on another machine without
# needing the build machine's conda env or system libraries on LD_LIBRARY_PATH.
#
# Usage: scripts/package_release.sh <build-bin-dir> <staging-dir>
#   e.g.  scripts/package_release.sh build/new/bin dist/mule-sgt-v1.0.0-linux-x86_64
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <build-bin-dir> <staging-dir>" >&2
    exit 1
fi

BUILD_BIN_DIR="$1"
STAGE_DIR="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v patchelf >/dev/null 2>&1; then
    echo "Error: patchelf not found on PATH. Install it into the build environment, e.g.:" >&2
    echo "  conda install -c conda-forge patchelf" >&2
    exit 1
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bin" "$STAGE_DIR/lib"

cp "$BUILD_BIN_DIR/MSGTEncoder" "$BUILD_BIN_DIR/MSGTDecoder" "$STAGE_DIR/bin/"
cp "$REPO_ROOT/README.md" "$STAGE_DIR/"

# Collect every unique resolved shared-library path from both binaries, excluding
# the dynamic linker itself and the vDSO, which are provided by the target kernel
# and must not be bundled.
mapfile -t DEPS < <(
    ldd "$STAGE_DIR/bin/MSGTEncoder" "$STAGE_DIR/bin/MSGTDecoder" \
        | awk '{print $3}' \
        | grep '^/' \
        | grep -v '^/lib64/ld-linux' \
        | sort -u
)

if [ "${#DEPS[@]}" -eq 0 ]; then
    echo "Error: ldd resolved no library paths — is $BUILD_BIN_DIR/MSGTEncoder a valid dynamically-linked binary?" >&2
    exit 1
fi

for dep in "${DEPS[@]}"; do
    cp -L "$dep" "$STAGE_DIR/lib/"
done

# Intel MKL (pulled in transitively by LibTorch) dlopen()s a CPU-dispatched compute
# backend (e.g. libmkl_avx512.so.2) and, for vector math, a matching libmkl_vml_*
# backend, chosen at runtime based on the machine it's actually running on, not the
# one it was built on — ldd can't see this, since it's not a direct ELF dependency.
# Bundle every dispatch backend (but not MKL's other unused interface variants —
# BLACS/ScaLAPACK for MPI, ILP64, gfortran, PGI/TBB/sequential threading — none of
# which apply here, since ldd already told us the actually-linked interface/threading
# libraries) so the bundle works regardless of the target CPU's feature set.
mkl_dep="$(printf '%s\n' "${DEPS[@]}" | grep '/libmkl_' | head -1 || true)"
if [ -n "$mkl_dep" ]; then
    mkl_dir="$(dirname "$mkl_dep")"
    cp -L "$mkl_dir"/libmkl_def.so* "$mkl_dir"/libmkl_avx*.so* "$mkl_dir"/libmkl_mc*.so* "$mkl_dir"/libmkl_vml_*.so* "$STAGE_DIR/lib/" 2>/dev/null || true
fi

patchelf --set-rpath '$ORIGIN/../lib' "$STAGE_DIR/bin/MSGTEncoder"
patchelf --set-rpath '$ORIGIN/../lib' "$STAGE_DIR/bin/MSGTDecoder"

echo "Packaged $(du -sh "$STAGE_DIR" | cut -f1) into $STAGE_DIR"
echo "Verifying no remaining unresolved dependencies..."
for bin in MSGTEncoder MSGTDecoder; do
    if ldd "$STAGE_DIR/bin/$bin" | grep -q "not found"; then
        echo "Error: $bin has unresolved dependencies after patching:" >&2
        ldd "$STAGE_DIR/bin/$bin" | grep "not found" >&2
        exit 1
    fi
done
echo "OK: both binaries resolve all dependencies from $STAGE_DIR/lib"
