#pragma once
#include <ATen/Parallel.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>

// Makes encoder and decoder output bit-identical across runs, machines and builds. Must be the first thing main does.
//  - MKL runs in strict conditional numerical reproducibility (CNR) mode on the AVX2 code path. Without it, MKL's
//    eigendecompositions and matrix products can differ in the last bits with memory alignment, which is enough to
//    flip near-tie RD decisions.
//  - Tensor ops run on one thread (MKL results also depend on the thread count); the RD search parallelises instead.
// The CNR mode can only be set before MKL's first call, so it comes before at::set_num_threads. MKL ships inside
// libtorch without headers, so its C API is looked up in the already loaded library (codes from mkl_service.h).
// strict = false (--no-strict) skips the CNR mode for CPUs without AVX2: output then still does not depend on the
// thread count, but can differ in the last bits between machines and builds.
inline void enableReproducibleMath(bool strict = true) {
    if (!strict) {
        std::cerr << "Warning: MKL strict reproducibility disabled (--no-strict); output may differ between machines" << std::endl;
        at::set_num_threads(1);
        return;
    }
    constexpr int MKL_CBWR_ALL = ~0, MKL_CBWR_AVX2 = 10, MKL_CBWR_STRICT = 0x10000;
    constexpr int mode = MKL_CBWR_AVX2 | MKL_CBWR_STRICT;
    using CbwrFn = int (*)(int);
    auto cbwrSet = reinterpret_cast<CbwrFn>(dlsym(RTLD_DEFAULT, "MKL_CBWR_Set"));
    auto cbwrGet = reinterpret_cast<CbwrFn>(dlsym(RTLD_DEFAULT, "MKL_CBWR_Get"));
    if (!cbwrSet || !cbwrGet) {
        std::cerr << "Error: MKL not found in libtorch; strict reproducibility cannot be enabled. Pass --no-strict to run without it." << std::endl;
        std::exit(1);
    }
    int status = cbwrSet(mode);
    if (status != 0 || cbwrGet(MKL_CBWR_ALL) != mode) {
        std::cerr << "Error: could not enable MKL strict reproducibility (MKL_CBWR_Set returned " << status
                  << "; it needs a CPU with AVX2). Pass --no-strict to run without it." << std::endl;
        std::exit(1);
    }
    at::set_num_threads(1);
}
