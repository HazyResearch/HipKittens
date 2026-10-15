#include "aux_reduce.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RmsReduceGlobals g, hipStream_t stream) {
    const int M = g.r.cols();
    if (g.r.rows() != 1)
        throw std::runtime_error("rms_reduce: r must be [M]");
    if (g.partials.cols() != M || g.partials.rows() < 1)
        throw std::runtime_error("rms_reduce: partials must be [N/64, M] with M = r's length");
    constexpr int THREADS = 256;                             // 4 wavefronts per block, one row each
    const long long total = (long long)M * kittens::WARP_THREADS;
    rms_reduce<<<(int)((total + THREADS - 1) / THREADS), THREADS, 0, stream>>>(g.partials, g.r);
    hip_check(hipGetLastError(), "rms_reduce launch");
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "RMS reduce: r[m] = 1 / sqrt(sum(partials[:, m]) / N + 1e-5), partials = [N/64, M] (fp32, "
              "from tk_partialrms), r = [M] (fp32)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::fp32, tensor_dtype::fp32},
        &RmsReduceGlobals::partials, &RmsReduceGlobals::r);
}
