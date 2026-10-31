#pragma once
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

// Argument checks shared by the RoPE bindings (rope, rmsnorm_rope, rmsnorm_rope_qkv). n is the output width; the
// head dim Dh must divide every part in `parts` (N for one weight; Nq and Nkv for QKV) so each head starts at
// table column 0. The epilogue reads [cos, sin] as one bf16x2 word and stores rotated pairs as bf16x2, so the
// table and the output must be 4-byte aligned.
template<typename Globals>
void check_rope_args(const char* name, const Globals& g, int n, std::initializer_list<int> parts) {
    const std::string who(name);
    const int dh = g.cos_sin.cols();
    bool divides = dh > 0 && dh % 2 == 0;
    for (int p : parts) divides = divides && p % dh == 0;
    if (g.cos_sin.rows() != g.a.rows() || !divides)
        throw std::runtime_error(who + ": cos_sin must be [M, Dh] with Dh even and dividing N (Nq and Nkv for QKV); "
                                 "M = a.rows()");
    if (g.rope_cols < 0 || g.rope_cols > n || g.rope_cols % BLOCK_SIZE)
        throw std::runtime_error(who + ": rope_cols must be a multiple of 256 in [0, N]");
    if (reinterpret_cast<uintptr_t>(g.c.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error(who + ": output must have 4-byte alignment");
    if (reinterpret_cast<uintptr_t>(g.cos_sin.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error(who + ": cos_sin must have 4-byte alignment");
}
