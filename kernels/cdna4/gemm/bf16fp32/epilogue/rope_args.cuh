#pragma once
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

// Shared RoPE checks. Dh must divide each of `parts` so every head starts at table column 0.
// cos_sin and c need 4-byte alignment: both are accessed as bf16x2.
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
