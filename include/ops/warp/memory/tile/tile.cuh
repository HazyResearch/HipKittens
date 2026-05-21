/**
 * @file
 * @brief An aggregate header of warp memory operations on tiles, where a single warp loads or stores data on its own.
 */

#pragma once

#include "shared_to_register.cuh"
#include "global_to_register.cuh"
#include "global_to_shared.cuh"

#include "assembly/tile.cuh"

#ifdef KITTENS_UDNA1
// gfx1250 hardware-accelerated transfer paths. Must come after
// global_to_shared.cuh because they share its padding descriptors and
// the `detail::subtile_flat` helper.
#include "tdm.cuh"
#include "async.cuh"
#endif
