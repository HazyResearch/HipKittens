#include "tile.cuh"

#ifdef TEST_WARP_MEMORY_TILE

void warp::memory::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
    warp::memory::tile::global_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
    warp::memory::tile::global_to_shared::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
    warp::memory::tile::shared_to_register::tests(results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_FP4_LOAD
    warp::memory::tile::fp4_load::tests(results);
#endif
}

#endif