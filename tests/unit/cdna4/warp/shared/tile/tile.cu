#include "tile.cuh"

#ifdef TEST_WARP_SHARED_TILE

void warp::shared::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/shared/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS
    warp::shared::tile::conversions::tests(results);
#endif
}

#endif