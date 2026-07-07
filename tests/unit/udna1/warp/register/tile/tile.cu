#include "tile.cuh"

#ifdef TEST_WARP_REGISTER_TILE

void warp::reg::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_MMA
    warp::reg::tile::mma::tests(results);
#endif
}

#endif
