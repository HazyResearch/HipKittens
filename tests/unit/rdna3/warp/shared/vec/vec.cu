#include "vec.cuh"

#ifdef TEST_WARP_SHARED_VEC

void warp::shared::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS
    warp::shared::vec::conversions::tests(results);
#endif
}

#endif