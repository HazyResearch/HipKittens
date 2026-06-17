#include "warp.cuh"

#ifdef TEST_WARP

using namespace warp;

void warp::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/warp tests!     ------------------------------\n"  << std::endl;
#ifdef TEST_WARP_MEMORY
    memory::tests(results);
#endif
}

#endif
