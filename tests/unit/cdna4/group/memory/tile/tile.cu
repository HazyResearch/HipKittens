#include "tile.cuh"

#ifdef TEST_GROUP_MEMORY_TILE

void group::memory::tile::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
    group::memory::tile::global_to_shared::tests(results);
#endif
}

#endif