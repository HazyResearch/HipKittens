#include "vec.cuh"

#ifdef TEST_GROUP_MEMORY_VEC

void group::memory::vec::tests(test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED
    group::memory::vec::global_to_shared::tests(results);
#endif
}

#endif