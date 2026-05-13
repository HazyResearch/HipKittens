// Timing result structure
struct TimingResult {
    float best_time_ms;
    float avg_time_ms;
    double best_tops;
    double avg_tops;
    int timing_iterations;
};

#define HipCheckError()    __hipCheckError( __FILE__, __LINE__ )
inline void __hipCheckError( const char *file, const int line ) {
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
    err = hipDeviceSynchronize();
    if( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
}

// Random initialization function — INT8 in [-3, 3] keeps the int32 accumulator
// well below 2^31 even for K = 8192 (max |Σ| ≤ 9 * K = 73728).
template <int M, int N, int K>
void random_init(std::vector<int8>& a_host, std::vector<int8>& b_host, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(-3, 3);
    for (int i = 0; i < M*K; i++) {
        a_host[i] = static_cast<int8>(dis(gen));
    }
    for (int i = 0; i < N*K; i++) {
        b_host[i] = static_cast<int8>(dis(gen));
    }
}
