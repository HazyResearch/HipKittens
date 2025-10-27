# make TARGET=tk_kernel SRC=kernel_1024.cpp
# python test_python.py 1024  mi325x_bf16_gemm.json

# make TARGET=tk_kernel SRC=kernel_2048.cpp
# python test_python.py 2048 mi325x_bf16_gemm.json

# make TARGET=tk_kernel SRC=kernel_4096.cpp
# python test_python.py 4096 mi325x_bf16_gemm.json

make TARGET=tk_kernel SRC=kernel_8192.cpp
python test_python.py 8192 mi325x_bf16_gemm.json

# make TARGET=tk_kernel SRC=kernel_16384.cpp
# python test_python.py 16384 mi325x_bf16_gemm.json