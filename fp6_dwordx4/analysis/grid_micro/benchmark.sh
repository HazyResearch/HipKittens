
make TARGET=tk_kernel SRC=kernel_8192_wNone.cpp
python test_python.py 0

make TARGET=tk_kernel SRC=kernel_8192_w2.cpp
python test_python.py 2

make TARGET=tk_kernel SRC=kernel_8192_w4.cpp
python test_python.py 4

make TARGET=tk_kernel SRC=kernel_8192_w8.cpp
python test_python.py 8

make TARGET=tk_kernel SRC=kernel_8192_w32.cpp
python test_python.py 32
