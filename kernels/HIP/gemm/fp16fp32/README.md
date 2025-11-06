# FP16 (base type) FP32 (accumulator type) matmuls
For illustrative and learning purposes. These roughly follow Simon Boehm's matmul kernel worklog here: https://siboehm.com/articles/22/CUDA-MMM

The order to look at these roughly goes (*italicized entries* aren't strictly a part of the ordering):

1. basic_matmul_row.cpp
2. basic_matmul_col.cpp
3. blocked_matmul.cpp
4. 1d_blocked_matmul.cpp
5. 2d_blocked_matmul.cpp
6. vectorize.cpp
7. *resolve_bank_conflicts.cpp*
8. *resolve_bank_extra_cols.cpp*
9. warptiling.cpp
10. warptiling_mfma.cpp
11. *double_buffering.cpp*
