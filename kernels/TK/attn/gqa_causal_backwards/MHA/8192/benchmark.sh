cd $THUNDERKITTENS_ROOT
git checkout asm_port
cd $THUNDERKITTENS_ROOT/../kernels/TK/attn/gqa_causal_backwards/MHA/8192
make SRC=GQA_bkwd.cpp TARGET=tk_kernel_bkwd

cd $THUNDERKITTENS_ROOT
git checkout port
cd $THUNDERKITTENS_ROOT/../kernels/TK/attn/gqa_causal_backwards/MHA/8192
make SRC=GQA_bkwd_prep.cpp TARGET=tk_kernel_bkwd_prep
make SRC=GQA_fwd.cpp TARGET=tk_kernel_fwd

python test_attn.py