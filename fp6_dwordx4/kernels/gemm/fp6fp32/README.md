

Check available ROCM intrinsics:

```bash

grep -r -i "fp6" /opt/rocm-7.0.0/include/ > out.log
```

https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/install/rocm.html

```bash
# measure bank conflicts
rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- ./tk_kernel

# view bank conflicts
python out/analyze_conflicts.py
```



