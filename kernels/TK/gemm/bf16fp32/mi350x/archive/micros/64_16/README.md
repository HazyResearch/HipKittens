
```bash
# measure bank conflicts
rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- python3 test_python.py

# view bank conflicts
python out/analyze_conflicts.py
```

