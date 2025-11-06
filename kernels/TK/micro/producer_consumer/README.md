


| Micro                       | Output Tile Size          | Notes                                           |
|-----------------------------|---------------------------|-------------------------------------------------|
| micro_02_2stage_8c4p.cpp                | (128)x(256)   | 8 consumers, 4 producers; 2-stage               |
| micro_03_3stage_8c4p.cpp                | (128)x(256)   | 8 consumers, 4 producers; 3-stage               |
| micro_04_2stage_12c4p.cpp               | (192)x(256)   | 12 consumers, 4 producers; 2-stage              |
| micro_05_2stage_16c2p.cpp  (Above SW limit)     | (128)x(256)   | 16 consumers, 4 producers; 2-stage      |
| micro_06_2stage_8c4p_64x96.cpp          | (128)x(256)   | 8 consumers, 4 producers; 2-stage; 96 reduction |
| micro_06_2stage_8c4p_96x64.cpp (Scratch/Spills) | (128)x(256)   | 8 consumers, 4 producers; 2-stage; 64 reduction |
| micro_07_2stage_8c4p_nblock8.cpp (Scratch/Spills) | (128)x(512)   | 8 consumers, 4 producers; 2-stage; n-block 8 |


## Debug async code

```bash
apt update && apt install screen
screen -S mysession
Ctrl+A, C - Create new window
Ctrl+A, N - Next window
Ctrl+A, P - Previous window
Ctrl+A, D - Detach session
Ctrl+A, K - Kill current window
```

Or:
```bash
timeout 30s python test_gemm.py
```


