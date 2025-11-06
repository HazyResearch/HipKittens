

This folder has two kernel versions: 
1. ```kernel_v1.cpp```: Uses tile sizes 256x256x32 and the gold schedule
2. ```kernel_v2.cpp```: Uses tile sizes 256x256x64 and the gold schedule, but has lots of register spills


To try any of these, toggle the target in the Makefile and ```make clean && make && python test_python.py```



