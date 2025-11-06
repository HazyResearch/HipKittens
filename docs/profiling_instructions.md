
## ROCProfiler

This is a profiler that we've used to observe statistics like bank conflicts, cache usage, pipeline utilization statistics, etc. 

One-time setup depending on the machine that you are using:
```bash
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
localedef -i en_US -f UTF-8 en_US.UTF-8 || true
```

Installation:
```bash
sudo apt install rocprofiler-compute
pip install -r /opt/rocm-6.4.1/libexec/rocprofiler-compute/requirements.txt
apt-get install locales
locale-gen en_US.UTF-8

# collect profile
rocprof-compute profile -n transpose_matmul --no-roof -- python test_python.py

# view statistics
rocprof-compute analyze -p workloads/transpose_matmul/MI300/ --gui
```

Potential failure mode:
- If you see an error with ```run_server``` being an invalid function, then edit the problematic file to use ```run()```.


## ROCProfiler V3

This profiler lets you see finer grained execution traces from your program. 


```bash
mkdir -p rocprofiler-setup
cd rocprofiler-setup

git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
sudo apt-get update
sudo apt-get install libdw-dev libelf-dev elfutils
cmake -B rocprofiler-sdk-build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm/ rocprofiler-sdk-source
cmake --build rocprofiler-sdk-build --target all --parallel 32
cmake --build rocprofiler-sdk-build --target install

# if using ubuntu 22.04 (like the rocm7.0 preview container)
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb
sudo dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb

# if using ubuntu 24.04 (like the rocm 6.4.1 container we used on the mi325x)
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb
sudo dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb

git clone https://github.com/ROCm/aqlprofile.git
cd aqlprofile
./build.sh
cd build
sudo make install
cd ../../..
rm -rf rocprofiler-setup
```

## Failure modes

1. No package 'libdw' found:
```bash
sudo apt-get update
sudo apt-get install libdw-dev libelf-dev elfutils
```

2. Decoder errors:
LD_DEBUG=libs rocprofv3 \
  --att=true \
  --att-library-path /opt/rocm/lib \
  -d transpose_matmul \
  -- python3 test_python.py 2>&1 | tee decoder_debug.log

Check detailed information about what is going wrong.


## Commands

Use these commands to collect traces for your kernel. This will produce a new trace directory for every kernel called inside of ```test_python.py```.

On MI325:
```bash
rocprofv3 --att=true \
          --att-library-path /opt/rocm-6.4.1/lib \
          -d transpose_matmul \
          -- python3 test_python.py
```

On MI350:
```bash
rocprofv3 --att=true \
          --att-library-path /opt/rocm/lib \
          -d transpose_matmul \
          -- python3 test_python.py
```


## Inspect results

Next we want to be able to visualize the trace. Clone below and build from source on your local filesystem:

```bash
git clone https://github.com/ROCm/rocprof-compute-viewer
cd rocprof-compute-viewer/

# on your local mac computer (for other platforms: https://github.com/ROCm/rocprof-compute-viewer?tab=readme-ov-file#building-from-source)
brew install qt@6
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$(brew --prefix qt@6)
make -j
```

Take the output folder produced in ```transpose_matmul``` above, which corresponds to your desired kernel and download it to your local laptop filesystem. Then import it into the viewer. 

Details on how to use the profiler once it's set up and your trace is loaded in can be found here: https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/latest/how-to/using_compute_viewer.html#using-compute-viewer. 

