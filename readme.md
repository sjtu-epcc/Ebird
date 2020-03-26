# Ebird
The source code of the paper: ["Ebird: Elastic Batch for Improving Responsiveness and Throughput of Deep Learning Services"](https://ieeexplore.ieee.org/abstract/document/8988602/) in ICCD 2019.

# Build from source:
- Prerequisites: Ninja, CUDA, CUDNN, GLOG
- Edit config.linux for settings about prerequisites including `GLOG`,`CUDA` and `CUDNN`
- Build
```
mkdir build
cd build
cmake -GNinja ..
cmkae --build .
```
# Referrence
[superneurons](https://github.com/linnanwang/superneurons-release)
# License
Licensed under an Apache-2.0 license.
