#!/bin/bash
if [ ! -d "build" ]
then
    mkdir build
fi
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/libtorch -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++-11 ..
cmake --build . --config Release
