#!/bin/bash
if [ ! -d "build" ]
then
    mkdir build
fi
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/libtorch ..
cmake --build . --config Release
