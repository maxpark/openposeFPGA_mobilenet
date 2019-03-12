#!/bin/bash

# merge HLS kernel files into one SDx kernel file
cat ../HLS_project/common_header_U1.h >> ./src/hw_kernel0.cpp
cat ../HLS_project/2DDataFeedCollect_U1.cpp >> ./src/hw_kernel0.cpp
cat ../HLS_project/2DDataFeed_U1.cpp >> ./src/hw_kernel0.cpp
cat ../HLS_project/2DDataCollect_U1.cpp >> ./src/hw_kernel0.cpp
cat ../HLS_project/2DPE_U1.cpp >> ./src/hw_kernel0.cpp
cat ../HLS_project/kernel.cpp >> ./src/hw_kernel0.cpp

# modify hw_kernel.cpp
python hw_kernel_modify.py -i src/hw_kernel0.cpp -o src/hw_kernel.cpp

# copy params.h to SDx project
cp ../HLS_project/params.h ./src/
