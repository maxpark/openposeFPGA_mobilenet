#!/bin/bash

# inst_gen
rm ./inst_gen/openpose.insts
rm ./inst_gen/params.h
rm ./inst_gen/weight_offset.dat
rm ./inst_gen/bias_offset.dat

# HLS_project
rm -rf ./HLS_project/HLS_kernel/output
rm ./HLS_project/2D*
rm ./HLS_project/common*
rm ./HLS_project/params.h
rm -rf ./HLS_project/pose_prj

# SDx_project
rm ./SDx_project/src/hw_kernel.cpp
rm ./SDx_project/src/hw_kernel0.cpp
rm ./SDx_project/src/params.h
cd ./SDx_project/System
make clean
cd -

# data
rm ./data/bias_reorg.bin
rm ./data/weight_reorg.bin
