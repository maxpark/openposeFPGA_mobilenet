OpenPose FPGA Implementation
==============

This repository includes the code to build FPGA accelerators for the OpenPose application. OpenPose is used for detecting human body, hands, and facial keypoints. For more details about the application, please refer to the original OpenPose repo [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

The original Openpose uses VGG network. In this project, we use the modified network that replaces the original VGG network with Mobilenet that helps cut down the computation cost dramatically. 
The network we used is from the repo [here](https://github.com/ildoonet/tf-pose-estimation).

The openpose algorithm contains two parts, the neural network and the graph processing. The current FPGA implementation accelerates the processing of neural network.

In this repo, we build an HLS-based FPGA accelerator for the OpenPose application. The target of this project is to achieve real-time processing with high energy efficiency. The code in this repository has been tested with Xilinx SDx 2018.3 on Xilinx VCU1525 platform.

The implemented accelerator achieves 14.0 FPS for 384x384 input. 

The structure of this repo is as follows:
```
.
|-- README.md: this file
|-- inst_gen: directory for instruction generator
|-- HLS_project: directory for HLS project
    |-- HLS_kernel: directory for accelerator kernel
|-- SDx_project: directory for SDx project
|-- data: contains the model data and inputs for the network
|-- dse: directory for design space exploration engine
|-- clean.sh: cleans the directory to the initial status
```

## Contents
1. [Getting Started](#getting-started)
2. [Latest Features](#latest-features)
3. [Design Details](#design-details)
4. [Version History](#version-history)

## Getting Started

The working flow of generating the FPGA accelerator for OpenPose contains three steps: 1. **instruction generation** 2. **kernel generation** 3. **project building**. In the first step, users need to provide with a model description file describing the CNN model used in the application. The instruction generator will parse the model configuration file and generate the instructions for the accelerator and header files for the HLS design. Next, we will copy the instructions and header files generated in the previous step to the HLS kernel directory and verify the design. Lastly, we will build the SDx project to generate the final bitstream running on-board. Please follow the instructions below to generate the design.

Before you start, first set up the environment using the command below:
```
export PRJ_PATH=$(pwd)
```
And you need dependencies below.
- python3
- Xilinx SDAccel 2018.3

1. **Instruction and Data Generation**

We will need to generate the instructions required for the FPGA kernel and pre-process the input data and weights of the network.

First, switch to the instruction generator folder.
```
cd $PRJ_PATH/inst_gen
```
Please modify the model configuration file `inst_gen/openpose.model` according to your network configuration. Run the following command to parse the model and generate the necessary files.
```
python inst_parse.py -t ./tile.json -m ./openpose.model -mc ./network_topology.json -i ./input.json
```
There will be four files generated: 
- `openpose.insts`: contains instructions to configure the FPGA acclerator to perform the computation tasks
- `params.h`: contains all the parameters required by the HLS kernel
- `weight_offset.dat`: helps the host program to load the weights
- `bias_offset.dat`: helps the host program to load the bias

Next, switch to the data folder.
```
cd $PRJ_PATH/data
```
Run the command below to pre-process all the data.
```
python data_reorg.py -t ../inst_gen/tile.json -m ../inst_gen/openpose.model -mc ../inst_gen/network_topology.json -i ../inst_gen/input.json -w weight.bin -b bias.bin
```

2. **Build the HLS kernel**

Switch to the HLS project directory.
```
cd $PRJ_PATH/HLS_project
```
Use the following command to generate the HLS kernel and prepare all the necessary files.
```
./design_prepare.sh
```
Next, you could run the HLS C simulation to verify the design.
```
vivado_hls -f hls_script.tcl
```
It will take several minutes or so to finish the C simulation. 

3. **Build the SDx project**

So far we have generated the HLS kernel files for the FPGA accelerator. Next, we will need to build the bitstream of the FPGA kernel.

    3.1 Prepare the SDx kernel

We need to combine all kernel files into one single file for SDx project. 
To start with, switch to SDx project directory.
```
cd $PRJ_PATH/SDx_project
```

And run the following script to generate the SDx kernel.
```
./sdx_kernel_create.sh
```

You should be able to see all the necessary kernel files in the `src` directory.

    3.2 Build the bitstream
    
Generate the bitstream under the `System` directory.
```
cd System
make all
```
It will take several hours or so to generate the bistream. You can change the target frequency in the makefile following the `--kernel_frequency [200]`.
You wil find the host file `pose_prj.exe` and the bistream `binary_container_1.xclbin` under the same directory.
For running the kernel, use the command:
```
./pose_prj.exe binary_container_1.xclbin
```

## Latest Features

## Design Details

## Version History

+ [2019-03-12] version 1.0 released
