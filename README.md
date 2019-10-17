# Distributed MPI based Heterogenous GPU Solver for Markov Decision Processes (MDP)

This repository provides an MPI ([Message Passing Interface](https://en.wikipedia.org/wiki/Message_Passing_Interface)) based solution for distributed solving of [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP) on multiple nodes, where each node has a dedicated GPU. These GPUs may be of heterogenous computing capabilities (for example NVIDIA GTX 2080 and a Jetson Nano).

This work was done for the fulfillment of my B.Eng. under the supervision of Prof. Tham Chen Khong, National University of Singapore. Also, this work is built on top of an existing solver for MDPs which use a single GPU on a single computer, provided by Kyle Wray in his github repository [nova](https://github.com/kylewray/nova).

The algorithm performs the Value Iteration algorithm to solve an MDP by distributing the different Markov states across machines. This distribution is dynamically handled in a way so that more powerful machines share a greater load of the problem. Also, nodes whose GPU may be busy with other workloads is dynamically given less number of Markov states.

## Instructions to run code
#### 1. Prerequisites
To prevent difficulties in running this code, ensure that all the below prerequisites are available in each node. Also keep in mind that all nodes have the exact same versions of MPI, CUDA etc.
1. You need to install a CUDA aware MPI library. One such MPI is that provided by [Open MPI](https://www.open-mpi.org) library. Note that OpenMPI by default install a CUDA unaware MPI. To look at instructions to enable CUDA in OpenMPI, look at [this link](https://www.open-mpi.org/faq/?category=buildcuda). Using a CUDA aware MPI library allows message passing directly between GPU memory across nodes.
2. C++
3. [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

#### 2. Compiling the code
Compiling the code requires using the nvidia `nvcc` command. Also, you need to provide the links to the MPI libraries installed on your computer to `nvcc`. The overall command on a Linux machine looks like:
`nvcc -I/path/to/openmpi/include/ -L/path/to/openmpi/lib/ -lmpi -std=c++11 *.cu *.cpp -o program`

#### 3. Executing the code
You can then run the executable generated using `mpiexec`. For example, the following solves 2 MDP problems one after the other over 4 processes on the same machine:
`mpiexec -np 4 ./program  mdp_1.raw nStates_1 mdp_2.raw nStates_2`

The program uses the `.raw` file format also used in the nova repository. To understand teh `.raw` format better, look at the code in `MDP_Create.py`. You can ask the program to solve multiple MDPs by adding more arguments. Note: Even though the solving of a single GPU is distributed, the MDPs themselves are solved one after the other. Future work may involve distributing multiple MDP problems over the network as well.

To run the code over a local cluster of machines, I suggest looking at [this link](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/).

## Code structure
A brief summary for each file is provided below for better debugging / improvements.
1. `MDP_Create.py`: Used to create random MDP `.raw` files to debug / analyze performance. The different parameters defining the MDP can be defined here, such as the number of states, number of actions, etc.
2. `gpu_stats.cpp`: A C++ thread that constantly measures the utilization of the GPU in each node, averaged over time. This information is periodically shared with all nodes to redistribute the workload if required. Supports both consumer GPUs using the `nvidia-smi` command as well as Jetson TX1/TX2 using the `tegrastats` command.
3. `mdp_gpu_solver.cpp`: Wrapper for the CUDA code that performs the Value Iteration algorithm
4. `mdp_vi_gpu.cu`: CUDA code that performs the value iteration algorithm. It defines the CUDA kernels to be executed on the GPU
5. `mdp.cu`: Class that saves the MDP data both on host as well as GPU device memory.
6. `value_function.cpp`: Class that saves the calculated optimal policy and value function.
7. `mpi-master-slave.cpp`: Main program that orchestrates the code and performs input/output and MPI message passing.

Also provided with the repository is a short technical paper written on the work. This paper does NOT consider the dynamic reallocation of workload that has been implemented in the code. However, it provides a good conceptual understanding of spreading the workload across multiple machines. It looks at the problem of computational offloading in a distributed factory floor setup, using MDPs to find the optimal time for performing maintenance. Some interesting results are also presented.