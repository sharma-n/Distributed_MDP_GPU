#define TAG_RESOURCE_DATA 2

#include <cmath>
#include <iostream>
#include "mdp.h"
#include "value_function.h"
#include <cfloat>
#include "mpi.h"
#include "mdp_vi_gpu.h"
#include "gpu_stats.h"
#include <chrono>
#include <cfloat>
#include <cuda_runtime.h>

__global__ void mdp_vi_bellman_update_gpu(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V,
        float *VPrime, unsigned int *pi, unsigned int startState, unsigned int stopState)
{
    // The current state as a function of the blocks and threads.
    int s;

    // The intermediate Q(s, a) value.
    float Qsa;/* message */

    // The index within S and T (i.e., in n*s*ns).
    int index;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spindex;

    // Compute the index of the state. Return if it is beyond the stopState.
    s = blockIdx.x * blockDim.x + threadIdx.x + startState;
    if (s>=n || s > stopState) {
        return;
    }

    // Nvidia GPUs follow IEEE floating point standards, so this should be safe.
    VPrime[s] = -FLT_MAX;

    // Compute max_{a in A} Q(s, a).
    for (int a = 0; a < m; a++) {
        // Compute Q(s, a) for this action.
        Qsa = R[s * m + a];

        for (int sp = 0; sp < ns; sp++) {
            index = s * m * ns + a * ns + sp;

            spindex = S[index];
            if (spindex < 0) {
                break;
            }

            Qsa += gamma * T[index] * V[spindex];
        }

        // __syncthreads(); //I don't see the point of this __syncthreads()

        if (a == 0 || Qsa > VPrime[s]) {
            VPrime[s] = Qsa;
            pi[s-startState] = a;
        }

        __syncthreads();  //neither this
    }
}

__global__ void copy_kernel(float *in, float *temp, int n, int startState, int stopState){
  int s = blockIdx.x * blockDim.x + threadIdx.x + startState;
  if (s>=n || s > stopState) {
      return;
  }
  temp[s-startState] = in[s];
}

__global__ void max_reduce(float *d_V, float *d_VPrime, float *g_idata, float *g_odata, unsigned int n)
{
  extern __shared__ float sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? abs(d_V[i]-d_VPrime[i]) : -FLT_MAX;

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
      if (tid < s && i < n)
          sdata[tid] = max(sdata[tid], sdata[tid + s]);
      __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

MDPVIGPU::MDPVIGPU(float *Vinitial_in, unsigned int numThreads_in,
  unsigned int size_in, unsigned int rank_in, MPI_Comm comm){
  VInitial = Vinitial_in;
  slaves = comm;
  numThreads = numThreads_in;
  size = size_in;
  rank = rank_in;
  listStates = new int[size];
  displs = new int[size] {0};
  for (size_t i = 1; i < size; i++) {
    displs[i] = displs[i-1]+listStates[i-1];
  }
}

MDPVIGPU::~MDPVIGPU(){
  delete [] h_change_max;
  delete [] listStates;
  delete [] displs;
  delete [] VInitial;
}

int MDPVIGPU::mdp_vi_initialize_gpu(mdp *MDP){
  if (MDP == nullptr || MDP->n == 0) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n", "Invalid arguments.");
      return -1;
  }

  // Reset the current horizon.
  currentHorizon = 0;

  // Create VInitial if undefined.
  bool createdVInitial = false;
  if (VInitial == nullptr) {
      VInitial = new float[MDP->n];
      for (unsigned int i = 0; i < MDP->n; i++) {
          VInitial[i] = 0.0f;
      }
      createdVInitial = true;
  }

  // cudaStreamCreate(&stream);

  // Create the device-side V.
  if (cudaMalloc(&d_V, MDP->n * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the value function.");
      return -1;
  }
  if (cudaMemcpy(d_V, VInitial, MDP->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to copy memory from host to device for the value function.");
      return -1;
  }

  if (cudaMalloc(&d_VPrime, MDP->n * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the value function (prime).");
      return -1;
  }
  if (cudaMemcpy(d_VPrime, VInitial, MDP->n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to copy memory from host to device for the value function (prime).");
      return -1;
  }

  // Create the device-side pi.
  if (cudaMalloc(&d_pi, numStates * sizeof(unsigned int)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the policy (pi).");
      return -1;
  }

  unsigned int *pi = new unsigned int[numStates];
  for (unsigned int i = 0; i < numStates; i++) {
      pi[i] = 0;
  }

  if (cudaMemcpy(d_pi, pi, numStates * sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to copy memory from host to device for pi.");
      return -1;
  }

  if (cudaMalloc(&d_temp, numStates * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the temporary partial value function.");
      return -1;
  }
  delete [] pi;

  if (cudaMalloc(&d_change, MDP->n * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the change in value function.");
      return -1;
  }
  h_change_max = new float[(int)((float)(MDP->n) / (float)numThreads) + 1];
  if (cudaMalloc(&d_change_max, MDP->n * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
              "Failed to allocate device-side memory for the max changes in value function.");
      return -1;
  }

  // If we created VInitial, then free it.
  if (createdVInitial) {
      delete [] VInitial;
      VInitial = nullptr;
  }

  return 0;
}

int MDPVIGPU::mdp_vi_uninitialize_gpu(mdp *MDP){
  if (MDP == nullptr) {
      fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n", "Invalid arguments.");
      return -1;
  }

  int result;

  result = 0;

  // Reset the current horizon.
  currentHorizon = 0;

  if (d_V != nullptr) {
      if (cudaFree(d_V) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                  "Failed to free memory from device for the value function.");
          result = -1;
      }
  }
  d_V = nullptr;

  if (d_VPrime != nullptr) {
      if (cudaFree(d_VPrime) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                  "Failed to free memory from device for the value function (prime).");
          result = -1;
      }
  }
  d_VPrime = nullptr;

  if (d_pi != nullptr) {
      if (cudaFree(d_pi) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                  "Failed to free memory from device for the policy (pi).");
          result = -1;
      }
  }
  d_pi = nullptr;

  if (d_temp != nullptr){
    if (cudaFree(d_temp) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                "Failed to free memory from device for temporary partial value function.");
        result = -1;
    }
  }
  d_temp = nullptr;

  if (d_change != nullptr){
    if (cudaFree(d_change) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                "Failed to free memory from device for change in value function.");
        result = -1;
    }
  }
  d_change = nullptr;

  if (d_change_max != nullptr){
    if (cudaFree(d_change_max) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                "Failed to free memory from device for change in value function.");
        result = -1;
    }
  }
  d_change_max = nullptr;

  //cudaStreamDestroy(stream);

  return result;
}

int MDPVIGPU::mdp_vi_execute_gpu(mdp *MDP, value_function *policy){
  // First, ensure data is valid.
  if (MDP == nullptr || MDP->n == 0 || MDP->ns == 0 || MDP->m == 0 ||
          MDP->S == nullptr || MDP->T == nullptr || MDP->R == nullptr ||
          MDP->gamma < 0.0f || MDP->gamma > 1.0f || MDP->horizon < 1 || policy == nullptr) {
      fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Invalid arguments.");
      return -1;
  }

  int result=0;


  // dim3 dimBlock(numThreads, 1, 1);
  // int nBlock = (int)((float)(MDP->n) / (float)numThreads) + 1;
  // dim3 dimGrid(nBlock, 1, 1);
  // int smemSize = numThreads * sizeof(float);
  // float max_eps = -FLT_MAX;
  // int numIter = 0;
  // do {
  //     // if (rank==0)
  //     //   result += mdp_vi_update_gpu_master(MDP);
  //     // else
  //     //gpu_use += get_gpu_usage();
  //     max_eps = -FLT_MAX;
  //     result += mdp_vi_update_gpu(MDP);
  //     if (result != 0) {
  //         fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to perform Bellman update on the GPU.");
  //
  //         int resultPrime = mdp_vi_uninitialize_gpu(MDP);
  //         if (resultPrime != 0) {
  //             fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
  //         }
  //         return result;
  //     }
  //     max_reduce<<<dimGrid, dimBlock, smemSize>>>(d_V, d_VPrime, d_change, d_change_max, MDP->n);
  //     cudaMemcpy(h_change_max, d_change_max, nBlock*sizeof(float), cudaMemcpyDeviceToHost);
  //     for (int i=0; i<nBlock; i++){
  //       max_eps = (max_eps>h_change_max[i]) ? max_eps : h_change_max[i];
  //     }
  //     numIter++;
  // } while (max_eps > MDP->epsilon);
  // if(rank == 0){
  //     std::cout << "Number of iterations: " << numIter << '\n';
  //     std::cout << "GPU use: " << get_gpu_usage() << '\n';
  // }

  while (currentHorizon < MDP->horizon) {
    if (currentHorizon%200==0) {
      int gpu_usage = get_gpu_usage();
      MPI_Send(&gpu_usage, 1, MPI_INT, 0, TAG_RESOURCE_DATA, MPI_COMM_WORLD);
      MPI_Status stat;
      MPI_Recv(listStates, size, MPI_INT, 0, TAG_RESOURCE_DATA, MPI_COMM_WORLD, &stat);
      startState = 0;
      stopState = -1;
      for (size_t i = 0; i < rank; i++) {
        startState += listStates[i];
      }
      for (size_t i = 0; i <= rank; i++) {
        stopState += listStates[i];
      }
      numStates = stopState-startState+1;
      numBlocks = (unsigned int)((float)(numStates) / (float)numThreads) + 1;
      for (size_t i = 1; i < size; i++) {
        displs[i] = displs[i-1]+listStates[i-1];
      }

      if (cudaFree(d_pi) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                  "Failed to free memory from device for the policy (pi).");
          result = -1;
      }
      d_pi = nullptr;
      if (cudaMalloc(&d_pi, numStates * sizeof(unsigned int)) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                  "Failed to allocate device-side memory for the policy (pi).");
          return -1;
      }
      if (cudaFree(d_temp) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_uninitialize_gpu]: %s\n",
                  "Failed to free memory from device for temporary partial value function.");
          result = -1;
      }
      d_temp = nullptr;
      if (cudaMalloc(&d_temp, numStates * sizeof(float)) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_initialize_gpu]: %s\n",
                  "Failed to allocate device-side memory for the temporary partial value function.");
          return -1;
      }
    }

    result += mdp_vi_update_gpu(MDP);
    if (result != 0) {
        fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to perform Bellman update on the GPU.");

        int resultPrime = mdp_vi_uninitialize_gpu(MDP);
        if (resultPrime != 0) {
            fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
        }
        return result;
    }
  }

  long long total_gather, total_bellman, total_copy;

  MPI_Reduce(&time_gather, &total_gather, 1, MPI_LONG_LONG, MPI_SUM, 0, slaves);
  MPI_Reduce(&time_bellman, &total_bellman, 1, MPI_LONG_LONG, MPI_SUM, 0, slaves);
  MPI_Reduce(&time_copy, &total_copy, 1, MPI_LONG_LONG, MPI_SUM, 0, slaves);
  if (rank==0) {
    std::cout << total_gather/size << ',' << total_bellman/size << ',' << total_copy/size << '\n';
  }
  result += mdp_vi_get_policy_gpu(MDP, policy);
  if (result != 0) {
      fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to get the policy.");

      int resultPrime = mdp_vi_uninitialize_gpu(MDP);
      if (resultPrime != 0) {
          fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
      }
      return result;
  }

  result += mdp_vi_uninitialize_gpu(MDP);
  if (result != 0) {
      fprintf(stderr, "Error[mdp_vi_execute_gpu]: %s\n", "Failed to uninitialize the GPU variables.");
      return result;
  }

  return result;
}

int MDPVIGPU::mdp_vi_update_gpu(mdp *MDP){
  // Compute the number of blocks.
  numBlocks = (unsigned int)((float)(numStates) / (float)numThreads) + 1;

  if (currentHorizon % 2 == 0) {
    auto start = std::chrono::high_resolution_clock::now();
    mdp_vi_bellman_update_gpu<<< numBlocks, numThreads, 0 >>>(
                  MDP->n, MDP->ns, MDP->m, MDP->gamma,
                  MDP->d_S, MDP->d_T, MDP->d_R,
                  d_V, d_VPrime, d_pi, startState, stopState);
    cudaDeviceSynchronize();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_bellman += microseconds;
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    mdp_vi_bellman_update_gpu<<< numBlocks, numThreads, 0 >>>(
                  MDP->n, MDP->ns, MDP->m, MDP->gamma,
                  MDP->d_S, MDP->d_T, MDP->d_R,
                  d_VPrime, d_V, d_pi, startState, stopState);
    cudaDeviceSynchronize();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_bellman += microseconds;
  }

  // Check if there was an error executing the kernel.
  if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_update_gpu]: %s\n",
                      "Failed to execute the 'Bellman update' kernel.");

      return -1;
  }

  // Wait for the kernel to finish before looping more.
  if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_update_gpu]: %s\n",
                  "Failed to synchronize the device after 'Bellman update' kernel.");
      return -1;
  }
  if (currentHorizon % 2 == 0) {
    auto start = std::chrono::high_resolution_clock::now();
    copy_kernel <<<numBlocks, numThreads, 0 >>> (d_VPrime, d_temp, MDP->n, startState, stopState);
    cudaDeviceSynchronize();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_copy += microseconds;
    start = std::chrono::high_resolution_clock::now();
    MPI_Allgatherv(d_temp, numStates, MPI_FLOAT, d_VPrime, listStates, displs, MPI_FLOAT, slaves);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_gather += microseconds;
  } else {
    auto start = std::chrono::high_resolution_clock::now();
    copy_kernel <<<numBlocks, numThreads, 0 >>> (d_V, d_temp, MDP->n, startState, stopState);
    cudaDeviceSynchronize();
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_copy += microseconds;
    start = std::chrono::high_resolution_clock::now();
    MPI_Allgatherv(d_temp, numStates, MPI_FLOAT, d_V, listStates, displs, MPI_FLOAT, slaves);
    elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    time_gather += microseconds;
  }

  currentHorizon++;

  return 0;
}

int MDPVIGPU::mdp_vi_get_policy_gpu(mdp *MDP, value_function *policy){
  if (MDP == nullptr || policy == nullptr) {
      fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n", "Invalid arguments.");
      return -1;
  }
  // Initialize the policy, which allocates memory.
  int result = policy->initialize(MDP->n, MDP->m, numStates);
  if (result != 0) {
      fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n", "Could not create the policy.");
      return -1;
  }

  // Copy the final (or intermediate) result, both V and pi, from device to host. This assumes
  // that the memory has been allocated for the variables provided.
  if (currentHorizon % 2 == 0) {
      if (cudaMemcpy(policy->V, d_V, MDP->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                  "Failed to copy memory from device to host for the value function.");
          return -1;
      }
  } else {
      if (cudaMemcpy(policy->V, d_VPrime, MDP->n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
                  "Failed to copy memory from device to host for the value function (prime).");
          return -1;
      }
  }

  if (cudaMemcpy(policy->pi, d_pi, numStates * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_vi_get_policy_gpu]: %s\n",
              "Failed to copy memory from device to host for the policy (pi).");
      return -1;
  }

  return 0;
}
