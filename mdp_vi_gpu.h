#ifndef MDP_VI_GPU_H
#define MDP_VI_GPU_H
#include "mdp.h"
#include "mpi.h"
#include "value_function.h"
#include <cuda_runtime.h>

class MDPVIGPU {
private:
  MPI_Comm slaves;
  float *VInitial;
  unsigned int numThreads;
  unsigned int numBlocks;
  unsigned int startState;
  int stopState;
  int size;
  int rank;
  unsigned int gpu_use=0;
  unsigned int currentHorizon;
  float *d_V=nullptr;
  float *d_VPrime=nullptr;
  float *d_temp=nullptr;
  float *d_change=nullptr;
  float *d_change_max=nullptr;
  float *h_change_max=nullptr;
  unsigned int *d_pi=nullptr;
  cudaStream_t stream;

  int mdp_vi_update_gpu(mdp *MDP);

public:
  unsigned int numStates;
  int *listStates=nullptr;
  int *displs=nullptr;

  MDPVIGPU(float* Vinitial_in, unsigned int numThreads_in, unsigned int size_in,
    unsigned int rank_in, MPI_Comm comm);
  ~MDPVIGPU();
  int mdp_vi_initialize_gpu(mdp *MDP);
  int mdp_vi_uninitialize_gpu(mdp *MDP);
  int mdp_vi_execute_gpu(mdp *MDP, value_function *policy);
  int mdp_vi_get_policy_gpu(mdp *MDP, value_function *policy);
  long long time_gather=0, time_bellman=0, time_copy=0;
};

#endif
