#include "mdp_gpu_solver.h"
#include <iostream>
#include "mpi.h"

MDPValueIterationGPU::MDPValueIterationGPU(mdp *MDP_in, int numThreads, float *Vinitial, int size, int rank, MPI_Comm slaves){
  if(MDP_in==nullptr || numThreads%32 != 0){
    fprintf(stderr, "Error[MDPValueIterationGPU]: %s\n", "Invalid arguments.");
    return;
  }
  MDP = MDP_in;

  if(Vinitial==nullptr){
    Vinitial = new float[MDP->n];
    if(MDP->gamma <1.0){
      for (size_t i = 0; i < MDP->n; i++) {
        Vinitial[i] = (float)MDP->Rmin/(1.0-MDP->gamma);
      }
    }
    else{
      for (size_t i = 0; i < MDP->n; i++) {
        Vinitial[i] = 0.0;
      }
    }
  }

  vi = new MDPVIGPU(Vinitial, numThreads, size, rank, slaves);
  int result = vi->mdp_vi_initialize_gpu(MDP);

  if(result != 0)
    fprintf(stderr, "Error[MDPValueIterationGPU]: %s\n", "Failed to initialize VI(GPU).");
}

MDPValueIterationGPU::~MDPValueIterationGPU(){
  int result = vi->mdp_vi_uninitialize_gpu(MDP);
  delete vi;
  if (result != 0){
    fprintf(stderr, "Error[MDPValueIterationGPU]: %s\n", "Failed to uninitialize the GPU MDP solver.");
  }
}

value_function MDPValueIterationGPU::solve(){
  value_function policy;
  int result = vi->mdp_vi_execute_gpu(MDP, &policy);

  if(result !=0 ){
    fprintf(stderr, "Error[MDPValueIterationGPU]: %s\n", "Failed to execute VI(GPU) solver.");
  }
  return policy;
}

void MDPValueIterationGPU::getStats(int *numStates, int** displs, int** listStates){
  *numStates = vi->numStates;
  *displs = vi->displs;
  *listStates = vi->listStates;
  return;
}
