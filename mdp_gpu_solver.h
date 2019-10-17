// mdp_gpu_solver.h
#ifndef mdp_gpu_solver_H
#define mdp_gpu_solver_H
#include "mdp.h"
#include "mdp_vi_gpu.h"
#include "mpi.h"

class MDPValueIterationGPU{
private:
  MDPVIGPU *vi;
  mdp *MDP;
public:
  MDPValueIterationGPU(mdp *MDP_in, int numThreads, float *Vinitial, int size, int rank, MPI_Comm slaves);
  ~MDPValueIterationGPU();
  value_function solve();
  void getStats(int *numStates, int ** displs, int **listStates);
};

#endif
