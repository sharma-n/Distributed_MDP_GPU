// Pseudocode for master slave

#define TAG_ASK_FOR_JOB 0
#define TAG_JOB_DATA 1
#define TAG_RESOURCE_DATA 2
#define TAG_STOP 3

#include <iostream>
#include "mdp.h"
#include "value_function.h"
#include "mdp_vi_gpu.h"
#include "mdp_gpu_solver.h"
#include "mpi.h"
#include <string>
#include <cstring>
#include <memory>
#include <array>
#include <stdlib.h>
#include "gpu_stats.h"

struct MDPSolveObject{
  mdp *MDP;
  value_function *policy;
  MDPValueIterationGPU *algorithm;
  ~MDPSolveObject(){
    delete algorithm;
    //policy->uninitialize();
    MDP->uninitialize();
    delete policy;
    delete MDP;
  }
};

struct Problem{
    std::string file;
    int nStates, h, w;
    int *pi_opt;
    Problem(std::string name, int n, int w_in, int h_in){
        file = name;
        nStates = n;
        pi_opt = new int[n];
        w = w_in;
        h = h_in;
    }
    Problem(std::string name, int n){
      file = name;
      nStates = n;
    }
    // ~Problem(){
    //   delete [] pi_opt;
    // }
};

// for the master you have to provide code for managing individual tasks
// and for managing the slaves with their tasks they are working on
void master (int nSlaves, int argc, char** argv) {
    int numProbs = (argc-1)/2;
    Problem** jobs = new Problem*[numProbs];
    for (int i=0; i<numProbs; i++){
        jobs[i] = new Problem(argv[2*i+1], std::strtol(argv[2*i+2]));
    }
    int currProb = 0;
    MPI_Status stat, stat2;
    bool all_dead = false;
    int ncores[nSlaves], state_split[nSlaves], gpu_usage[nSlaves];
    float r[nSlaves]; //relative processing power
    float l[nSlaves]; //average load
    int numStates[nSlaves]; //states being handled by each GPU
    int total_cores=0;
    for (size_t i = 0; i < nSlaves; i++){
        MPI_Recv(ncores+i, 1, MPI_INT, i+1, TAG_RESOURCE_DATA, MPI_COMM_WORLD, &stat);
        total_cores += ncores[i];
    }
    for (size_t i = 0; i < nSlaves; i++) {
      r[i] = float(ncores[i])/total_cores;
    }
    while (currProb<numProbs || !all_dead) {

        // Wait for any incomming message
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

        // Store rank of receiver into slave_rank
        int slave_rank = stat.MPI_SOURCE;

        // Decide according to the tag which type of message we have got
        if (stat.MPI_TAG == TAG_ASK_FOR_JOB && slave_rank==1) {
            if(currProb==0){
                MPI_Recv(nullptr, 0, MPI_INT, slave_rank, TAG_ASK_FOR_JOB, MPI_COMM_WORLD, &stat2);
            }else{
              jobs[currProb-1].pi_opt = new int[jobs[currProb-1].nStates];
                MPI_Recv(jobs[currProb-1].pi_opt, jobs[currProb-1].nStates, MPI_INT, slave_rank, TAG_ASK_FOR_JOB, MPI_COMM_WORLD, &stat2);
                // std::cout << "Optimal POlicy for " << jobs[currProb-1].file << ":\n";
                // char actions[]{'L','U','R','D'};
                // for (size_t i = 0; i < jobs[currProb-1].h; i++) {
                //     for (size_t j = 0; j < jobs[currProb-1].w; j++) {
                //         int idx = i*(jobs[currProb-1].w)+j;
                //         std::cout << actions[jobs[currProb-1].pi_opt[idx]] << ' ';
                //     }
                //     std::cout << '\n';
                // }
                delete [] jobs[currProb-1].pi_opt;
            }
            if (currProb<numProbs) {
                // here we have unprocessed jobs , we send the job to slaves
                for (size_t i = 0; i < nSlaves; i++) {
                    int job_size = jobs[currProb].file.size();
                    char *job = new char[job_size + 1];
                    std::memcpy( job, jobs[currProb].file.c_str(), job_size );
                    job[job_size] = '\0';
                    MPI_Send(job, job_size+1, MPI_CHAR, i+1, TAG_JOB_DATA, MPI_COMM_WORLD);
                    delete [] job;
                }
                currProb++;
            } else {
                // send stop msg to slaves
                for (size_t i = 0; i < nSlaves; i++) {
                    MPI_Send(nullptr, 0, MPI_INT, i+1 , TAG_STOP , MPI_COMM_WORLD);
                }
                all_dead = true;
            }
        } else {  //stat.MPI_TAG = TAG_RESOURCE_DATA
            int total_use = 0;
            for (size_t i = 0; i < nSlaves; i++){
                MPI_Recv(gpu_usage+i, 1, MPI_INT, i+1, TAG_RESOURCE_DATA, MPI_COMM_WORLD, &stat2);
                total_use += gpu_usage[i];
            }
            if (total_use == 0) total_use = 1;
            float den = 0;
            for (size_t i = 0; i < nSlaves; i++) {
                l[i] = float(gpu_usage[i])/total_use;
                if (l[i]==0) l[i]=0.01; //prevent divide-by-zero
                den +=r[i]/l[i];
            }
            int temp_total = 0;
            for (size_t i = 0; i < nSlaves-1; i++) {
                numStates[i] = (int)(r[i]*jobs[currProb-1].nStates/(l[i]*den));
                temp_total+=numStates[i];
            }
            numStates[nSlaves-1] = jobs[currProb-1].nStates-temp_total;
            for (size_t i = 0; i < nSlaves; i++) {
              MPI_Send(numStates, nSlaves, MPI_INT, i+1, TAG_RESOURCE_DATA, MPI_COMM_WORLD);
            }
        }
    }
    for (int i=0; i<numProbs; i++){
        delete jobs[i];
    }
    delete [] jobs;
}

void slave (MPI_Comm slaves) {
    int rank, size, ncores;
    MPI_Comm_rank(slaves, &rank);
    MPI_Comm_size(slaves, &size);
    int stopped = 0;
    std::string filename;
    MPI_Status stat , stat2 ;
    ncores = get_gpu_cores();
    MPI_Send(&ncores, 1, MPI_INT, 0, TAG_RESOURCE_DATA, MPI_COMM_WORLD);

    if(rank == 0){
        MPI_Send (nullptr, 0, MPI_INT, 0, TAG_ASK_FOR_JOB , MPI_COMM_WORLD) ;
    }
    do {
        // Here we send a message to the master asking for a job
        MPI_Probe (0 ,MPI_ANY_TAG , MPI_COMM_WORLD , & stat ) ;
        if ( stat.MPI_TAG == TAG_JOB_DATA ) {
            int n;
            MPI_Get_count(&stat, MPI_CHAR, &n);
            char filename_tmp[n];
            // Retrieve job data from master into msg_buffer
            MPI_Recv ( filename_tmp, n, MPI_CHAR, 0, TAG_JOB_DATA , MPI_COMM_WORLD , & stat2 ) ;

            MDPSolveObject *mdpObj = new MDPSolveObject();
            mdpObj->MDP = new mdp();
            mdpObj->MDP->load_raw_mdp(filename_tmp);
            mdpObj->MDP->initialize();
            mdpObj->MDP->horizon = 1000;
            mdpObj->MDP->epsilon = 0.001;

            mdpObj->algorithm = new MDPValueIterationGPU(mdpObj->MDP, 1024, (float *)nullptr, size, rank, slaves);
            value_function policy = mdpObj->algorithm->solve();

            // Collecting the policy from all clusters
            int *pi_final = new int[mdpObj->MDP->n];
            int *displs;
            int *listStates;
            int numStates;

            mdpObj->algorithm->getStats(&numStates, &displs, &listStates);
            MPI_Gatherv(policy.pi, numStates, MPI_INT, pi_final, listStates, displs, MPI_INT, 0, slaves);
            // send result to master
            if (rank == 0) {
                MPI_Send ( pi_final, mdpObj->MDP->n, MPI_INT, 0, TAG_ASK_FOR_JOB, MPI_COMM_WORLD);
            }
            delete mdpObj;
            delete [] pi_final;
        } else {
            // We got a stop message we have to retrieve it by using MPI_Recv
            MPI_Recv (nullptr, 0, MPI_INT, 0, TAG_STOP , MPI_COMM_WORLD , &stat2);
            stopped = 1;
        }
    } while (stopped == 0);
}


int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc , & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm slaves;
    MPI_Comm_split( MPI_COMM_WORLD, ( rank == 0 ), rank, &slaves );
    std::thread t(read_tegrastats);
    if (rank == 0) {
        master (size-1, argc, argv) ;
    } else {
        slave (slaves) ;
    }
    stop_gpu_stats();
    t.join();
    MPI_Comm_free(&slaves);
    MPI_Finalize () ;
}
