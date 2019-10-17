// mdp.h
#ifndef mdp_H
#define mdp_H
#include <string>

class mdp
{
  private:
    bool cpuIsInitialized = false;
    bool gpuIsInitialized = false;
    int mdp_initialize_successors_gpu();
    int mdp_uninitialize_successors_gpu();
    int mdp_initialize_state_transitions_gpu();
    int mdp_uninitialize_state_transitions_gpu();
    int mdp_initialize_rewards_gpu();
    int mdp_uninitialize_rewards_gpu();
    int mdp_initialize_goals_gpu();
    int mdp_uninitialize_goals_gpu();
    void split_string(std::string s, int *arr, size_t n, size_t offset);
    void split_string(std::string s, float *arr, size_t n, size_t offset);

  public:
    unsigned int n=0,  //number of states
      ns=0,   //max number of successors
      m=0,    //number of actions
      k=0,    //number of objective(reward) functions
      s0=0,   //initial state index (if an SSP MDP)
      horizon=0,  //positive integer for horizon
      ng=0;   //number of goals (if an SSP MDP) or 0 (otherwise)
    float gamma=0.0, Rmin = 0.0, Rmax = 0.0, epsilon = 0.0;
    int *goals = nullptr;
    int *S = nullptr;   //state transition matrix
    float *T = nullptr; //state transition probabilites
    float *R = nullptr; //reward function matrix
    int *d_S = nullptr, *d_goals = nullptr; //device side matrices
    float *d_T = nullptr, *d_R = nullptr;

    int load_raw_mdp(char *filename);
    int load_raw_mdp(std::string filename);
    int initialize();
    int uninitialize();
};

#endif
