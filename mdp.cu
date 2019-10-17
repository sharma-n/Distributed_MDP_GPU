#include "mdp.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
using namespace std;

int mdp::mdp_initialize_successors_gpu(){
  if(n == 0 || m == 0 || ns == 0 || S == nullptr){
    fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n", "Invalid input.");
    return -1;
  }

  // Allocate the memory on the device.
  if (cudaMalloc(&d_S, n*m*ns * sizeof(int)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n",
              "Failed to allocate device-side memory for the successor states.");
      return -1;
  }

  // Copy the data from the host to the device.
  if (cudaMemcpy(d_S, S, n*m*ns * sizeof(int),
                  cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n",
              "Failed to copy memory from host to device for the successor states.");
      return -1;
  }

  return 0;
}
int mdp::mdp_uninitialize_successors_gpu(){
  if (d_S != nullptr) {
      if (cudaFree(d_S) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_uninitialize_successors_gpu]: %s\n",
                  "Failed to free device-side memory for the successor states.");
          return -1;
      }
  }
  d_S = nullptr;
  return 0;
}

int mdp::mdp_initialize_state_transitions_gpu(){
  // Ensure the data is valid.
  if (n == 0 || m == 0 || ns == 0 || T == nullptr) {
      fprintf(stderr, "Error[mdp_initialize_state_transitions_gpu]: %s\n", "Invalid input.");
      return -1;
  }

  // Allocate the memory on the device.
  if (cudaMalloc(&d_T, n*m*ns * sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_state_transitions_gpu]: %s\n",
              "Failed to allocate device-side memory for the state transitions.");
      return -1;
  }

  // Copy the data from the host to the device.
  if (cudaMemcpy(d_T, T, n*m*ns * sizeof(float),
                  cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[nova_mdp_pbvi_initialize_state_transitions]: %s\n",
              "Failed to copy memory from host to device for the state transitions.");
      return -1;
  }

  return 0;
}
int mdp::mdp_uninitialize_state_transitions_gpu(){
  if (d_T != nullptr) {
      if (cudaFree(d_T) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_uninitialize_state_transitions_gpu]: %s\n",
                  "Failed to free device-side memory for the state transitions.");
          return -1;
      }
  }
  d_T = nullptr;

  return 0;





}

int mdp::mdp_initialize_rewards_gpu(){
  // Ensure the data is valid.
  if (n == 0 || m == 0 || R == nullptr) {
      fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n", "Invalid input.");
      return -1;
  }

  // Allocate the memory on the device.
  if (cudaMalloc(&d_R, n*m* sizeof(float)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n",
              "Failed to allocate device-side memory for the rewards.");
      return -1;
  }

  // Copy the data from the host to the device.
  if (cudaMemcpy(d_R, R, n*m * sizeof(float),
                  cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n",
              "Failed to copy memory from host to device for the rewards.");
      return -1;
  }

  return 0;
}
int mdp::mdp_uninitialize_rewards_gpu(){
  if (d_R != nullptr) {
      if (cudaFree(d_R) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_uninitialize_rewards_gpu]: %s\n",
                  "Failed to free device-side memory for the rewards.");
          return -1;
      }
  }
  d_R = nullptr;

  return 0;
}

int mdp::mdp_initialize_goals_gpu(){
  // Ensure the data is valid.
  if (ng == 0 || goals == nullptr) {
      fprintf(stderr, "Error[mdp_initialize_goals_gpu]: %s\n", "Invalid input.");
      return -1;
  }

  // Allocate the memory on the device.
  if (cudaMalloc(&d_goals, ng * sizeof(unsigned int)) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_goals_gpu]: %s\n",
              "Failed to allocate device-side memory for the goals.");
      return -1;
  }

  // Copy the data from the host to the device.
  if (cudaMemcpy(d_goals, goals, ng * sizeof(unsigned int),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Error[mdp_initialize_goals_gpu]: %s\n",
              "Failed to copy memory from host to device for the goals.");
      return -1;
  }

  return 0;
}
int mdp::mdp_uninitialize_goals_gpu(){
  if (d_goals != nullptr) {
      if (cudaFree(d_goals) != cudaSuccess) {
          fprintf(stderr, "Error[mdp_uninitialize_goals_gpu]: %s\n",
                  "Failed to free device-side memory for the goals.");
          return -1;
      }
  }
  d_goals = nullptr;

  return 0;
}

void mdp::split_string(string s, int *arr, size_t n, size_t offset){
  if(n==1){
    arr[offset+0] = stoi(s);
    return;
  }
  istringstream tokenStream(s);
  string token;
  for (size_t i = 0; i < n; i++) {
    getline(tokenStream, token, ',');
    arr[offset+i] = stoi(token);
  }
  return;
}
void mdp::split_string(string s, float *arr, size_t n, size_t offset){
  if(n==1){
    arr[offset+0] = stof(s);
    return;
  }
  istringstream tokenStream(s);
  string token;
  for (size_t i = 0; i < n; i++) {
    getline(tokenStream, token, ',');
    arr[offset+i] = stof(token);
  }
  return;
}

int mdp::load_raw_mdp(std::string filename){
    return load_raw_mdp(filename.c_str());
}

int mdp::load_raw_mdp(char *filename){
  uninitialize();

  ifstream raw_file(filename);
  string token;
  getline(raw_file, token, ',');
  n = stoi(token);
  getline(raw_file, token, ',');
  ns = stoi(token);
  getline(raw_file, token, ',');
  m = stoi(token);
  getline(raw_file, token, ',');
  k = stoi(token);
  getline(raw_file, token, ',');
  s0 = stoi(token);
  getline(raw_file, token, ',');
  ng = stoi(token);
  getline(raw_file, token, ',');
  horizon = stoi(token);
  getline(raw_file, token);
  gamma = stof(token);

  getline(raw_file, token);
  if(ng>0){
    goals = new int[ng];
    split_string(token, goals, ng, 0);
  }

  S = new int[n*m*ns];
  for (size_t a = 0; a < m; a++) {
      for (size_t s = 0; s < n; s++) {
        getline(raw_file, token);
        split_string(token, S, ns, (s*m*ns)+(a*ns));
      }
  }

  T = new float[n*m*ns];
  for (size_t a = 0; a < m; a++) {
      for (size_t s = 0; s < n; s++) {
        getline(raw_file, token);
        split_string(token, T, ns, (s*m*ns)+(a*ns));
      }
  }

  R = new float[n*m];
  for (size_t a = 0; a < m; a++) {
    getline(raw_file, token);
    istringstream tokenStream(token);
    string token2;
    for (size_t s = 0; s < n; s++) {
      getline(tokenStream, token2, ',');
      R[s*m+a] = stof(token2);
    }
  }

  Rmin = *min_element(R, R+n*m);
  Rmax = *max_element(R, R+n*m);
  raw_file.close();
  cpuIsInitialized = true;
  return 0;
}

int mdp::initialize(){
  if(!cpuIsInitialized){
    fprintf(stderr, "Error[initialize]: %s\n",
            "No MDP loaded. Use load_raw_mdp().");
    return -1;
  }
  if(gpuIsInitialized)
    return 0;
  int result = 0;
  result += mdp_initialize_successors_gpu();
  result += mdp_initialize_state_transitions_gpu();
  result += mdp_initialize_rewards_gpu();
  if(ng>0)
    result += mdp_initialize_goals_gpu();
  gpuIsInitialized = true;
  return result;
}

int mdp::uninitialize(){
  if(!cpuIsInitialized)
    return 0;
  int result = 0;
  if (gpuIsInitialized){
    result += mdp_uninitialize_successors_gpu();
    result += mdp_uninitialize_state_transitions_gpu();
    result += mdp_uninitialize_rewards_gpu();
    if(ng>0)
      result += mdp_uninitialize_goals_gpu();
    gpuIsInitialized = false;
  }
  n=0;ns=0;m=0;gamma=0.0;horizon=0;epsilon=0.0;s0=0;ng=0;
  if(goals != nullptr)
    delete [] goals;
  goals = nullptr;

  if(S != nullptr)
    delete [] S;
  S = nullptr;

  if(T != nullptr)
    delete [] T;
  T = nullptr;

  if(R != nullptr)
    delete [] R;
  R = nullptr;

  cpuIsInitialized = false;
  return result;
}
//
// void print_3d(int *a, size_t n, size_t m, size_t ns) {
//   for (size_t i = 0; i < n; i++) {
//     for (size_t j = 0; j < m; j++) {
//       for (size_t k = 0; k < ns; k++) {
//         std::cout << a[i*(n-1) + j*(m-1) + k] << ' ';
//       }
//       std::cout << '\n';
//     }
//     std::cout << '\n';
//   }
// }
// void print_3d(float *a, size_t n, size_t m, size_t ns) {
//   for (size_t i = 0; i < n; i++) {
//     for (size_t j = 0; j < m; j++) {
//       for (size_t k = 0; k < ns; k++) {
//         std::cout << a[i*(n-1) + j*(m-1) + k] << ' ';
//       }
//       std::cout << '\n';
//     }
//     std::cout << '\n';
//   }
// }
