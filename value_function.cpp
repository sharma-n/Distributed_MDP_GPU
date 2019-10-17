#include "value_function.h"
#include <stdio.h>

int value_function::initialize(unsigned int in_n, unsigned int in_m, unsigned int numStates){
  if (in_n == 0 || in_m == 0) {
      fprintf(stderr, "Error[mdp_value_function_initialize]: %s\n", "Invalid input.");
      return -1;
  }

  n = in_n;
  m = in_m;

  S = nullptr;
  V = new float[n];
  pi = new unsigned int[numStates];

  return 0;
}

int value_function::uninitialize(){

  n = 0;
  m = 0;

  if (S != nullptr) {
      delete [] S;
  }
  S = nullptr;

  if (V != nullptr) {
      delete [] V;
  }
  V = nullptr;

  if (pi != nullptr) {
      delete [] pi;
  }
  pi = nullptr;
  return 0;
}
