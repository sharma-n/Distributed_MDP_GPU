// value_function.h
#ifndef value_function_H
#define value_function_H

class value_function
{
  private:
    unsigned int n;
    unsigned int m;
    unsigned int *S;

  public:
    float *V;
    unsigned int *pi;
    int initialize(unsigned int n, unsigned int m, unsigned int numStates);
    int uninitialize();

};

#endif
