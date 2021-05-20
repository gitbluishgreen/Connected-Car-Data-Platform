#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>
#include "proj_types.cuh"
__global__ void set_zero(double*);
__global__ void addMatrix(double* a,int* b);
__global__ void DropVerticesKernel(int,int,int*,int*);
__global__ void FindMinDistance(int ,int*,int*,int*);
__global__ void FindArgMin(int,int*,int*,int*,int*);
__global__ void Relax(int,int*,int*,int*,int*,int*,int*);
__global__ void changeRowsKernel(int, int* ,Schema*, Schema*);
__global__ void selectKernel(Schema*,int,Schema*,int*, SelectQuery*);
#endif