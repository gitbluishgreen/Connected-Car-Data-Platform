//this file contains all kernels used in the program.
#ifndef KERNEL_H
#define KERNEL_H
#include "proj_types.hpp"
#include <cuda.h>

__global__ void set_zero(int* a)
{
    a[blockIdx.x*blockDim.x + threadIdx.x] = 0;
}
__global__ void changeRowsKernel(int numberOfRowsToBeModified, int* indices_to_overwrite, Schema* deviceRowsToBeModified, Schema* Database
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRowsToBeModified)
    {
        StateDatabase[indices_to_overwrite[id]] = deviceRowsToBeModified[id];
    }
}

__global__ void selectKernel(
    Schema* StateDatabase,
    int numberOfRows,
    Schema* selectedValues,
    int* endIndexSelectedValues,
    SelectQuery* select_query
)
{
    //select logic with expression evaluation.
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRows && select_query->select_expression->evalute_bool_expression(StateDataBase[id]))
    {
        int i = atomicAdd(endIndexSelectedValues, 1);
        selectedValues[i] = StateDatabase[id];
    }
}
__global__ void DropVerticesKernel(int numberOfVertices,int numberOfDroppedVertices,int* deviceAdjacencyMatrix,int* deviceDroppedVertices)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/(numberOfVertices);
    int j = id % numberOfVertices;
    if(id < numberOfDroppedVertices*numberOfVertices)
        atomicOr(deviceAdjacencyMatrix + numberOfVertices*deviceDroppedVertices[i] + j,INT_MAX);
}
}

__global__ void FindMinDistance(int numberOfVertices,int* deviceUsed,int* deviceDistance,int* minDistance)
{   
    // printf("init mindist = %d\n", *minDistance);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id]){
        atomicMin(minDistance, deviceDistance[id]);
    }
}

__global__ void FindArgMin(int numberOfVertices,int* deviceUsed,int* deviceDistance,int* minDistance,int* argMinVertex)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id] && *minDistance == deviceDistance[id]){
        *argMinVertex = id;
    }
}

__global void addMatrix(int* a,int* b)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    a[ind]  += b[ind];
}

__global__ void Relax(
    int numberOfVertices,
    int* deviceAdjacencyMatrix,
    int* deviceUsed,
    int* deviceDistance,
    int* deviceParent,
    int* minDistance,
    int* argMinVertex
)
{
    deviceUsed[*argMinVertex] = 1;
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id] && deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id] != INT_MAX){
        // printf("argMinVertex = %d\n", *argMinVertex);
        if(deviceDistance[id] > deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id]+*minDistance){
            // printf("%d %d\n", deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id], *minDistance);
            deviceDistance[id] = deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id]+*minDistance;
            deviceParent[id] = *argMinVertex;
        }
    }
}
#endif
