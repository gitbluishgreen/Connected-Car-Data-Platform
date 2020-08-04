//this file contains all kernels used in the program.
#ifndef KERNEL_H
#define KERNEL_H
#include "proj_types.h"
#include <cuda.h>
__global__ void changeRowsKernel(int numberOfRowsToBeModified,Schema* deviceRowsToBeModified, Schema* StateDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRowsToBeModified)
    {
        StateDatabase[deviceRowsToBeModified[id].database_index] = deviceRowsToBeModified[id];
    }
}

__global__ void selectKernel(
    int* StateDatabase,
    int numberOfRows,
    int numberOfAttributes,
    int* selectedValues,
    int selectionCol,
    int conditionCol,
    int conditionValue,
    int* endIndexSelectedValues
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfAttributes && StateDatabase[id*numberOfAttributes+conditionCol] == conditionValue){
        int i = atomicAdd(endIndexSelectedValues, 1);
        selectedValues[i] = StateDatabase[id*numberOfAttributes+selectionCol];
    }
}
__global__ void DropVerticesKernel(int numberOfVertices,int numberOfDroppedVertices,int* deviceAdjacencyMatrix,int* deviceDroppedVertices)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/(numberOfVertices);
    int j = id % numberOfVertices;
    if(id < numberOfDroppedVertices*numberOfVertices)
        atomicAnd(deviceAdjacencyMatrix + numberOfVertices*deviceDroppedVertices[i] + j,INT_MAX);
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
