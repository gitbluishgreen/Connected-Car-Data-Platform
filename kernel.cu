#include "kernel.cuh"
#include "proj_types.cuh"
__global__ void set_zero(double* a)
{
    a[blockIdx.x*blockDim.x + threadIdx.x] = 0;
}
__global__ void addMatrix(double* a,int* b)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    a[ind]  += b[ind];
}

__global__ void DropVerticesKernel(int numberOfVertices,int numberOfDroppedVertices,int* deviceAdjacencyMatrix,int* deviceDroppedVertices)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/(numberOfVertices);
    int j = id % numberOfVertices;
    if(id < numberOfDroppedVertices*numberOfVertices)
        atomicOr(deviceAdjacencyMatrix + numberOfVertices*deviceDroppedVertices[i] + j,INT_MAX);
}

__global__ void FindMinDistance(int numberOfVertices,int* deviceUsed,int* deviceDistance,int* minDistance)
{   
    // printf("init mindist = %d\n", *minDistance);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id])
        atomicMin(minDistance, deviceDistance[id]);
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
    int* argMinVertex)
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

__global__ void changeRowsKernel(int numberOfRowsToBeModified, int* indices_to_overwrite, Schema* deviceRowsToBeModified, Schema* Database
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRowsToBeModified)
    {
        Database[indices_to_overwrite[id]] = deviceRowsToBeModified[id];
    }
}

__global__ void selectKernel(
    Schema* StateDatabase,
    int numberOfRows,
    Schema* selectedValues,
    int* endIndexSelectedValues,
    SelectQuery* select_query)
{
    //select logic with expression evaluation.
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRows)
    {
        if(select_query->select_expression == NULL){
        int i = atomicAdd(endIndexSelectedValues, 1);
        selectedValues[i] = StateDatabase[id];}
        else if(select_query->select_expression->evaluate_bool_expression(StateDatabase[id]))
        {
            int i = atomicAdd(endIndexSelectedValues, 1);
            selectedValues[i] = StateDatabase[id];
        }
    }
}