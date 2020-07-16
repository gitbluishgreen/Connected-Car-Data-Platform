#include<bits/stdc++.h>
#include<cuda.h>
#include<thrust/device_vector.h>

using namespace std;


int nt, nb;
void init_bt(int val){
    // initialize the num thread blks and grids:
    if(val <= 1024){
        nt = val;
        nb = 1;
    }
    else{
        nt = 1024;
        nb = (val+1024-1)/1024;
    }
}

__global__ void setRowMapKernel(
    int numberOfRows,
    int numberOfRowsToBeModified,
    int numberOfAttributes,
    int primaryKey,
    int* deviceMapToRows,
    int* deviceRowsToBeModified,
    int* deviceDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int j = id/numberOfRows;
    int i = id%numberOfRows;
    if(j < numberOfRowsToBeModified && deviceDatabase[numberOfAttributes*i+primaryKey] == deviceRowsToBeModified[numberOfAttributes*j+primaryKey]){
        deviceMapToRows[j] = i;
    }
}

__global__ void changeRowsKernel(
    int numberOfRowsToBeModified,
    int numberOfAttributes,
    int* deviceMapToRows,
    int* deviceRowsToBeModified, 
    int* deviceDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int j = id/numberOfRowsToBeModified;
    int k = id%numberOfRowsToBeModified;
    if(j < numberOfAttributes){
        int i = deviceMapToRows[k];
        deviceDatabase[i*numberOfAttributes+j] = deviceRowsToBeModified[k*numberOfAttributes+j];
    }
}


class DatabaseManagementSystem{
private:
    int* deviceDatabase;
    const int numberOfAttributes;
    const int numberOfRows;
    const int primaryKey;
    map<string, int> attributesToCol;
public:

    DatabaseManagementSystem(
        int numAtt,
        int numRows, 
        int pKey,
        vector<string> attList, 
        int* initTable
    ):
        numberOfAttributes(numAtt),
        numberOfRows(numRows),
        primaryKey(pKey)
    {
        for(int i = 0; i < numberOfAttributes; i ++) attributesToCol[attList[i]] = i;
        cudaMalloc(&deviceDatabase, numberOfAttributes*numberOfRows*sizeof(int));
        cudaMemcpy(deviceDatabase, initTable, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyHostToDevice);        
    }

    
    void WriteRows(int numberOfRowsToBeModified, int* hostRowsToBeModified){
        // Find the row numbers to be modified in the database
        // This is done parallely, using the fact that the primary key of each row
        // in the argument rowsToBeModified uniquely defines a row in the actual database.
        int* deviceRowsToBeModified;
        int* deviceMapToRows;
        cudaMalloc(&deviceRowsToBeModified, numberOfRowsToBeModified*numberOfAttributes*sizeof(int));
        cudaMalloc(&deviceMapToRows, numberOfRowsToBeModified*sizeof(int));
        cudaMemcpy(deviceRowsToBeModified, hostRowsToBeModified, numberOfRowsToBeModified*numberOfAttributes*sizeof(int), cudaMemcpyHostToDevice);
        
        cout << "Starting map compute" << endl;
        init_bt(numberOfRows*numberOfRowsToBeModified);
        setRowMapKernel<<<nb, nt>>>(
            numberOfRows,
            numberOfRowsToBeModified,
            numberOfAttributes,
            primaryKey,
            deviceMapToRows,
            deviceRowsToBeModified,
            deviceDatabase
        );
        cudaDeviceSynchronize();
        cout << "Map Computed!!" << endl;
        // The next task, using the row numbers acquired with the above kernel, 
        // fire numberOfAttributes*numRowsToBeModified threads to modify the cells 
        // of the actual database
        init_bt(numberOfRowsToBeModified*numberOfAttributes);
        changeRowsKernel<<<nb, nt>>>(
            numberOfRowsToBeModified,
            numberOfAttributes,
            deviceMapToRows,
            deviceRowsToBeModified, 
            deviceDatabase 
        );
        cudaDeviceSynchronize();
        cout << "Write Completed!!" << endl;
    }

    void PrintDatabase(){
        int* hostDatabase = new int[numberOfAttributes*numberOfRows];
        cudaMemcpy(hostDatabase, deviceDatabase, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < numberOfRows; i ++){
            for(int j = 0; j < numberOfAttributes; j ++){
                cout << hostDatabase[i*numberOfAttributes+j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};

int main(){
    int numberOfRows, numberOfAttributes, pKey;
    cin >> numberOfRows >> numberOfAttributes >> pKey;
    vector<string> attList(numberOfAttributes);
    for(int i = 0; i < numberOfAttributes; i ++){
        cin >> attList[i];
    }
    int* initTable = new int[numberOfRows*numberOfAttributes];
    for(int i = 0; i < numberOfRows; i ++){
        for(int j = 0; j < numberOfAttributes; j ++){
            cin >> initTable[i*numberOfAttributes+j];
        }
    }
    DatabaseManagementSystem dbms(
        numberOfAttributes,
        numberOfRows,
        pKey,
        attList, 
        initTable
    );

    dbms.PrintDatabase();

    int q;
    cin >> q;
    while(q--){
        int numberOfRowsToBeModified;
        cin >> numberOfRowsToBeModified;
        int* hostRowsToBeModified = new int[numberOfRowsToBeModified*numberOfAttributes];
        for(int i = 0; i < numberOfRowsToBeModified; i ++){
            for(int j = 0; j < numberOfAttributes; j ++){
                cin >> hostRowsToBeModified[i*numberOfAttributes+j];
            }
        }
        dbms.WriteRows(
            numberOfRowsToBeModified,
            hostRowsToBeModified
        );
        dbms.PrintDatabase();
    }
    return 0;
}