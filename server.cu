#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cuda.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <signal.h>
#include <mutex> //to forbid concurrent reads and writes.
#include <thread>//2 threads: one listener and one moderator.
#include <fcntl.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include "Expression.tab.cuh" //to parse any input queries.
#include "proj_types.cuh"
void initialize(int,int,int*,std::map<int,int>*);
Table* t;
GPSSystem* gps_object;
int numberOfCars;
int numberOfVertices;
int numberOfRows;
std::map<int,int>* car_map;
static struct sigaction sa;
void message_handler(int sig, siginfo_t* sig_details,void* context)
{
    int fd = shm_open("shm_server_1",O_RDONLY,0666);
    int* ptr = (int*)mmap(0,4,PROT_READ,MAP_SHARED,fd,0);
    if(*ptr == 1)
    {
        //convoy request. Proceed to read the set of cars in the convoy, and resolve the same.
        fd = shm_open("shm_server_2",O_RDONLY,0666);
        ptr = (int*)mmap(0,sizeof(bool)*numberOfCars,PROT_READ,MAP_SHARED,fd,0);//list of cars participating.
        //now we need to call the convoy query to resolve the same.
        //add a call to get the position of all cars first and then see.
        std::set<int> participating_cars;
        for(int i = 0;i < numberOfCars;i++)
        {
            if(ptr[i])
                participating_cars.insert((*car_map)[i]);
        }
        SelectQuery* s;
        cudaMallocHost((void**)&s,sizeof(Schema));
        ExpressionNode* gt;
        cudaMallocHost((void**)&gt,sizeof(ExpressionNode));
        gt->column_name = "vehicle_id";
        gt->left_hand_term = gt->right_hand_term = NULL;
        s->group_term = gt;
        s->select_columns.push_back("vehicle_id");
        std::set<Schema,select_comparator> sx = t->select(s);
        /*WARNING: Possible deadlock here, (if worklist is writing after acquiring lock, and signal comes in, deadlock occurs 
        between listener thread and action thread, since an arbitrary thread handles the signal.)*/
        cudaFreeHost(s);
        cudaFreeHost(gt);
        std::map<int,int> car_details;
        for(Schema o: sx)
        {
            if(participating_cars.find(o.vehicle_id) != participating_cars.end())
                car_details[o.vehicle_id] = o.origin_vertex;
        }
        gps_object->convoyNodeFinder(car_details);//takes care of what is needed, including sending a signal to all.
    }
    else if(*ptr == 2 || *ptr == 3)
    {
        //fuel routing or garage
        int x = *ptr;
        int sending_car = sig_details->si_pid;
        std::set<int> dropped_vertices;
        for(int i = 0;i < numberOfVertices;i++)
        {
            double y = (double)rand();
            if(y/RAND_MAX < 0.5)//adjust manually
                dropped_vertices.insert(i);//have to ensure that this does not have current position of the car itself!
        }
        //dropped vertices haveto be computed randomly. 
        std::vector<int> path = gps_object->findGarageOrBunk(sending_car,x,dropped_vertices);
         /*WARNING: Possible deadlock here, (if worklist is writing after acquiring lock, and signal comes in, deadlock occurs 
        between listener thread and action thread, since an arbitrary thread handles the signal.)*/
        char c[20];
        sprintf(c,"shm_3_%d",sending_car);
        fd = shm_open(c,O_CREAT | O_RDWR, 0666);
        ftruncate(fd,4);
        ptr = (int*)mmap(0,4,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = path.size();
        sprintf(c,"shm_4_%d",sending_car);
        fd = shm_open(c,O_CREAT|O_RDWR,0666);
        ftruncate(fd,sizeof(int)*path.size());
        int* ptr = (int*)mmap(0,path.size(),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        for(int i = 0;i < path.size();i++)
            ptr[i] = path[i];
        kill(SIGUSR1,sending_car);//send the updated path back to the end user.
    }
}
// Aux function to print cuda error:
void cudaErr(){
    // Get last error:
    cudaError_t err = cudaGetLastError();
    printf("error=%d, %s, %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
}
void listener(int* fd)
{
    //listen on fd[0].
    Schema s(10);
    while(read(fd[0],&s,sizeof(Schema)))
    {
        t->update_worklist(s);
    }
}
int main()
{
    sa.sa_sigaction = &message_handler;
    sa.sa_flags |= SA_SIGINFO;
    if(sigaction(SIGUSR1,&sa,NULL) != 0)
    {
        std::cout<<"Error while initializing the signal handler!\n";
    }
    int max_wl_size;
    std::cin >>numberOfRows>>max_wl_size>>numberOfCars>>numberOfVertices;
    t = new Table(numberOfRows,numberOfCars,max_wl_size);
    int f[2];
    pipe(f);
    int fd = shm_open("adjacency_matrix",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*numberOfVertices*sizeof(int));
    int* hostAdjacencyMatrix = (int*)mmap(0,numberOfVertices*numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0; i < numberOfVertices; i ++){
        for(int j = 0; j < numberOfVertices; j ++){
            std::cin >> hostAdjacencyMatrix[i*numberOfVertices+j];
            if(hostAdjacencyMatrix[i*numberOfVertices+j] < 0) hostAdjacencyMatrix[i*numberOfVertices+j] = INT_MAX;
        }
    }
    car_map = new std::map<int,int>();
    fd = shm_open("vertex_type",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*sizeof(int));//each node has an associated type. 
    int* type_array = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0;i < numberOfVertices;i++)
        std::cin>>type_array[i];//1 for normal, 2 for fuel station, 3 for garage
    gps_object = new GPSSystem(numberOfVertices, hostAdjacencyMatrix);
    std::thread t1(initialize,numberOfCars,numberOfVertices,f,car_map);//creates and runs the cars.
    std::thread t2(listener,f);//listens for server messages.
    t1.join();
    std::cout<<"I've done my duty of creating cars. Now I'll stop.\n";
    t2.join();
    return 0;//end of program
}
