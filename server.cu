#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <signal.h>
#include <mutex> //to forbid concurrent reads and writes.
#include <thread>//4 threads: one listener,one moderator,one query API and one message resolver. 
#include <fcntl.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include "Expression.tab.cuh" //to parse any input queries.
#include "proj_types.cuh" //types
SelectQuery* process_query(std::string);
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
        char* x;
        cudaMallocHost((void**)&x,11*sizeof(char));
        x[0] = 'v';
        x[1] = 'e';
        x[2] = 'h';
        x[3] = 'i';
        x[4] = 'c';
        x[5] = 'l';
        x[6] = 'e';
        x[7] = '_';
        x[8] = 'i';
        x[9] = 'd';
        s->select_columns = new std::vector<char*>();
        s->distinct_query = true;//suffices to set this flag, now compare values of these columns.
        s->select_columns->push_back(x);//select columns will be used by the comparator to display.
        std::set<Schema,select_comparator> sx = t->select(s);//may have to be changed.
        /*WARNING: Possible deadlock here, (if worklist is writing after acquiring lock, and signal comes in, deadlock occurs 
        between listener thread and action thread, since an arbitrary thread handles the signal.)*/
        cudaFreeHost(s);
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
void display(ExpressionNode* curr_node,ExpressionNode* parent)
{
    if(curr_node == NULL)
        return;
    std::cout<<curr_node<<":\n";
    if(curr_node->exp_operator != NULL)
        std::cout<<"Opcode: "<<curr_node->exp_operator<<'\n';
    if(curr_node->column_name != NULL)
        std::cout<<"Column name: "<<curr_node->column_name<<'\n';
    std::cout<<"Value,Type,Parent "<<curr_node->value<<" "<<curr_node->type_of_expr<<" "<<parent<<'\n';
    display(curr_node->left_hand_term,curr_node);
    display(curr_node->right_hand_term,curr_node);
    
}

void show(SelectQuery* sq)
{
    if(sq == NULL)
    {
        std::cout<<"NULL, so can't print!\n";
    }       
    else
    {
        if(sq->select_columns != NULL)
            for(auto it: *(sq->select_columns))
                std::cout<<it<<' ';
        if(sq->aggregate_columns != NULL)
            for(auto it: *(sq->aggregate_columns))
                std::cout<<"("<<it.first<<" "<<it.second<<"),";
        //std::cout<<'\n';
        std::cout<<sq->limit_term<<'\n';
        display(sq->select_expression,NULL);
        // Schema s1;
        // s1.vehicle_id = 100;
        // printf("Came here for one query and it is %p!\n",sq->select_expression);
        // if(sq->select_expression != NULL)
        // {
        //     printf("Res is %d\n",sq->select_expression->evaluate_bool_expression(s1));
        // }
    }
}


void query_resolver()
{
    std::string s;
    std::ifstream inp;
    inp.open("query.txt",std::ifstream::in);
    while(std::getline(inp,s))
    {
        if(s == "KILL")
            break;
        SelectQuery* sq = process_query(s);
        //std::cout<<"Query:\n";
        show(sq);
        std::set<Schema,select_comparator> res = t->select(sq);
        std::cout<<"Return value is "<<res.size()<<'\n';
        t->PrintDatabase();
    }
    inp.close();
}

int main(int argc, char* argv[])
{
    sa.sa_sigaction = &message_handler;
    sa.sa_flags |= SA_SIGINFO;
    if(sigaction(SIGUSR1,&sa,NULL) != 0)
    {
        std::cout<<"Error while initializing the signal handler!\n";
    }
    int max_wl_size;
    std::ifstream input_file;
    input_file.open(argv[1],std::ifstream::in);
    input_file >>numberOfRows>>max_wl_size>>numberOfCars>>numberOfVertices;
    t = new Table(numberOfRows,numberOfCars,max_wl_size);
    int f[2];
    pipe(f);
    int fd = shm_open("adjacency_matrix",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*numberOfVertices*sizeof(int));
    int* hostAdjacencyMatrix = (int*)mmap(0,numberOfVertices*numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0; i < numberOfVertices; i ++){
        for(int j = 0; j < numberOfVertices; j ++){
            input_file >> hostAdjacencyMatrix[i*numberOfVertices+j];
            if(hostAdjacencyMatrix[i*numberOfVertices+j] < 0) hostAdjacencyMatrix[i*numberOfVertices+j] = INT_MAX;
        }
    }
    car_map = new std::map<int,int>();
    fd = shm_open("vertex_type",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*sizeof(int));//each node has an associated type. 
    int* type_array = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0;i < numberOfVertices;i++)
        input_file >> type_array[i];//1 for normal, 2 for fuel station, 3 for garage
    gps_object = new GPSSystem(numberOfVertices, hostAdjacencyMatrix);
    input_file.close();
    std::thread t1(initialize,numberOfCars,numberOfVertices,f,car_map);//creates and runs the cars.
    std::thread t2(listener,f);//listens for server messages.
    std::thread t3(query_resolver);
    t1.join();
    t2.join();
    t3.join();
    return 0;//end of program
}
