#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <chrono>
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

void show_normal_query(const std::vector<Schema>& selected_rows,SelectQuery* select_query)
{
    for(Schema s: selected_rows)
    {
        std::cout<<"{";
        for(char* it: *(select_query->select_columns))
        {
            if(str_equal(it,"vehicle_id"))
                std::cout<<"vehicle_id:"<<s.vehicle_id<<",";
            if(str_equal(it,"oil_life_pct"))
                std::cout<<"oil_life_pct:"<<s.oil_life_pct<<",";
            if(str_equal(it,"tire_p_rl"))
                std::cout<<"tire_p_rl:"<<s.tire_p_rl<<",";
            if(str_equal(it,"tire_p_rr"))
                std::cout<<"tire_p_rr:"<<s.tire_p_rr<<",";
            if(str_equal(it,"tire_p_fl"))
                std::cout<<"tire_p_fl:"<<s.tire_p_fl<<",";
            if(str_equal(it,"tire_p_fr"))
                std::cout<<"tire_p_fr:"<<s.tire_p_fr<<",";
            if(str_equal(it,"batt_volt"))
                std::cout<<"batt_volt:"<<s.batt_volt<<",";
            if(str_equal(it,"fuel_percentage"))
                std::cout<<"fuel_percentage:"<<s.fuel_percentage<<",";
            if(str_equal(it,"accel"))
                std::cout<<"accel:"<<s.accel<<",";
            if(str_equal(it,"seatbelt"))
                std::cout<<"seatbelt:"<<s.seatbelt<<",";
            if(str_equal(it,"hard_brake"))
                std::cout<<"hard_brake:"<<s.hard_brake<<",";
            if(str_equal(it,"door_lock"))
                std::cout<<"door_lock:"<<s.door_lock<<",";
            if(str_equal(it,"clutch"))
                std::cout<<"clutch:"<<s.clutch<<",";
            if(str_equal(it,"hard_steer"))
                std::cout<<"hard_steer:"<<s.hard_steer<<",";
            if(str_equal(it,"speed"))
                std::cout<<"speed:"<<s.speed<<",";
            if(str_equal(it,"distance"))
                std::cout<<"distance:"<<s.distance<<",";
            if(str_equal(it,"origin_vertex"))
                std::cout<<"origin_vertex:"<<s.origin_vertex<<",";
            if(str_equal(it,"destination_vertex"))
                std::cout<<"destination_vertex:"<<s.destination_vertex<<",";
        }
        std::cout<<"}\n";
    }
}

void show_aggregate_query(const std::pair<std::vector<std::vector<std::pair<double,double>>>,std::vector<std::string>>& v,SelectQuery* select_query)
{
    //std::cout<<"Hey,got here! "<<v.second.size()<<'\n';
    int i = 0,j=0;
    for(i=0;i<v.first.size();i++)
    {
        std::cout<<v.second[i]<<":\t{";
        int sz = v.first[i].size();
        for(j=0;j<sz-1;j++)
            std::cout<<"("<<v.first[i][j].first<<","<<v.first[i][j].second<<"),";
        std::cout<<"("<<v.first[i][sz-1].first<<","<<v.first[i][sz-1].second<<")}\n";
    }
}

void show(SelectQuery* sq)
{
    if(sq->select_columns != NULL)
        for(auto it: *(sq->select_columns))
            std::cout<<it<<' ';
    if(sq->aggregate_columns != NULL)
        for(auto it: *(sq->aggregate_columns)){
            std::cout<<"("<<it.first<<" "<<it.second<<"),";
            std::cout<<'\n';
        }
    std::cout<<sq->limit_term<<'\n';
    display(sq->select_expression,NULL);
}

void request_resolver(int* file_descriptor)
{
    request_body* rb = new request_body(0,0,0);
    while(read(file_descriptor[0],rb,sizeof(request_body)))
    {
    //fuel routing or garage
        //std::cout<<"Resolving garage request from "<<rb->sending_car<<'\n';
        int fd = shm_open("vertex_type",O_RDONLY,0666);
        ftruncate(fd,numberOfVertices*sizeof(int));//each node has an associated type. 
        int* type_array = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
        int sending_car = rb->sending_car;//this car's pid. 
        std::map<int,int> current_position = t->get_latest_position();
        int curr_pos = current_position[sending_car];
        std::set<int> dropped_vertices;
        for(int i = 0;i < numberOfVertices;i++)
        {
            double y = (double)rand();
            if(y/RAND_MAX < 0.5 && type_array[i] == 1 && i != curr_pos)//adjust manually
                dropped_vertices.insert(i);//have to ensure that this does not have current position of the car itself!
        }
        close(fd);
        //dropped vertices haveto be computed randomly. 
        std::vector<int> path = gps_object->findGarageOrBunk(curr_pos,rb->request_type,dropped_vertices);
        char c[20];
        // std::cout<<"Path for "<<rb->sending_car<<": ";
        // for(auto it: path)
        //     std::cout<<it<<' ';
        // std::cout<<'\n';
        if(path.size() <= 1)
            continue;//don't send.
        sprintf(c,"shm_1_%d",sending_car);
        fd = shm_open(c,O_CREAT | O_RDWR,0666);
        ftruncate(fd,4);
        int* ptr = (int*)mmap(0,sizeof(int),PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = 1;
        close(fd);
        c[4] = '2';
        fd = shm_open(c,O_CREAT | O_RDWR,0666);
        ftruncate(fd,4);
        ptr = (int*)mmap(0,sizeof(int),PROT_READ | PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = rb->anomaly_flag;
        close(fd);
        c[4] = '3';
        fd = shm_open(c,O_CREAT | O_RDWR, 0666);
        ftruncate(fd,4);
        ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = path.size();
        close(fd);
        c[4] = '4';
        fd = shm_open(c,O_CREAT|O_RDWR,0666);
        ftruncate(fd,sizeof(int)*path.size());
        ptr = (int*)mmap(0,path.size(),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        for(int i = 0;i < path.size();i++)
            ptr[i] = path[i];
        close(fd);
        kill(sending_car,SIGUSR1);//send the updated path back to the end user.
    }
}

void query_resolver(int* file_descriptor)//pipe to write to request resolver
{
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::string s;
    std::ifstream inp;
    inp.open("query.txt",std::ifstream::in);
    while(std::getline(inp,s))
    {
        if(s == "KILL")
            break;
        else if(s.find("CONVOY",0) == 0)
        {
            //convoy query. Else, it's a select query.
            std::stringstream inp_q(s);
            inp_q>>s;//remove the word convoy
            int car_cnt;
            inp_q>>car_cnt;
            bool fl = false;
            std::vector<int> v;
            if(car_cnt <= 0)
                fl = true;
            else
            {
                v.resize(car_cnt);
            }
            for(int i = 0;i < car_cnt;i++)
            {
                int y;
                inp_q>>y;
                if(y <= 0 || y > numberOfCars)
                {
                    fl = true;
                    break;
                }
                v[i] = y;
            }
            if(fl)
                continue;//rejected query.
            
            //convoy request. Proceed to read the set of cars in the convoy, and resolve the same.
            std::cout<<"Convoy query!\n";
            std::map<int,int> returned_details = t->get_latest_position();//gets latest position of all cars. Lock needed?
            std::map<int,int> car_details;
            for(int it: v)
            {
                std::cout<<it<<'\n';
                int x = car_map->find(it)->second;
                car_details[x] = returned_details[x];
            }
            gps_object->convoyNodeFinder(car_details);//takes care of what is needed, including sending a signal to all.
        }
        else
        {
            SelectQuery* sq = process_query(s);
            if(sq == NULL){
                std::cout<<"Ill-formatted query!\n";
                continue;
            }
            //show(sq);
            if(sq->aggregate_columns != NULL)
            {
                std::pair<std::vector<std::vector<std::pair<double,double>>>,std::vector<std::string>> v = t->aggregate_select(sq);
                show_aggregate_query(v,sq);
            }
            else
            {
                std::vector<Schema> v = t->normal_select(sq);
                show_normal_query(v,sq);
            }
            //t->PrintDatabase();
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    inp.close();
}

int main(int argc, char* argv[])
{
    int max_wl_size;
    std::ifstream input_file;
    input_file.open(argv[1],std::ifstream::in);
    input_file >>numberOfRows>>max_wl_size>>numberOfCars>>numberOfVertices;
    //std::cout<<numberOfRows<<" "<<max_wl_size<<" "<<numberOfCars<<" "<<numberOfVertices<<'\n'; 
    int f[2];
    pipe(f);
    int request_fd[2];
    pipe(request_fd);
    t = new Table(numberOfRows,numberOfCars,max_wl_size,request_fd);
    int fd = shm_open("adjacency_matrix",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*numberOfVertices*sizeof(int));
    int* hostAdjacencyMatrix = (int*)mmap(0,numberOfVertices*numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0; i < numberOfVertices; i ++){
        for(int j = 0; j < numberOfVertices; j ++){
            input_file >> hostAdjacencyMatrix[i*numberOfVertices+j];
            if(hostAdjacencyMatrix[i*numberOfVertices+j] < 0) hostAdjacencyMatrix[i*numberOfVertices+j] = INT_MAX;
        }
    }
    close(fd);
    car_map = new std::map<int,int>();
    fd = shm_open("vertex_type",O_CREAT|O_RDWR,0666);
    ftruncate(fd,numberOfVertices*sizeof(int));//each node has an associated type. 
    int* type_array = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    for(int i = 0;i < numberOfVertices;i++)
        input_file >> type_array[i];//1 for normal, 2 for fuel station, 3 for garage
    close(fd);
    gps_object = new GPSSystem(numberOfVertices, hostAdjacencyMatrix);
    input_file.close();
    std::thread t1(initialize,numberOfCars,numberOfVertices,f,car_map);//creates and runs the cars.
    std::thread t2(listener,f);//listens for server messages.
    std::thread t3(query_resolver,request_fd);
    std::thread t4(request_resolver,request_fd);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    return 0;//end of program
}
