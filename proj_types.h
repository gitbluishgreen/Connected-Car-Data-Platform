//this file defines all types used in the programme.
#ifndef PROJ_TYPES
#define PROJ_TYPES
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <mutex>
#include <thread>
#include <set>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <limits.h>
#include <cuda.h>
#include <kernel.h>
class Schema
{
public:
    int vehicle_id;//the process ID of the car serves as the vehicle ID.
    int database_index;//fixed constant for mapping to the database purposes for anomaly detection.
    double oil_life_pct;//oil percentage
    double tire_p_rl;//tire pressures on each tire.
    double tire_p_rr;
    double tire_p_fl;
    double tire_p_fr;
    double batt_volt;
    double fuel_percentage;
    bool accel;
    bool seatbelt;
    bool hard_brake;
    bool door_lock;
    bool gear_toggle;
    bool clutch;
    bool hard_steer;
    double speed;
    double distance;
    int origin_vertex;
    int destination_vertex;
};
class ExpressionNode
{
    public:
        ExpressionNode* left_hand_term;
        std::string column_name;//column name of the expression
        double value;//value of scalar field
        std::string exp_operator;
        ExpressionNode* right_hand_term;
        int type_of_expr;//can be either 1,2 or 3, if bool/integer/floating point.
        ExpressionNode()
        {
            left_hand_term = right_hand_term = NULL;
        }
        ExpressionNode(std::string op)
        {
            left_hand_term = right_hand_term = NULL;
            exp_operator = op;
        }
        bool evaluate_int_expr(const Schema& s)
        {
            if(type_of_expr != 2)
                return false;
            if(left_hand_term == NULL)
                return 0;
            else
            {
                if(right_hand_term == NULL)
                {
                    if(column_name == "vehicle_id")
                        return s.vehicle_id;
                    else if(column_name == "origin_vertex")
                        return s.origin_vertex;
                    else if(column_name == "destination_vertex")
                        return s.destination_vertex;
                }
                else
                {
                    int x = left_hand_term->evaluate_int_expr(s);
                    int y = right_hand_term->evaluate_int_expr(s);
                    if(exp_operator == "plus")
                        return x+y;
                    else if(exp_operator == "minus")
                        return x-y;
                    else if(exp_operator == "mult")
                        return x*y;
                    else if(exp_operator == "div")
                        return x/y;
                    else if(exp_operator == "modulo")
                        return x%y;
                    else
                        return 0;
                    }
            }
        }
        double evaluate_double_expr(const Schema& s)
        {
            if(type_of_expr != 3)
                return 0.0;
            else if(right_hand_term == NULL)
            {
                if(column_name == "oil_life_pct")
                    return s.oil_life_pct;//this is on device memory.
                else if(column_name == "tire_p_rl")
                    return s.tire_p_rl;//this is on device memory.
                else if(column_name == "tire_p_rr")
                    return s.tire_p_rr;//this is on device memory.
                else if(column_name == "tire_p_fl")
                    return s.tire_p_fl;//this is on device memory.
                else if(column_name == "tire_p_fr")
                    return s.tire_p_fr;//this is on device memory.
                else if(column_name == "batt_volt")
                    return s.batt_volt;//this is on device memory.
                else if(column_name == "fuel_percentage")
                    return s.fuel_percentage;//this is on device memory.
                else if(column_name == "speed")
                    return s.speed;//this is on device memory.
                else if(column_name == "distance")
                    return s.distance;//this is on device memory.
                else
                    return value;//this is on pinned memory.
            }
            else
            {
                double a1 = left_hand_term->evaluate_double_expr(s);
                double a2 = right_hand_term->evaluate_double_expr(s);
                if(exp_operator == "plus")
                    return a1+a2;
                else if(exp_operator =="minus")
                    return a1-a2;
                else if(exp_operator =="mult")
                    return a1*a2;
                else if(exp_operator == "div")
                    return a1/a2;
                else 
                    return 0;
            }
        }
        bool evaluate_bool_expr(const Schema& s)
        {
            if(type_of_expr != 1)
                return false;
            else if(right_hand_term == NULL)
            {
                if(exp_operator == "not")
                    return !left_hand_term->evaluate_bool_expr(s);
                if(column_name == "accel")
                    return s.accel;
                else if(column_name == "seatbelt")
                    return s.seatbelt;
                else if(column_name == "hard_brake")
                    return s.hard_brake;
                else if(column_name == "door_lock")
                    return s.door_lock;
                else if(column_name == "gear_toggle")
                    return s.gear_toggle;
                else if(column_name == "clutch")
                    return s.clutch;
                else if(column_name == "hard_steer")
                    return s.hard_steer;
                else
                {
                    //read value column
                    bool res = (value == 0.0)?false:true;
                    return res;
                }
            }
            else
            {
                bool x = left_hand_term->evaluate_bool_expr(s);
                bool y = right_hand_term->evaluate_bool_expr(s);
                if(exp_operator == "Or")
                    return x|y;
                else if(exp_operator == "And")
                    return x&y;
                else
                    return false;
            }
        }
};

class SelectQuery
{
	public:
		bool distinct_query;
		std::vector<std::string> select_columns;//either this field or agg columns will be active.
        std::vector<std::pair<std::string,ExpressionNode*>> aggregate_columns;
		std::vector<std::pair<ExpressionNode*,bool>> order_term;
		ExpressionNode* select_expression;
        ExpressionNode* group_term;
		int limit_term;
		SelectQuery()
        {
            distinct_query = false;
            limit_term = -1;
        }
};

class select_comparator
{
    private:
        SelectQuery* select_query;//on host
    public:
        select_comparator(SelectQuery* sq)
        {
            select_query = sq;//pointer to a pinned memory location.
        }
        int operator ()(const Schema& s1,const Schema& s2)//all on host memory now.
        {
            if(select_query->group_term != NULL)
            {
                int v1 = select_query->group_term->type_of_expr;
                if(v1 == 2)
                {
                    /*WARNING: We are running the function on host here.*/
                    int a1  = select_query->group_term->evaluate_int_expr(s1);
                    int a2 = select_query->group_term->evaluate_int_expr(s2); 
                    if(a1 == a2)
                        return 0;//equal
                    else
                    {
                       if(select_query->order_term.size() != 0)
                        {
                            for(std::pair<ExpressionNode*,bool>& p: select_query->order_term)
                            {
                                if(p.first->type_of_expr == 2)
                                {
                                    int a11 = p.first->evaluate_int_expr(s1);
                                    int a22 = p.first->evaluate_int_expr(s2);
                                    if(a11 == a22)
                                        continue;
                                    else
                                    {
                                        if(!p.second)
                                            return (a11 < a22)?-1:1;
                                        else
                                            return (a11 < a22)?1:-1;
                                    }
                                }
                                else
                                {
                                    double a1 = p.first->evaluate_double_expr(s1);
                                    double a2 = p.first->evaluate_double_expr(s2);
                                    if(a1 == a2)
                                        continue;
                                    else
                                    {
                                        if(!p.second)
                                            return (a11 < a22)?-1:1;
                                        else
                                            return (a11 < a22)?1:-1;
                                    }
                                }
                            }
                            return -1;
                        }
                       else
                        return -1;//simply order as read by table. 
                    }
                }
                else
                {
                    /*WARNING: We are running the function on host here.*/
                    double a1  = select_query->group_term->evaluate_double_expr(s1);
                    double a2 = select_query->group_term->evaluate_double_expr(s2); 
                    if(a1 == a2)
                        return 0;//equal
                    else
                    {
                       if(select_query->order_term.size() != 0)
                        {
                            for(std::pair<ExpressionNode*,bool>& p: select_query->order_term)
                            {
                                if(p.first->type_of_expr == 2)
                                {
                                    int a1 = p.first->evaluate_int_expr(s1);
                                    int a2 = p.first->evaluate_int_expr(s2);
                                    if(a1 == a2)
                                        continue;
                                    else
                                    {
                                        if(!p.second)
                                            return (a1 < a2)?-1:1;
                                        else
                                            return (a1 < a2)?1:-1;
                                    }
                                }
                                else
                                {
                                    double a1 = p.first->evaluate_double_expr(s1);
                                    double a2 = p.first->evaluate_double_expr(s2);
                                    if(a1 == a2)
                                        continue;
                                    else
                                    {
                                        if(!p.second)
                                            return (a1 < a2)?-1:1;
                                        else
                                            return (a1 < a2)?1:-1;
                                    }
                                }
                            }
                            return -1;
                        }
                       else
                        return -1;//simply order as read by table. 
                    }   
                }
            }
            else
            {
                //just order by if present.
                if(select_query->order_term.size() != 0)
                {
                    for(std::pair<ExpressionNode*,bool> p: select_query->order_term)
                    {
                        if(p.first->type_of_expr == 2)
                        {
                            int a1 = p.first->evaluate_int_expr(s1);
                            int a2 = p.first->evaluate_int_expr(s2);
                            if(a1 == a2)
                                continue;
                            else
                            {
                                if(!p.second)
                                    return (a1 < a2)?-1:1;
                                else
                                    return (a1 < a2)?1:-1;
                            }
                        }
                        else
                        {
                            double a1 = p.first->evaluate_double_expr(s1);
                            double a2 = p.first->evaluate_double_expr(s2);
                            if(a1 == a2)
                                continue;
                            else
                            {
                                if(!p.second)
                                    return (a1 < a2)?-1:1;
                                else
                                    return (a1 < a2)?1:-1;
                            }
                        }
                    }
                    return -1;
                }
                else
                    return -1;//simply order as read by table.
            }
        }
};
//SELECT x,y,z from T where gear_toggle && speed < 20
//A rows: launch A GPU threads 
//obj -> obj.evaluate_bool_expr(row_object)
class Table{
private:
    int nb,nt;
    Schema* StateDatabase;//The table is represented by an object array. Primary keys are as intended.
    int* anomaly_states;//This table is to track state transitions for anomaly detection.
    int num_states;//number of various anomaly states needed. 
    std::map<int,Schema> work_list;//stores a worklist of various queries to lazily update. State updates happen here.
    const int numberOfRows;
    int write_index;//which row has to be overwritten?
    int max_worklist_size;
    std::mutex mtx;
    Limits l;
    void WriteRows()
    {
        int num_mod_rows = work_list.size();
        init_bt(num_mod_rows);
        int* device_indices;
        Schema* deviceRowsToBeModified;
        cudaMalloc(&device_indices,num_mod_rows*sizeof(int));
        cudaMalloc(&deviceRowsToBeModified,num_mod_rows*sizeof(Schema));
        int i = 0;
        for(std::pair<int,Schema> s: work_list)
        {
            cudaMemcpy(deviceRowsToBeModified+i,&(s.second),cudaMemcpyHostToDevice);
            cudaMemcpy(device_indices+i,&(s.first),cudaMemcpyHostToDevice);
        }
        changeRowsKernel<<<nb,nt>>>(num_mod_rows,device_indices,deviceRowsToBeModified,StateDatabase);
    }

public:
    void init_bt(int num_threads)
    {
        nb = ceil((1.0*num_threads)/1024);
        nt = 1024;
    }

    Table(
        int numRows,int numberOfCars,int max_wl_size
    ):
        numberOfRows(numRows), max_worklist_size(max_wl_size)
    {
        cudaMalloc(&StateDatabase, numberOfRows*sizeof(Schema));//constant size of the table. This will be overwritten.
        num_states = 10;
        write_index = 0;
        anomaly_states = (int*)calloc(num_states * numberOfCars,sizeof(int));        
    }
    void update_worklist(Schema& s)
    {
        state_update(s);
        int x = write_index;
        write_index = (write_index + 1)%numberOfRows;
        work_list[x] = s;//update the schema object being stored.
        if(work_list.size() == max_worklist_size)
        {
            //place a lock on thw worklist here.
            mtx.lock();
            WriteRows();
            mtx.unlock();
            work_list.clear();
        }
    }
    void state_update(Schema& s)
    {
        int ind = s.database_index;
        int* row = (anomaly_states + num_states*ind);
        int anomaly_flag = 0;
        if(s.oil_life_pct < l.min_oil_level)
        {
            row[0] = std::min(row[0]+1,(int)l.oil_violation_time);
            if(row[0] == l.oil_violation_time)
                anomaly_flag |= 1;
        }
        else
            row[0] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[1] = std::min(row[1]+1,(int)l.pressure_violation_time);
            if(row[1] == l.pressure_violation_time)
                anomaly_flag |= 1<<1;
        }
        else
            row[1] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[2] = std::min(row[1]+1,(int)l.pressure_violation_time);
            if(row[2] == l.pressure_violation_time)
                anomaly_flag |= 1<<2;
        }
        else
            row[2] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[3] = std::min(row[1]+1,(int)l.pressure_violation_time);
            if(row[3] == l.pressure_violation_time)
                anomaly_flag |= 1<<3;
        }
        else
            row[3] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[4] = std::min(row[1]+1,(int)l.pressure_violation_time);
            if(row[4] == l.pressure_violation_time)
                anomaly_flag |= 1<<4;
        }
        else
            row[4] = 0;
        if(s.batt_volt < l.min_voltage)
        {
            row[5] = std::min(row[5]+1,(int)l.voltage_violation_time);
            if(row[5] == (int)l.voltage_violation_time)
                anomaly_flag |= 1<<5;
        }
        else
            row[5] = 0;
        if(s.fuel_percentage < l.min_fuel_percentage)
        {
            row[6] = std::min(row[6]+1,(int)l.fuel_violation_time);
            if(row[6] == (int)l.fuel_violation_time)
                anomaly_flag |= 1<<6;
        }
        else
            row[6] = 0;
        if(s.hard_brake)
        {
            row[7] = std::min(row[7]+1,(int)l.brake_violation_time);
            if(row[7] == (int)l.brake_violation_time)
                anomaly_flag |= 1<<7;
        }
        else
            row[7] = 0;
        if(!s.door_lock)
        {
            row[8] = std::min(row[8]+1,(int)l.door_violation_time);
            if(row[8] == l.door_violation_time)
                anomaly_flag |= 1<<8;
        }
        else
            row[8] = 0;
        if(s.hard_steer)
        {
            row[9] = std::min(row[9]+1,(int)l.steer_violation_time);
            if(row[9] == (int)l.steer_violation_time)
                anomaly_flag |= 1<<9;
        }
        if(anomaly_flag != 0)
        {
            char c[20];
            sprintf(c,"shm_1_%d",s.vehicle_id);
            int fd = shm_open(c,O_CREAT | O_RDWR,0666);
            sprintf(c,"shm_2_%d",s.vehicle_id);
            int fd1 = shm_open(c,O_CREAT|O_RDWR,0666);
            int* ptr = (int*)mmap(0,4,PROT_READ | PROT_WRITE, MAP_SHARED,fd,0);
            int* ptr1 = (int*)mmap(0,4,PROT_READ|PROT_WRITE,MAP_SHARED,fd1,0);
            *ptr1 = anomaly_flag;
            *ptr = 1;//written, now send a signal to handle anomaly.
            kill(s.vehicle_id,SIGUSR1); 
        }
    }
    std::set<Schema,select_comparator> Select(SelectQuery* select_query)//parse the query and filter out what's relevant. 
    {
        //acquire a lock before writing? Mostly needed, can be a lock using mutex across both threads. 
        int* selectedValues;
        int* endIndexSelectedValues;
        Schema* retArr;
        int size;
        cudaMalloc(&selectedValues, numberOfRows*sizeof(int));//row indices that were selected 
        cudaMalloc(&endIndexSelectedValues, sizeof(int));
        cudaMemset(endIndexSelectedValues,0,4);//set this value to zero.        
        init_bt(numberOfRows);
        selectKernel<<<nb, nt>>>(
                StateDatabase,
                numberOfRows,
                selectedValues,
                endIndexSelectedValues,
                select_query
            );
        cudaDeviceSynchronize();
        cudaMemcpy(&size, endIndexSelectedValues, sizeof(int), cudaMemcpyDeviceToHost);
        retArr = new Schema[size];
        cudaMemcpy(retArr, selectedValues, size*sizeof(Schema), cudaMemcpyDeviceToHost);
        std::set<Schema,select_comparator> return_values(retArr, retArr+size,select_comparator(select_query));
        int ms = std::max(return_values.size(),select_query->limit_term);
        while(return_values.size() > ms)
        {
            return_values.erase(std::prev(return_values.end()));//remove the last few elements of the select query.
        }
        return return_values;
    }
    void Table::PrintDatabase()
    {
        //iterate through the table and print out. 
    }
};
class GPSSystem{
private:
    int numberOfVertices;
    int* hostAdjacencyMatrix;
    int nb,nt;
    void init_bt(int numThreads)
    {
        nb = ceil(numThreads/1024.0);
        nt = 1024;
    }    
    std::pair<int*,int*> GPSSystem::djikstra_result(int source,std::set<int>& setDroppedVertices)
    {
        //returns the result after running the Djikstra algorithm, of distance matrix and parent matrices.
        //first value is array of parents, second one is distance as a matrix.
        //This can be used later for whatever purpose.
        // Phase one, make a new device Adjacency Matrix using the old one and
        // the set of dropped vertices
        int numberOfDroppedVertices = setDroppedVertices.size();
        int* hostDroppedVertices = new int[numberOfDroppedVertices];
        int* deviceDroppedVertices;
        int* deviceAdjacencyMatrix;
        int idx = 0;
        for(auto vertex: setDroppedVertices)
        {
            hostDroppedVertices[idx++] = vertex;
        }
        cudaMalloc(&deviceDroppedVertices, numberOfDroppedVertices*sizeof(int));
        cudaMemcpy(deviceDroppedVertices, hostDroppedVertices, numberOfDroppedVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&deviceAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int));
        cudaMemcpy(deviceAdjacencyMatrix, hostAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
        if(numberOfDroppedVertices != 0)
        {
            init_bt(numberOfVertices*numberOfDroppedVertices);
            DropVerticesKernel<<<nb, nt>>>(numberOfVertices,numberOfDroppedVertices,deviceAdjacencyMatrix,deviceDroppedVertices);
                cudaDeviceSynchronize();
        }
            // Phase two, Implement Dijkstra:
            int  hostNumberOfUsedVertices = 0;
            int* minDistance;
            int* hostMinDistance = new int;
            *hostMinDistance = INT_MAX;
            cudaMalloc(&minDistance, sizeof(int));        
            int* argMinVertex;
            cudaMalloc(&argMinVertex, sizeof(int));
            
            int* deviceUsed;
            int* hostUsed = new int[numberOfVertices];
            for(int i = 0; i < numberOfVertices; i++){
                hostUsed[i] = 0;
            }
            int* deviceDistance;
            int* hostDistance = new int[numberOfVertices];
            for(int i = 0; i < numberOfVertices; i ++){
                hostDistance[i] = ((i == source)?0:INT_MAX);
            }
            int* deviceParent;
            int* hostParent = new int[numberOfVertices];
            cudaMalloc((void**)&deviceUsed, numberOfVertices*sizeof(int));
            cudaMalloc((void**)&deviceDistance, numberOfVertices*sizeof(int));
            cudaMalloc((void**)&deviceParent, numberOfVertices*sizeof(int));
            cudaMemcpy(deviceUsed, hostUsed, numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(deviceDistance, hostDistance, numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
            
            while(hostNumberOfUsedVertices < numberOfVertices){
                cudaMemcpy(minDistance, hostMinDistance, sizeof(int), cudaMemcpyHostToDevice);            
                init_bt(numberOfVertices);
                FindMinDistance<<<nb, nt>>>(
                    numberOfVertices,
                    deviceUsed,
                    deviceDistance,
                    minDistance
                );
                cudaDeviceSynchronize();
                FindArgMin<<<nb, nt>>>(
                    numberOfVertices,
                    deviceUsed,
                    deviceDistance,
                    minDistance,
                    argMinVertex
                );
                cudaDeviceSynchronize();
                Relax<<<nb, nt>>>(
                    numberOfVertices,
                    deviceAdjacencyMatrix,
                    deviceUsed,
                    deviceDistance,
                    deviceParent,
                    minDistance,
                    argMinVertex
                );
                cudaDeviceSynchronize();
                hostNumberOfUsedVertices++;
            }
        cudaMemcpy(hostParent, deviceParent, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDistance, deviceDistance, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
        return std::make_pair(hostParent,hostDistance);
    }
public:
    GPSSystem(int numVert, int* initMat){
    numberOfVertices = numVert;
    hostAdjacencyMatrix = new int[numberOfVertices*numberOfVertices];
    for(int i = 0; i < numberOfVertices*numberOfVertices; i++){
        hostAdjacencyMatrix[i] = initMat[i];
    }
}
    std::vector<int> GPSSystem::PathFinder(int source, int destination,std::set<int>& setDroppedVertices){
    std::pair<int*,int*> value = djikstra_result(source, setDroppedVertices);
    int hostParent[numberOfVertices];
    int hostDistance[numberOfVertices];
    cudaMemcpy(hostParent,value.first,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDistance,value.second,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
    std::vector<int> path;
    if(hostDistance[destination] == INT_MAX) return path;
    int currentVertex = destination;
    while(currentVertex != source){
        path.push_back(currentVertex);
        currentVertex = hostParent[currentVertex];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
    return path;
    }
    void convoyNodeFinder(std::map<int,int>& car_ids)
    {
        //djikstras kernel call,and then cumulatively add those distances. Then check the city with least sum of 
        //distance and ask cars to converge there.
        int* sum_array;
        cudaMalloc(&sum_array,numberOfVertices*sizeof(int));
        init_bt(numberOfVertices);
        set_zero<<<nb,nt>>>(sum_array);
        cudaDeviceSynchronize();
        std::set<int> included_vertices;
        for(std::pair<int,int> p: car_ids)
        {
            included_vertices.insert(p.second);
        }
        std::set<int> droppedVertices;
        for(int i = 0;i < numberOfVertices;i++)
        {
            if(included_vertices.find(i) == included_vertices.end())
            {
                double y = rand()/RAND_MAX;
                if(y <= 0.5)//randomly dropped, can be adjusted. 
                    droppedVertices.insert(i);
            }
        }
        std::vector<int*> parent_array;
        std::pair<int*,int*> p;
        init_bt(numberOfVertices);
        for(int i: included_vertices)
        {
            p = djikstra_result(i,droppedVertices);
            parent_array.push_back(p.first);
            addMatrix<<<nb,nt>>>(sum_array,p.second);
            cudaDeviceSynchronize();
        }
        int min_index = thrust::min_element(thrust::device,sum_array,sum_array + numberOfVertices) - sum_array;
        //now write to shared memory and send a signal to each car.
        for(std::pair<int,int> p: car_ids)//gives their respective current positions.
        {
            std::vector<int> path;
            int curr = min_index;
            while(curr != -1)
            {
                path.push_back(curr);
                curr = parent_array[p.second][curr];
            }
            std::reverse(path.begin(),path.end());
            char c[20];
            sprintf(c,"shm_1_%d",p.first);
            int fd = shm_open(c,O_CREAT|O_RDWR,0666);
            ftruncate(fd,sizeof(int));
            int* ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            *ptr = 2;
            c[4] = 2;
            fd = shm_open(c,O_CREAT|O_RDWR,0666);
            ftruncate(fd,sizeof(int));
            int* ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            *ptr = path.size();
            c[4] = 3;
            fd = shm_open(c,O_CREAT|O_RDWR,0666);
            ftruncate(fd,path.size()*sizeof(int));
            int* ptr = (int*)mmap(0,path.size()*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            for(int i = 0;i < path.size();i++)
                ptr[i] = path[i];
            kill(p.first,SIGUSR1);
            //update the path here by writing to shared memory. 
        }
    }

    std::vector<int> GPSSystem::findGarageOrBunk(int source,int target_type,std::set<int>& setDroppedVertices){
        std::pair<int*,int*> value = djikstra_result(source, setDroppedVertices);
        int hostParent[numberOfVertices];
        int hostDistance[numberOfVertices];
        cudaMemcpy(hostParent,value.first,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDistance,value.second,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
        int fd = shm_open("vertex_type",O_RDONLY,0666);
        int* arr = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
        int min_dist = INT_MAX;
        int path_v = -1;
        for(int i = 0;i < numberOfVertices;i++)
        {
            if(arr[i] == target_type)
            {
                if(hostDistance[i] < min_dist)
                {
                    min_dist = hostDistance[i];
                    path_v = i;
                }
            }
        }
        std::vector<int> path;
        if(min_dist == INT_MAX)
            return path;
        int currentVertex = path_v;
        while(currentVertex != source){
            path.push_back(currentVertex);
            currentVertex = hostParent[currentVertex];
        }
        path.push_back(source);
        std::reverse(path.begin(), path.end());
        return path;
    }
};
class Limits
{
    
    //Sometimes, certain status messages may contain incorrect update information. Anomaly detetion seeks to check if 
    //this is actually the case by seeing if a contiguous number of such messages are actually received. One such
    //inconsistent message is an anomaly caused by transmission errors rather than an actual update. 
  
    public: 
        double mileage = 1650;//in km/% of full tank (effectively kmpl) Assume 30 kmpl, 55 L tank.
        double interval_between_messages = 0.1;//10 messages per second.
        double speed_violation_time;//the number of contiguous status messages that indicate a fault.
        double brake_violation_time;
        double seatbelt_violation_time;
        double pressure_violation_time;
        double oil_violation_time;
        double door_violation_time;
        double fuel_violation_time;
        double steer_violaton_time;
        double voltage_violation_time;
        double steer_violation_time;
        double max_speed = 100;
        double min_pressure = 30;//pounds per square inch
        double min_voltage = 13.7;//for a typical car, voltage while running is 13.7-14.7 volts
        double engine_temperature_max = 104.44;//typical engine temperatures are in the range 195-220 degrees Fahrenheit.
        double engine_temperature_min = 90.56;
        double min_oil_level = 0.4;//minimum admissible oil percentage
        double min_fuel_percentage = 0.1;//minimum fuel percentage allowable 
        double brake_recovery_time = 2;//2 seconds to reaccelerate.
       	Limits()
        {
            brake_violation_time = 1/interval_between_messages;//hard brake for 10 continuous messages
            seatbelt_violation_time = 2/interval_between_messages;
            pressure_violation_time = 2/interval_between_messages;
            oil_violation_time = 4/interval_between_messages;
            door_violation_time = 2/interval_between_messages;
            fuel_violation_time = 4/interval_between_messages;
            steer_violaton_time = 1/interval_between_messages;
            voltage_violation_time = 4/interval_between_messages;
        }

};
#endif
