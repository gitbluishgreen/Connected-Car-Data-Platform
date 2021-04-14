//this file defines all types used in the programme.
#include <string>
#include <vector>
#include "proj_types.hpp"
#include "kernel.h"
#include <cuda.h>
//SELECT x,y,z from T where gear_toggle && speed < 20
//A rows: launch A GPU threads 
//obj -> obj.evaluate_bool_expr(row_object)
ExpressionNode::ExpressionNode()
{
    left_hand_term = right_hand_term = NULL;
}
ExpressionNode::ExpressionNode(std::string op)
{
    left_hand_term = right_hand_term = NULL;
	exp_operator = op;
}
bool ExpressionNode::evaluate_int_expr(const Schema& s)
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
double ExpressionNode::evaluate_double_expr(const Schema& s)
{
    if(type_of_expr != 3)
        return 0.0;
    else if(right_hand_term == NULL)
    {
        double oil_life_pct;//oil percentage
        double tire_p_rl;//tire pressures on each tire.
        double tire_p_rr;
        double tire_p_fl;
        double tire_p_fr;
        double batt_volt;
        double fuel_percentage;
        double speed;
        double distance;
        if(column_name == "oil_life_pct")
            return oil_life_pct;
        else if(column_name == "tire_p_rl")
            return tire_p_rl;
        else if(column_name == "tire_p_rr")
            return tire_p_rr;
        else if(column_name == "tire_p_lf")
            return tire_p_lf;
        else if(column_name == "tire_p_lr")
            return tire_p_lr;
        else if(column_name == "batt_volt")
            return batt_volt;
        else if(column_name == "fuel_percentage")
            return fuel_percentage;
        else if(column_name == "speed")
            return speed;
        else if(column_name == "distance")
            return distance;
        else
            return value;
    }
    else
    {
        double a1 = left_hand_term->evaluate_double_expr();
        double a2 = right_hand_term->evaluate_double_expr();
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
bool ExpressionNode::evaluate_bool_expr(const Schema& s)
{
	if(type_of_expr != 1)
		return false;
    else if(right_hand_term == NULL)
    {
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
SelectQuery::SelectQuery()
{
    distinct_query = false;
    limit_term = -1;
}

Table::Table(
        int numRows,int numberOfCars,int max_wl_size
    ):
        numberOfRows(numRows), max_worklist_size(max_wl_size)
    {
        cudaMalloc(&StateDatabase, numberOfRows*sizeof(Schema));//constant size of the table. This will be overwritten.
        num_states = 10
        write_index = 0;
        anomaly_states = (int*)calloc(num_states * numberOfCars,sizeof(int));
        cudaMemcpy(StateDatabase, initTable, numberOfRows*sizeof(Schema), cudaMemcpyHostToDevice);        
    }

void Table::init_bt(int num_threads)
{
    nb = ceil((1.0*num_threads)/1024);
    nt = 1024;
}
void Table::state_update(Schema& s)
{
    int ind = s.database_index;
    int* row = (anomaly_states + num_states*ind);
    int anomaly_flag = 0;
    if(s.oil_life_pct < l.min_oil_level)
    {
        row[0] = min(row[0]+1,l.oil_violation_time);
        if(row[0] == l.oil_violation_time)
            anomaly_flag |= 1;
    }
    else
        row[0] = 0;
    if(s.tire_p_rl < l.min_pressure)
    {
        row[1] = min(row[1]+1,l.pressure_violation_time);
        if(row[1] == l.pressure_violation_time)
            anomaly_flag |= 1<<1;
    }
    else
        row[1] = 0;
    if(s.tire_p_rl < l.min_pressure)
    {
        row[2] = min(row[1]+1,l.pressure_violation_time);
        if(row[2] == l.pressure_violation_time)
            anomaly_flag |= 1<<2;
    }
    else
        row[2] = 0;
    if(s.tire_p_rl < l.min_pressure)
    {
        row[3] = min(row[1]+1,l.pressure_violation_time);
        if(row[3] == l.pressure_violation_time)
            anomaly_flag |= 1<<3;
    }
    else
        row[3] = 0;
    if(s.tire_p_rl < l.min_pressure)
    {
        row[4] = min(row[1]+1,l.pressure_violation_time);
        if(row[4] == l.pressure_violation_time)
            anomaly_flag |= 1<<4;
    }
    else
        row[4] = 0;
    if(s.batt_volt < s.min_voltage)
    {
        row[5] = min(row[5]+1,l.voltage_violation_time);
        if(row[5] == l.voltage_violation_time)
            anomaly_flag |= 1<<5;
    }
    else
        row[5] = 0;
    if(s.fuel_percentage < l.min_fuel_percentage)
    {
        row[6] = min(row[6]+1,l.fuel_violation_time);
        if(row[6] == l.fuel_violation_time)
            anomaly_flag |= 1<<6;
    }
    else
        row[6] = 0;
    if(s.hard_brake)
    {
        row[7] = min(row[7]+1,l.brake_violation_time);
        if(row[7] == l.brake_violation_time)
            anomaly_flag |= 1<<7;
    }
    else
        row[7] = 0;
    if(!s.door_lock)
    {
        row[8] = min(row[8]+1,l.door_violation_time);
        if(row[8] == l.door_violation_time)
            anomaly_flag |= 1<<8;
    }
    else
        row[8] = 0;
    if(s.hard_steer)
    {
        row[9] = min(row[9]+1,l.steer_violation_time);
        if(row[9] == l.steer_violation_time)
            anomaly_flag |= 1<<9;
    }
    if(anomaly_flag != 0)
    {
        char c[20];
        sprintf(c,"shm_1_%d",s.vehicle_id);
        int fd = shm_open(c,O_READ | O_WRITE,0666);
        sprintf(c,"shm_2_%d",s.vehicle_id);
        int fd1 = shm_open(c,O_READ | O_WRITE,0666);
        int* ptr = (int*)mmap(0,4,PROT_READ | PROT_WRITE, MAP_SHARED,fd,0);
        int* ptr1 = (int*)mmap(0,4,PROT_READ|PROT_WRITE,MAP_SHARED,fd1,0);
        *ptr1 = anomaly_flag;
        *ptr = 1;//written, now send a signal to handle anomaly.
        kill(s.vehicle_id,SIGUSR1); 
    }
}
void Table::update_worklist(Schema& s)
{
    state_update(s);
    int x = write_index;
    write_index = (write_index + 1)%numberOfRows;
    work_list[x] = s;//update the schema object being stored.
    if(work_list.size() == max_worklist_size)
    {
        WriteRows();
        work_list.clear();
    }
}

void Table::WriteRows(){
        // Find the row numbers to be modified in the database
        // This is done parallely, using the fact that the primary key of each row
        // in the argument rowsToBeModified uniquely defines a row in the actual database.
    Schema* devicerowsToBeModified;
    int* indices_to_overwrite;
    cudaMalloc(&devicerowsToBeModified,work_list.size()*sizeof(Schema));
    cudaMalloc(&indices_to_overwrite,work_list.size()*sizof(int));
    int i = 0;
    for(std::pair<int,Schema> obj: work_list)
    {
        cudaMemcpy(devicerowsToBeModified + i,obj.second,cudaMemcpyHostToDevice);
        cudaMemcpy(indices_to_overwrite + i,obj.first,cudaMemcpyHostToDevice);
        i++;
    }
    init_bt(work_list.size());//linear number of threads are enough.
        // The next task, using the row numbers acquired with the above kernel, 
        // fire numberOfAttributes*numRowsToBeModified threads to modify the cells 
        // of the actual database
    changeRowsKernel<<<nb, nt>>>(
        work_list.size(),
        indices_to_overwrite,
        deviceRowsToBeModified, 
        StateDatabase 
    );//this kernel has to overwrite the existing allocated memory.
    cudaDeviceSynchronize();
        //std::cout << "Write Completed!!" << endl;
}

std::set<Schema> Table::Select(SelectQuery* select_query)//parse the query and filter out what's relevant. 
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
    std::set<Schema> ret(retArr, retArr+size);
    delete retArr;
    return ret;
}

void Table::PrintDatabase(){
    int* hostDatabase = new int[numberOfAttributes*numberOfRows];
    cudaMemcpy(hostDatabase, StateDatabase, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < numberOfRows; i ++){
        for(int j = 0; j < numberOfAttributes; j ++){
            std::cout << hostDatabase[i*numberOfAttributes+j] << " ";
        }
        std::cout << endl;
    }
    std::cout << endl;
}
std::pair<int*,int*> GPSsystem::djikstra_result(int source,std::set<int>& setDroppedVertices)
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
    return make_pair(hostParent,hostDistance);
}
GPSsystem::GPSsystem(int numVert, int* initMat){
    numberOfVertices = numVert;
    hostAdjacencyMatrix = new int[numberOfVertices*numberOfVertices];
    for(int i = 0; i < numberOfVertices*numberOfVertices; i++){
        hostAdjacencyMatrix[i] = initMat[i];
    }
}
void GPSsystem::convoyNodeFinder(std::map<int,int>& car_ids)
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
    for(std::pair<int,int> car_ids)//gives their respective current positions.
    {
        std::vector<int> path;
        int curr = min_index;
        while(curr != -1)
        {
            path.push_back(curr);
            curr = parent_array[p.second][curr];
        }
        reverse(path.begin(),path.end());
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
    reverse(path.begin(), path.end());
    return path;
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
    reverse(path.begin(), path.end());
    return path;
}
Limits::Limits(){}