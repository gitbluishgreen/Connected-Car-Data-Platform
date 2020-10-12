//this file defines all types used in the programme.
#ifndef PROJ_TYPES
#define PROJ_TYPES
#include <string>
#include <vector>
#include "kernel.h"
#include <cuda.h>
class Schema
{
public:
    int vehicle_id;//the process ID of the car serves as the vehicle ID.
    int database_index;//fixed constant for mapping to the database purposes.
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
    Schema(){}  
};
class ExpressionNode
{
    public:
        ExpressionNode* left_hand_term;
		std::string column_name;//column name of the expression
		double value;//value of scalar field
        std::string exp_operator;
        ExpressionNode* right_hand_term;
        int type_of_expr;//can be either 1,2 or 3, if bool/decimal/integer
        ExpressionNode()
        {
            left_hand_term = right_hand_term = NULL;
        }
        ExpressionNode(std::string op)
        {
            left_hand_term = right_hand_term = NULL;
			exp_operator = op;
        }
        bool evaluate_int_expr(Table t)
        {
            if(type_of_expr != 1)
                return false;
            if(left_hand_term == NULL)

            else
            {
                if(left_hand_term->type_of_expr == 1)
                {
                    if(right_hand_term->type_of_expr == 1)
                    {
                        bool z = 
                    }
                }
            }
            
        }
        double evaluate_double_expr(Table t)
        {
            
        }
		bool evaluate_bool_expr(Table t)
		{
			if(type_of_expr != 1)
				return false;
			if(right_hand_term != NULL)
			{
				bool x = left_hand_term->evaluate_bool_expr();
				bool y = right_hand_term->evaluate_bool_expr();
				if(exp_operator == "Or")
					return x|y;
				else if(exp_operator == "And")
					return x&y;
			}
			else //left and right are both null. Read value from memory.
			{
				
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
		int limit_val;
		SelectQuery()
		{
			distinct_query = false;
		}
};
class Table{
private:
    Schema* StateDatabase;//The table is represented by an object array. Primary keys are as intended.
    int* anomaly_states;//This table is to track state transitions for anomaly detection.
    int num_states;
    map<int,Schema> work_list;//stores a worklist of vaious queries to lazily update. State updates happen here.
    const int numberOfRows;
public:
        Table(
        int numRows, 
        Schema* initTable
        int* anomaly_states;
    ):
        numberOfRows(numRows)
    {
        cudaMalloc(&StateDatabase, numberOfRows*sizeof(Schema));
        num_states = 10;
        anomaly_states = (int*)calloc(num_states * numberOfRows,sizeof(int));
        cudaMemcpy(StateDatabase, initTable, numberOfRows*sizeof(Schema), cudaMemcpyHostToDevice);        
    }

    void state_update(Schema& s)
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
    void update_worklist(Schema& s)
    {
        state_update(s);
        work_list[s.database_index] = s;//update the schema object being stored.
    }

    vector<Schema> get_pending_writes()
    {
        vector<Schema> v;
        for(auto it: work_list)
            v.push_back(it.second);
        return v;
    }

    void WriteRows(vector<Schema> RowsToBeWritten){
        // Find the row numbers to be modified in the database
        // This is done parallely, using the fact that the primary key of each row
        // in the argument rowsToBeModified uniquely defines a row in the actual database.
        Schema* devicerowsToBeModified;
        cudaMemcpy(deviceRowsToBeModified, hostRowsToBeModified, RowsToBeWritten.size()*sizeof(Schema), cudaMemcpyHostToDevice);
        init_bt(RowsToBeWritten.size());//linear number of threads are enough.
        // The next task, using the row numbers acquired with the above kernel, 
        // fire numberOfAttributes*numRowsToBeModified threads to modify the cells 
        // of the actual database
        changeRowsKernel<<<nb, nt>>>(
            numberOfRowsToBeModified,
            deviceRowsToBeModified, 
            StateDatabase 
        );
        cudaDeviceSynchronize();
        //std::cout << "Write Completed!!" << endl;
    }

    map<int,Schema> Select(vector<std::string> columns,std::string conditionAttribute,int conditionValue)
    {
        int* selectedValues;
        int* endIndexSelectedValues;
        int* retArr;
        int temp = 0;
        int size;
        int selectionCol = attributesToCol[selectionAttribute];
        int conditionCol = attributesToCol[conditionAttribute];
        cudaMalloc(&selectedValues, numberOfRows*sizeof(int));//row indices that were selected 
        cudaMalloc(&endIndexSelectedValues, sizeof(int));        
        cudaMemcpy(selectedValues, &temp, sizeof(int), cudaMemcpyHostToDevice);
        init_bt(numberOfRows);
        selectKernel<<<nb, nt>>>(
            StateDatabase,
            numberOfRows,
            numberOfAttributes,
            selectedValues,
            selectionCol,
            conditionCol,
            conditionValue,
            endIndexSelectedValues
        );
        cudaMemcpy(&size, endIndexSelectedValues, sizeof(int), cudaMemcpyDeviceToHost);
        retArr = new int[size];
        cudaMemcpy(retArr, selectedValues, size*sizeof(int), cudaMemcpyDeviceToHost);
        set<int> ret(retArr, retArr+size);
        return ret;
    }

    void PrintDatabase(){
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
};
class GPSsystem{
private:
    int numberOfVertices;
    int* hostAdjacencyMatrix;    
    std::pair<int*,int*> djikstra_result(int source,std::set<int>& setDroppedVertices)
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
        return make_pair(hostParent,hostDistance);
        cudaMemcpy(hostParent, deviceParent, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDistance, deviceDistance, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
    }
public:
    GPSsystem(int numVert, int* initMat){
        numberOfVertices = numVert;
        hostAdjacencyMatrix = new int[numberOfVertices*numberOfVertices];
        for(int i = 0; i < numberOfVertices*numberOfVertices; i++){
            hostAdjacencyMatrix[i] = initMat[i];
        }
    }
    void convoyNodeFinder(std::vector<bool> included_vertices)
    {
        //djikstras kernel call,and then cumulatively add those distances. Then check the city with least sum of 
        //distance and ask cars to converge there.
        int* sum_array;
        cudaMalloc(&sum_array,numberOfVertices*sizeof(int));
        init_bt(numberOfVertices);
        set_zero<<<nb,nt>>>(sum_array);
        cudaDeviceSynchronize();
        std::set<int> droppedVertices;
        for(int i = 0;i < numberOfVertices;i++)
        {
            if(!included_vertices[i])
            {
                double y = rand()/RAND_MAX;
                if(y <= 0.5)
                    droppedVertices.insert(i);
            }
        }
        vector<int*> parent_array;
        std::pair<int*,int*> p;
        init_bt(numberOfVertices);
        for(i = 0;i < numberOfVertices; i++)
        {
            if(included_vertices[i])
            {
                p = djikstra_result(i,droppedVertices);
                parent_array.push_back(p.first);
                addMatrix<<<nb,nt>>>(sum_array,p.second);
                cudaDeviceSynchronize();
            }
        }
        int min_index = thrust::min_element(thrust::device,sum_array,sum_array + numberOfVertices) - sum_array;
        //now write to shared memory and send a signal to each car.
        for(int i = 0;i < numberOfVertices;i++)
        {
            if(included[i])
            {
                vector<int> path;
            }
        }
    }

    std::vector<int> PathFinder(int source, int destination,set<int> setDroppedVertices){
        std::pair<int*,int*> value = djikstra_result(source, setDroppedVertices);
        int hostParent[numberOfVertices];
        int hostDistance[numberOfVertices*numberOfVertices];
        cudaMemcpy(hostParent,value.first,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDistance,value.second,numberOfVertices*sizeof(int),cudaMemcpyDeviceToHost);
        if(hostDistance[destination] == INT_MAX) return vector<int>();
        vector<int> path;
        int currentVertex = destination;
        while(currentVertex != source){
            path.push_back(currentVertex);
            currentVertex = hostParent[currentVertex];
        }
        path.push_back(source);
        reverse(path.begin(), path.end());
        return path;
    }
};
class Limits
{
    /*
    Sometimes, certain status messages may contain incorrect update information. Anomaly detetion seeks to check if 
    this is actually the case by seeing if a contiguous number of such messages are actually received. One such
    inconsistent message is an anomaly caused by transmission errors rather than an actual update. 
    */
    public: 
        double mileage = 1650;//in km/% of full tank (effectively kmpl) Assume 30 kmpl, 55 L tank.
        double interval_between_messages = 0.1;//10 messages per second.
        double speed_violation_time = 1/interval_between_messages;//the number of contiguous status messages that indicate a fault.
        double brake_violation_time = 1/interval_between_messages;//hard brake for 10 continuous messages
        double seatbelt_violation_time = 2/interval_between_messages;
        double pressure_violation_time = 2/interval_between_mesages;
        double oil_violation_time = 4/interval_between_messages;
        double door_violation_time = 2/interval_between_messages;
        double fuel_violation_time = 4/interval_between_mesages;
        double steer_violaton_time = 1/interval_between_messages;
        double max_speed = 100;
        double min_pressure = 30;//pounds per square inch
        double min_voltage = 13.7;//for a typical car, voltage while running is 13.7-14.7 volts
        double engine_temperature_max = 104.44;//typical engine temperatures are in the range 195-220 degrees Fahrenheit.
        double engine_temperature_min = 90.56;
        double min_oil_level = 0.4;//minimum admissible oil percentage
        double min_fuel_percentage = 0.2;//minimum fuel percentage allowable 
        Limits(){}
};
#endif