#ifndef PROJ_TYPES_H
#define PROJ_TYPES_H
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <mutex>
#include <thread>
#include <set>
#include <unistd.h>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <limits.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
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
    Schema();
    Schema(int);
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
        double max_voltage = 14.7;
        double max_pressure = 35;//max psi
        double engine_temperature_max = 104.44;//typical engine temperatures are in the range 195-220 degrees Fahrenheit.
        double engine_temperature_min = 90.56;
        double min_oil_level = 0.4;//minimum admissible oil percentage
        double min_fuel_percentage = 0.1;//minimum fuel percentage allowable 
        double brake_recovery_time = 2;//2 seconds to reaccelerate.
       	Limits();
};
class ExpressionNode
{
    public:
        ExpressionNode* left_hand_term;
        char* column_name;//column name of the expression
        double value;//value of scalar field
        char* exp_operator;
        ExpressionNode* right_hand_term;
        int type_of_expr;//can be either 1,2 or 3, if bool/integer/floating point.
        __host__ __device__ ExpressionNode();
        __host__ __device__ ExpressionNode(char*);
        __host__ __device__ int evaluate_int_expr(const Schema&);
        __host__ __device__ double evaluate_double_expr(const Schema&);
        __host__ __device__ bool evaluate_bool_expression(const Schema&);
};

class SelectQuery
{
	public:
		bool distinct_query;
		std::vector<char*> select_columns;//either this field or agg columns will be active.
        std::vector<std::pair<char*,ExpressionNode*>> aggregate_columns;
		std::vector<std::pair<ExpressionNode*,bool>> order_term;
		ExpressionNode* select_expression;
        ExpressionNode* group_term;
		int limit_term;
		SelectQuery();
};

class select_comparator
{
    private:
        SelectQuery* select_query;//on host
    public:
        select_comparator(SelectQuery*);
        int operator ()(const Schema&,const Schema&);
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
    Limits* l;
    void WriteRows();

public:
    void init_bt(int);
    Table(int,int,int);
    void update_worklist(Schema&);
    void state_update(Schema&);
    std::set<Schema,select_comparator> select(SelectQuery*);
    void PrintDatabase();
};
class GPSSystem{
private:
    int numberOfVertices;
    int* hostAdjacencyMatrix;
    int nb,nt;
    void init_bt(int);    
    std::pair<int*,int*> djikstra_result(int,std::set<int>&);
public:
    GPSSystem(int numVert, int* initMat);
    std::vector<int> PathFinder(int,int,std::set<int>&);
    void convoyNodeFinder(std::map<int,int>&);
    std::vector<int> findGarageOrBunk(int,int,std::set<int>&);
};

__host__ __device__ bool str_equal(const char*,const char*);
#endif