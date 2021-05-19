#include <unistd.h>
#include <limits.h>
#include <iostream>
#include "proj_types.cuh"
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <vector>
#include <time.h>
#include <sys/wait.h>
#include <signal.h>
#include <cmath>
#include <random>
std::mt19937 generator (123);
std::uniform_real_distribution<double> distribution(0.0, 1.0);
int write_fd;
const double conversion_factor = 1.0/3600.0;
int pid;
int count = 0;
//path[i] -> path[i+1]
std::vector<int> path;//the path the car has to follow if a convoy request is being made.
int* adjacency_matrix;
double oil_level;
double pressure_rr;
double pressure_fr;
double pressure_rl;
double pressure_fl;
double voltage;
double fuel_percentage;
double distance_covered;
bool brake = false;
bool door_lock = true;
int origin_index;
int destination_index;
int increment_index;
Limits* limit_object;
static struct sigaction sa1;
void car_message_handler(int sig, siginfo_t* sig_details,void* context)
{
    //handles the signal. Could be an anomaly signal from the master,convoy or communication message.
    std::cout<<pid<<" was pulled up by master!\n";
    char c[20];
    sprintf(c,"shm_1_%d",pid);//primary shared memory used to decide the type of message.
    int* ptr;
    int fd = shm_open(c,O_RDONLY,0666);
    ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);//reads the message type received.
    if(*ptr == 1)
    {
        //type 1 message: correction request
        sprintf(c,"shm_2_%d",pid);
        close(fd);
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,4,PROT_READ,MAP_SHARED,fd,0);
        int x = *ptr;
        close(fd);
        int c1[10];
        for(int i = 0;i < 10;i++)
        {
            int t = x>>i;
            c1[i] == t&1;
        }
        
        if(c1[0] || c1[1] || c1[2] || c1[3] || c1[4] || c1[5])
        {
            //oil percentage low. Maintenance needed!
            oil_level = 1.0;
            pressure_rl = pressure_rr = pressure_fl = pressure_fr = (limit_object->min_pressure+limit_object->max_pressure)/2.0;
            voltage = (limit_object->min_voltage + limit_object->max_voltage)/2.0;
            sprintf(c,"shm_3_%d",pid);
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            int sz = *ptr;
            close(fd);
            c[4] = '4';
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sz*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            path.resize(sz);
            for(int i =0;i < sz;i++)
                path[i] = ptr[i];
            close(fd);
            origin_index = 0;
            destination_index = 1;
            increment_index = 1;
        }
        if(c1[6])
        {
            fuel_percentage = 1.0;
            sprintf(c,"shm_3_%d",pid);
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            int sz = *ptr;
            close(fd);
            c[4] = '4';
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sz*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            path.resize(sz);
            for(int i =0;i < sz;i++)
                path[i] = ptr[i];
            close(fd);
            origin_index = 0;
            destination_index = 1;
            increment_index = 1;
        }
        if(c1[7])
        {
            //brakes have been hit.
            brake = false;
            sleep(limit_object->brake_recovery_time);//within this much time, it returns to the original speed.
        }
        if(c1[8])
        {
            //lock the door.
            door_lock = true;
        }
        if(c1[9])
        {
            //hard steer, path gets reversed?
            increment_index = increment_index * -1;//path is reversed now.
        }
    }
    else if(*ptr == 2)
    {
        //convoy/garage/bunk request. This message informs the car of the coordinate it has to rerout to and what path to follow
        close(fd);//earlier fd has to be closed, that's why segfault.....
        sprintf(c,"shm_3_%d",pid);
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);//read the array size first.
        int a_sz = *ptr;
        close(fd);
        c[4] = '4';
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,4*a_sz,PROT_READ,MAP_SHARED,fd,0);
        path.resize(a_sz);
        for(int i = 0;i < a_sz;i++)
            path[i] = ptr[i];
        close(fd);
        origin_index = 0;
        destination_index = 1;
        increment_index = 1;
    }
    // else if(*ptr == 3)
    // {
    //     //message from another car. Take appropriate action.
    //     Add this feature if necessary.
    // }
}
void run_state(int numberOfCars,int numberOfVertices)
{
    double prev_speed = 0.0;//starting from scratch
    int prev_gear = 0;
    int fd = shm_open("adjacency_matrix",O_RDONLY,0666);
    adjacency_matrix = (int*)mmap(0,numberOfVertices*numberOfVertices*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
    pid = getpid();
    oil_level = 1.0;//start with full value.Reduces at a constant rate. 
    pressure_fl = limit_object->min_pressure;
    pressure_fr = limit_object->min_pressure;
    pressure_rl = limit_object->min_pressure;
    pressure_rr = limit_object->min_pressure;
    voltage = limit_object->min_voltage;
    fuel_percentage = 1.0;//start with full tank. Use some draining rate based on speed.
    origin_index = 0;
    destination_index = 1;
    distance_covered = 0;
    double total_distance_covered = 0;
    Schema obj(10);//object creation.
    obj.vehicle_id =  pid;
    double new_speed = 0.0;
    struct timespec sleep_time;
    struct timespec rem_time;
    path.resize(numberOfVertices);
    for(int i = 0;i < numberOfVertices;i++)
        path[i] = i;
    while(true)
    {
        obj.database_index = count;
        obj.tire_p_rl = pressure_rl + (limit_object->max_pressure - limit_object->min_pressure)*distribution(generator);
        obj.tire_p_rr = pressure_rr + (limit_object->max_pressure - limit_object->min_pressure)*distribution(generator);
        obj.tire_p_fl = pressure_fl + (limit_object->max_pressure - limit_object->min_pressure)*distribution(generator);
        obj.tire_p_fr = pressure_fr + (limit_object->max_pressure - limit_object->min_pressure)*distribution(generator);
        obj.batt_volt = voltage + (limit_object->max_voltage-limit_object->min_voltage)*distribution(generator);
        obj.origin_vertex = path[origin_index];
        obj.destination_vertex = path[destination_index];
        if(prev_speed < limit_object->max_speed * 0.9)
        {
            new_speed = prev_speed + distribution(generator)*2;
            int new_gear;
            if(new_speed < 10)
                new_gear = 1;
            else if(new_speed < 30)
                new_gear = 2;
            else if(new_speed < 50)
                new_gear = 3;
            else if(new_speed < 70)
                new_gear = 4;
            else
                new_gear = 5;
            obj.gear_toggle = (prev_gear != new_gear);
            obj.clutch = (prev_gear != new_gear); 
            obj.speed = new_speed;
            obj.accel = true;
            prev_gear = new_gear;
            prev_speed = new_speed;
        }
        else
        {
            new_speed = prev_speed - distribution(generator)*2;
            int new_gear;
            if(new_speed < 10)
                new_gear = 1;
            else if(new_speed < 30)
                new_gear = 2;
            else if(new_speed < 50)
                new_gear = 3;
            else if(new_speed < 70)
                new_gear = 4;
            else
                new_gear = 5;
            obj.gear_toggle = (prev_gear != new_gear);
            obj.clutch = (prev_gear != new_gear); 
            obj.speed = new_speed;
            obj.accel = true;
            prev_gear = new_gear;
            prev_speed = new_speed;   
        }
        double incremental_distance = limit_object->interval_between_messages*(prev_speed*conversion_factor + (new_speed-prev_speed)*0.5*limit_object->interval_between_messages*conversion_factor);
        distance_covered +=  incremental_distance;   
        total_distance_covered += incremental_distance;
        if(distance_covered >= adjacency_matrix[numberOfCars*path[origin_index]+path[destination_index]])
        {
            distance_covered = 0;
            origin_index = origin_index + increment_index;
            destination_index = destination_index + increment_index;
        }
        if(destination_index == path.size())
        {
            increment_index = -1;
            destination_index = path.size()-2;
            origin_index = path.size()-1;
        }
        if(destination_index == -1)
        {
            increment_index = 1;
            origin_index = 0;
            destination_index = 1;
        }
        obj.fuel_percentage = fuel_percentage - limit_object->interval_between_messages*conversion_factor*new_speed/limit_object->mileage;
        obj.hard_brake = brake;
        obj.door_lock = door_lock;
        oil_level = 1.0 - distance_covered*limit_object->oil_capacity;
        //std::cout<<"For "<<pid<<" Oil_life_pct is "<<oil_level<<" distance is "<<distance_covered<<" and incremental distance is "<<incremental_distance<<'\n';
        //std::cout<<"Oil level for "<<pid<<" is "<<oil_level<<" and distance is "<<incremental_distance<<'\n';
        obj.oil_life_pct = oil_level;
        obj.distance = total_distance_covered;
        write(write_fd,&obj,sizeof(obj));
        sleep_time.tv_nsec = (long)(1e9*limit_object->interval_between_messages);
        nanosleep(&sleep_time,&rem_time);//interval between messages
    }
    close(fd);
}

void initialize(int numberOfCars,int numberOfVertices,int* file_descriptor,std::map<int,int>* car_map)
{
    int i;
    srand(time(NULL));
    write_fd = file_descriptor[1];
    limit_object = new Limits();
    for(i=0;i<numberOfCars;i++)
    {
        pid = fork();
        if(pid < 0)
        {
            std::cout<<"Error creating Car #"<<i<<'\n';
            return;
        }
        else if(pid == 0)
        {//child process
            sa1.sa_sigaction = &car_message_handler;
            sa1.sa_flags |= SA_SIGINFO;
            if(sigaction(SIGUSR1,&sa1,NULL) != 0)
            {
                std::cout<<"Error while initializing the signal handler!\n";
            }
            run_state(numberOfCars,numberOfVertices);
            exit(0);
        }
        else
        {
            (*car_map)[count] = pid;
            count++;
            //parent process 
        }
    }
    //returns to the join function. Thread 2 still runs, server is still alive.
}
