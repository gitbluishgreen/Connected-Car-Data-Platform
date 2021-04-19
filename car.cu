#include <unistd.h>
#include <limits.h>
#include <iostream>
#include "proj_types.h"
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/wait.h>
#include <signal.h>
#include <cmath>
#include <random>
std::mt19937 generator (123);
std::uniform_real_distribution<double> distribution(0.0, 1.0);
int write_fd;
int pid;
int count = 0;
//path[i] -> path[i+1]
int* path;//the path the car has to follow if a convoy request is being made.
double oil_level;
double pressure_rr;
double pressure_fr;
double pressure_rl;
double pressure_fl;
double voltage;
double fuel_percentage;
double distance_covered;
bool brake = false;
int origin_index;
int destination_index;
int path_size;
int increment_index;
Limits limit_object();
static struct sigaction sa;
void message_handler(int sig, siginfo_t* sig_details,void* context)
{
    //handles the signal. Could be an anomaly signal from the master,convoy or communication message.
    char c[20];
    sprintf(c,"shm_1_%d",pid);//primary shared memory used to decide the type of message.
    int* ptr;
    int fd = shm_open(c,O_RDONLY,0666);
    ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);//reads the message type received.
    if(*ptr == 1)
    {
        //type 1 message: correction request
        sprintf(c,"shm_2_%d",pid);
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,4,PROT_READ,MAP_SHARED,fd,0);
        int x = *ptr;
        int c[10];
        for(int i = 0;i < 10;i++)
        {
            c[i] == (x>>i)&1;
        }
        
        if(c[0] || c[1] || c[2] || c[3] || c[4] || c[5])
        {
            //oil percentage low. Maintenance needed!
            oil_level = 1.0;
            pressure_rl = pressure_rr = pressure_fl = pressure_fr =(limit_object.min_pressure+limit_object.max_pressure)/2.0;
            voltage = (limit_object.min_voltage + limit_object.max_voltage)/2.0;
            sprintf(c,"shm_3_%d",pid);
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            int sz = *ptr;
            c[4] = '4';
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sz*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            if(path != NULL)
                delete path;
            path = new int[sz];
            for(int i =0;i < sz;i++)
                path[i] = ptr[i];
            origin_index = 0;
            destination_index = 1;
            increment_index = 1;
        }
        if(c[6])
        {
            fuel_percentage = 1.0;
            sprintf(c,"shm_3_%d",pid);
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            int sz = *ptr;
            c[4] = '4';
            fd = shm_open(c,O_RDONLY,0666);
            ptr = (int*)mmap(0,sz*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
            if(path != NULL)
                delete path;
            path = new int[sz];
            for(int i =0;i < sz;i++)
                path[i] = ptr[i];
            origin_index = 0;
            destination_index = 1;
            increment_index = 1;
        }
        if(c[7])
        {
            //brakes have been hit.
            brake = false;
            sleep(limit_object.brake_recovery_time);//within this much time, it returns to the original speed.
        }
        if(c[8])
        {
            //lock the door.
            door_lock = true;
        }
        if(c[9])
        {
            //hard steer, path gets reversed?
            increment_index = increment_index * -1;//path is reversed now.
        }
        
    }
    else if(*ptr == 2)
    {
        //convoy request. This message informs the car of the coordinate it has to rerout to and what path to follow
        sprintf(c,"shm_3_%d",pid);
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,sizeof(int),PROT_READ,MAP_SHARED,fd,0);//read the array size first.
        int a_sz = *ptr;
        sprintf(c,"shm_4_%d",pid);
        fd = shm_open(c,O_RDONLY,0666);
        ptr = (int*)mmap(0,4*a_sz,PROT_READ,MAP_SHARED,fd,0);
        if(path != NULL)
            delete path;//free the old memory before allocating new one.
        path = new int[a_sz];
        for(int i = 0;i < a_sz;i++)
            path[i] = ptr[i];
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
void run_state()
{
    double prev_speed = 0.0;//starting from scratch
    int prev_gear = 0;
    int fd = shm_open("adjacency_matrix",O_RDONLY,0666);
    int* ptr = (int*)mmap(0,numberOfCars*numberOfCars*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
    pid = getpid();
    oil_level = limit_object.min_oil_level;
    pressure_fl = limit_object.min_pressure;
    pressure_fr = limit_object.min_pressure;
    pressure_rl = limit_object.min_pressure;
    pressure_rr = limit_object.min_pressure;
    voltage = limit_object.min_voltage;
    fuel_percentage = 1;//start with full tank. Use some draining rate based on speed.
    origin_index = 0;
    destination_index = 1;
    distance_covered = 0;
    Schema obj();
    obj.vehicle_id =  pid;
    double new_speed = 0.0;
    while(true)
    {
        obj.database_index = count;
        obj.oil_life_pct = limit_object.min_oil_level + (1-limit_object.min_oil_level)*distribution(generator);
        obj.tire_p_rl = pressure_rl + (limit_object.max_pressure - limit_object.min_pressure)*distribution(generator);
        obj.tire_p_rr = pressure_rr + (limit_object.max_pressure - limit_object.min_pressure)*distribution(generator);
        obj.tire_p_fl = pressure_fl + (limit_object.max_pressure - limit_object.min_pressure)*distribution(generator);
        obj.tire_p_fr = pressure_fr + (limit_object.max_pressure - limit_object.min_pressure)*distribution(generator);
        obj.batt_volt = voltage + (limit_object.max_voltage-limit_object.min_voltage)*distribution(generator);
        if(prev_speed < limit_object.max_speed * 0.9)
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
            obj.clutch = (old_gear != new_gear); 
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
            obj.clutch = (old_gear != new_gear); 
            obj.speed = new_speed;
            obj.accel = true;
            prev_gear = new_gear;
            prev_speed = new_speed;   
        }
        distance_covered += limit_object.interval_between_messages*(prev_speed + (new_speed-prev_speed)*0.5*limit_object.interval_between_messages);
        char c[20];
        sprintf(c,"shm_3_%d",pid);
        int fd = shm_open(c,O_RDONLY,0666);
        int* sz = (int*)mmap(0,4,PROT_READ,MAP_SHARED,fd,0);
        sprintf(c,"shm_4_%d",pid);
        fd = shm_open(c,O_RDONLY,0666);
        int* arr = (int*)mmap(0,4 * (*a_sz),PROT_READ,MAP_SHARED,fd,0);
        
        if(distance_covered >= ptr[numberOfCars*arr[origin_index]+arr[destination_index])
        {
            origin_vertex = origin_vertex + increment_index;
            destination_vertex = destination_vertex + increment_index;
        }
        if(destination_vertex == sz)
        {
            increment_index = -1;
            destination_vertex = sz-2;
            origin_vertex = sz-1;
        }
        if(destination_vertex == -1)
        {
            increment_index = 1;
            origin_vertex = 0;
            destination_vertex = 1;
        }
        obj.fuel_percentage = fuel_percentage - limit_object.interval_between_messages* new_speed/limit_object.mileage;
        obj.hard_brake = brake;
        obj.door_lock = true;
        write(write_fd,&obj,sizeof(obj));
    }
}

void initialize(int numberOfCars,int* file_descriptor,std::map<int,int>& car_map)
{
    int i;
    srand(time(NULL));
    write_fd = file_descriptor[1];
    for(i=0;i<numberOfCars;i++)
    {
        int pid = fork();
        if(pid < 0)
        {
            std::cout<<"Error creating Car #"<<i<<'\n';
            return;
        }
        else if(pid == 0)
        {//child process
            sa.sa_sigaction = &message_handler;
            sa.sa_flags |= SA_SIGINFO;
            if(sigaction(SIGUSR1,&sa,NULL) != 0)
            {
                std::cout<<"Error while initializing the signal handler!\n";
            }
            run_state();
            exit(0);
        }
        else
        {
            car_map[count] = pid;
            count++;
            //parent process 
        }
    }
    wait(NULL);//wait for all cars to finish.
    exit(0);
}
