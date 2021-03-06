//this file defines all types and headers used.
#include "proj_types.cuh"
#include "kernel.cuh"
Schema::Schema(){}
Schema::Schema(int dummy){}
__host__ __device__ Schema::Schema(const Schema& s)
{
    vehicle_id = s.vehicle_id;//the process ID of the car serves as the vehicle ID.
    database_index = s.database_index;//fixed constant for mapping to the database purposes for anomaly detection.
    oil_life_pct = s.oil_life_pct;//oil percentage
    tire_p_rl = s.tire_p_rl;//tire pressures on each tire.
    tire_p_rr = s.tire_p_rr;
    tire_p_fl = s.tire_p_fl;
    tire_p_fr = s.tire_p_fr;
    batt_volt = s.batt_volt;
    fuel_percentage = s.fuel_percentage;
    accel = s.accel;
    seatbelt = s.seatbelt;
    hard_brake = s.hard_brake;
    door_lock = s.door_lock;
    gear_toggle = s.gear_toggle;
    clutch = s.clutch;
    hard_steer = s.hard_steer;
    speed = s.speed;
    distance = s.distance;
    origin_vertex = s.origin_vertex;
    destination_vertex = s.destination_vertex;
}
__host__ __device__ Schema& Schema::operator = (const Schema& s)
{
    this->vehicle_id = s.vehicle_id;//the process ID of the car serves as the vehicle ID.
    this->database_index = s.database_index;//fixed constant for mapping to the database purposes for anomaly detection.
    this->oil_life_pct = s.oil_life_pct;//oil percentage
    this->tire_p_rl = s.tire_p_rl;//tire pressures on each tire.
    this->tire_p_rr = s.tire_p_rr;
    this->tire_p_fl = s.tire_p_fl;
    this->tire_p_fr = s.tire_p_fr;
    this->batt_volt = s.batt_volt;
    this->fuel_percentage = s.fuel_percentage;
    this->accel = s.accel;
    this->seatbelt = s.seatbelt;
    this->hard_brake = s.hard_brake;
    this->door_lock = s.door_lock;
    this->gear_toggle = s.gear_toggle;
    this->clutch = s.clutch;
    this->hard_steer = s.hard_steer;
    this->speed = s.speed;
    this->distance = s.distance;
    this->origin_vertex = s.origin_vertex;
    this->destination_vertex = s.destination_vertex;
    return *this;
}

Limits::Limits()
{
    brake_violation_time = 1/interval_between_messages;//hard brake for 10 continuous messages
    seatbelt_violation_time = 2/interval_between_messages;
    pressure_violation_time = 2/interval_between_messages;
    oil_violation_time = 4/interval_between_messages;
    door_violation_time = 2/interval_between_messages;
    fuel_violation_time = 4/interval_between_messages;
    steer_violaton_time = 1/interval_between_messages;
    voltage_violation_time = 4/interval_between_messages;
    speed_violation_time = 2/interval_between_messages;
}
__host__ __device__ ExpressionNode::ExpressionNode()
{
    left_hand_term = right_hand_term = NULL;
}
__host__ __device__ ExpressionNode::ExpressionNode(char* op)
{
    left_hand_term = right_hand_term = NULL;
    exp_operator = op;
}
__host__ __device__ int ExpressionNode::evaluate_int_expression(const Schema& s)
{
    if(type_of_expr != 2)
        return 0;
    if(left_hand_term == NULL)
    {
        if(str_equal(column_name,"vehicle_id"))
                return s.vehicle_id;
        else if(str_equal(column_name,"origin_vertex"))
            return s.origin_vertex;
        else if(str_equal(column_name,"destination_vertex"))
            return s.destination_vertex;
        else
            return (int)value;
    }
    else
    {
        int x = left_hand_term->evaluate_int_expression(s);
        int y = right_hand_term->evaluate_int_expression(s);
        if(str_equal(exp_operator,"plus"))
            return x+y;
        else if(str_equal(exp_operator,"minus"))
            return x-y;
        else if(str_equal(exp_operator,"mult"))
            return x*y;
        else if(str_equal(exp_operator,"div"))
            return x/y;
        else if(str_equal(exp_operator,"modulo"))
            return x%y;
        else
            return 0;
    }
}
__host__ __device__ double ExpressionNode::evaluate_double_expression(const Schema& s)
{
    if(type_of_expr != 3)
        return 0.0;
    else if(right_hand_term == NULL)
    {
        if(str_equal(column_name,"oil_life_pct"))
            return s.oil_life_pct;//this is on device memory.
        else if(str_equal(column_name,"tire_p_rl"))
            return s.tire_p_rl;//this is on device memory.
        else if(str_equal(column_name,"tire_p_rr"))
            return s.tire_p_rr;//this is on device memory.
        else if(str_equal(column_name,"tire_p_fl"))
            return s.tire_p_fl;//this is on device memory.
        else if(str_equal(column_name,"tire_p_fr"))
            return s.tire_p_fr;//this is on device memory.
        else if(str_equal(column_name,"batt_volt"))
            return s.batt_volt;//this is on device memory.
        else if(str_equal(column_name,"fuel_percentage"))
            return s.fuel_percentage;//this is on device memory.
        else if(str_equal(column_name,"speed"))
            return s.speed;//this is on device memory.
        else if(str_equal(column_name,"distance"))
            return s.distance;//this is on device memory.
        else
            return value;//this is on pinned memory.
    }
    else
    {
        double a1,a2;
        if(left_hand_term->type_of_expr == 3)
            a1 = left_hand_term->evaluate_double_expression(s);
        else
            a1 = (double)left_hand_term->evaluate_int_expression(s);
        if(right_hand_term->type_of_expr == 3)
            a2 = right_hand_term->evaluate_double_expression(s);
        else
            a2 = (double)right_hand_term->evaluate_int_expression(s);
        if(str_equal(exp_operator,"plus"))
            return a1+a2;
        else if(str_equal(exp_operator,"minus"))
            return a1-a2;
        else if(str_equal(exp_operator,"mult"))
            return a1*a2;
        else if(str_equal(exp_operator,"div"))
            return a1/a2;
        else 
            return 0;
    }
}
__host__ __device__ bool ExpressionNode::evaluate_bool_expression(const Schema& s)
{
    if(type_of_expr != 1)
        return false;
    else if(right_hand_term == NULL)
    {
        if(str_equal(exp_operator,"not"))
            return !left_hand_term->evaluate_bool_expression(s);
        if(str_equal(column_name,"accel"))
            return s.accel;
        else if(str_equal(column_name,"seatbelt"))
            return s.seatbelt;
        else if(str_equal(column_name,"hard_brake"))
            return s.hard_brake;
        else if(str_equal(column_name,"door_lock"))
            return s.door_lock;
        else if(str_equal(column_name,"gear_toggle"))
            return s.gear_toggle;
        else if(str_equal(column_name,"clutch"))
            return s.clutch;
        else if(str_equal(column_name,"hard_steer"))
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
        if(left_hand_term->type_of_expr == 1)
        {
            bool x = left_hand_term->evaluate_bool_expression(s);
            bool y = right_hand_term->evaluate_bool_expression(s);
            if(str_equal(exp_operator,"or"))
                return x|y;
            else if(str_equal(exp_operator,"and"))
                return x&y;
            else
                return false;
        }
        else
        {
            double x,y;
            if(left_hand_term->type_of_expr == 2)
                x = left_hand_term->evaluate_int_expression(s);
            else
                x = left_hand_term->evaluate_double_expression(s);
            if(right_hand_term->type_of_expr == 2)
                y = right_hand_term->evaluate_int_expression(s);
            else
                y = right_hand_term->evaluate_double_expression(s);
            if(str_equal(exp_operator,"greater"))
                return x > y;
            if(str_equal(exp_operator,"lesser"))
                return x < y;
            if(str_equal(exp_operator,"greaterequal"))
                return x >= y;
            if(str_equal(exp_operator,"lesserequal"))
                return x <= y;
            if(str_equal(exp_operator,"doubleequal"))
                return x == y;
            if(str_equal(exp_operator,"notequal"))
                return x != y;
            else
                return false;
        }
    }
}

SelectQuery::SelectQuery()
{
    distinct = false;
    limit_term = -1;
}

select_comparator::select_comparator(SelectQuery* sq)
{
    select_query = sq;//pointer to a pinned memory location.
}
bool select_comparator::operator ()(const Schema& s1,const Schema& s2)//all on host memory now.
{
    for(std::pair<ExpressionNode*,bool> p: *(select_query->order_term))
    {
        if(p.first->type_of_expr == 2)
        {
            int a11 = p.first->evaluate_int_expression(s1);
            int a22 = p.first->evaluate_int_expression(s2);
            if(a11 == a22)
                continue;
            else
            {
                if(!p.second)
                    return (a11 < a22);
                else
                    return (a11 > a22);
            }
        }
        else
        {
            double a11 = p.first->evaluate_double_expression(s1);
            double a22 = p.first->evaluate_double_expression(s2);
            if(a11 == a22)
                continue;
            else
            {
                if(!p.second)
                    return (a11 < a22);
                else
                    return (a11 > a22);
            }
        }
    }
    return false;
}
distinct_comparator::distinct_comparator(SelectQuery* sq)
{
    select_query = sq;
}
bool distinct_comparator::operator ()(const Schema&s1, const Schema& s2)
{
     //select term exists. Order by this for now.
    std::vector<char*>::iterator it;
    for(it = select_query->select_columns->begin();it != select_query->select_columns->end();it++)
    {
        if(str_equal(*it,"vehicle_id")){
            if(s1.vehicle_id == s2.vehicle_id)
                continue;
            else
                return s1.vehicle_id < s2.vehicle_id;
        }
        else if(str_equal(*it,"origin_vertex"))
        {
            if(s1.origin_vertex == s2.origin_vertex)
                continue;
            else
                return s1.origin_vertex < s2.origin_vertex;
        }
        else if(str_equal(*it,"destination_vertex")){
            if(s1.destination_vertex == s2.destination_vertex)
                continue;
            else
                return s1.destination_vertex < s2.destination_vertex;
        }
        else if(str_equal(*it,"oil_life_pct")){
            if(s1.oil_life_pct == s2.oil_life_pct)  
                continue;
            else
                return s1.oil_life_pct < s2.oil_life_pct;
        }
        else if(str_equal(*it,"tire_p_rl")){
            if(s1.tire_p_rl == s2.tire_p_rl)
                continue;
            else
                return (s1.tire_p_rl < s2.tire_p_rl);
        }
        else if(str_equal(*it,"tire_p_rr")){
            if(s1.tire_p_rr == s2.tire_p_rr)
                continue;
            else
                return s1.tire_p_rr < s2.tire_p_rr;
        }
        else if(str_equal(*it,"tire_p_fl")){
            if(s1.tire_p_fl == s2.tire_p_fl)
                continue;
            else
                return s1.tire_p_fl < s2.tire_p_fl;
        }
        else if(str_equal(*it,"tire_p_fr")){
            if(s1.tire_p_fr == s2.tire_p_fr)
                continue;
            else
                return s1.tire_p_fr < s2.tire_p_fr;
        }
        else if(str_equal(*it,"batt_volt")){
            if(s1.batt_volt == s2.batt_volt)
                continue;
            return s1.batt_volt < s2.batt_volt;
        }
        else if(str_equal(*it,"fuel_percentage")){
            if(s1.fuel_percentage == s2.fuel_percentage)    
                continue;
            else
                s1.fuel_percentage < s2.fuel_percentage;

        }
        else if(str_equal(*it,"accel")){
            if(s1.accel == s2.accel)
                continue;
            else
                return s1.accel < s2.accel;
        }
        else if(str_equal(*it,"seatbelt")){
            if(s1.seatbelt == s2.seatbelt)
                continue;
            else
                return s1.seatbelt < s2.seatbelt;
        }
        else if(str_equal(*it,"hard_brake")){
            if(s1.hard_brake == s2.hard_brake)
                continue;
            else
                return s1.hard_brake < s2.hard_brake;
        }
        else if(str_equal(*it,"door_lock")){
            if(s1.door_lock == s2.door_lock)
                continue;
            else
                return s1.door_lock < s2.door_lock;
        }
        else if(str_equal(*it,"gear_toggle")){
            if(s1.gear_toggle == s2.gear_toggle)
                continue;
            else
                return s1.gear_toggle < s2.gear_toggle;
        }
        else if(str_equal(*it,"clutch")){
            if(s1.clutch == s2.clutch)
                continue;
            else
                return s1.clutch < s2.clutch;
        }
        else if(str_equal(*it,"hard_steer")){
            if(s1.hard_steer == s2.hard_steer)
                continue;
            else
                return s1.hard_steer < s2.hard_steer;
        }
        else if(str_equal(*it,"speed")){
            if(s1.speed == s2.speed)
                continue;
            else
                return s1.speed < s2.speed;
        }
        else if(str_equal(*it,"distance")){
            if(s1.distance == s2.distance)  
                continue;
            else
                return s1.distance < s2.distance;
        }
    }
    return false;
}

void Table::WriteRows()
{
    int num_mod_rows = work_list.size();
    init_bt(num_mod_rows);
    int* device_indices;
    Schema* deviceRowsToBeModified;
    cudaMalloc((void**)&device_indices,num_mod_rows*sizeof(int));
    cudaMalloc((void**)&deviceRowsToBeModified,num_mod_rows*sizeof(Schema));
    int i = 0;
    Schema* hostRowsToBeModified = (Schema*)malloc(sizeof(Schema)*num_mod_rows);
    int* hostIndicesToBeModified = (int*)malloc(sizeof(int)*num_mod_rows);
    for(std::pair<int,Schema> s: work_list)
    {
        hostRowsToBeModified[i] = s.second;
        hostIndicesToBeModified[i] = s.first;
        i++;
    }
    cudaMemcpy(deviceRowsToBeModified,hostRowsToBeModified,num_mod_rows*sizeof(Schema),cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices,hostIndicesToBeModified,num_mod_rows*sizeof(int),cudaMemcpyHostToDevice);
    changeRowsKernel<<<nb,nt>>>(num_mod_rows,device_indices,deviceRowsToBeModified,StateDatabase);
    cudaDeviceSynchronize();
    cudaFree(deviceRowsToBeModified);
    cudaFree(device_indices);
}

void Table::init_bt(int num_threads)
{
    nb = ceil((1.0*num_threads)/1024);
    nt = 1024;
}

Table::Table(
    int numRows,int numCars,int max_wl_size,int* rfd
):
    numberOfRows(numRows), numberOfCars(numCars),max_worklist_size(max_wl_size)
{
    request_file_descriptor = rfd;
    l = new Limits();
    cudaMalloc((void**)&StateDatabase, numberOfRows*sizeof(Schema));//constant size of the table. This will be overwritten.
    num_states = 11;
    write_index = 0;
    anomaly_states = (int*)calloc(num_states * numCars,sizeof(int));        
}

void Table::update_worklist(Schema& s)
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
std::map<int,int> Table::get_latest_position()
{
    std::map<int,int> m;
    for(auto it: latest_message)
        m[it.first] = it.second.origin_vertex;
    return m;
}
void Table::state_update(Schema& s)
{
    latest_message[s.vehicle_id] = s;
    int ind = s.database_index;
    int* row = (anomaly_states + num_states*ind);
    int anomaly_flag = 0;
    bool b[11] = {false,false,false,false,false,false,false,false,false,false,false};
    if(s.oil_life_pct < l->min_oil_level)
    {
        row[0] = std::min(row[0]+1,(int)(l->oil_violation_time));
        if(row[0] == l->oil_violation_time)
        {
            anomaly_flag |= 1;
            b[0] = true;
            row[0] = 0;
        }
    }
    else
        row[0] = 0;
    if(s.tire_p_rl < l->min_pressure)
    {
        row[1] = std::min(row[1]+1,(int)(l->pressure_violation_time));
        if(row[1] == l->pressure_violation_time)
        {
            anomaly_flag |= (1<<1);
            b[1] = true;
            row[1] = 0;
        }
    }
    else
        row[1] = 0;
    if(s.tire_p_rl < l->min_pressure)
    {
        row[2] = std::min(row[2]+1,(int)(l->pressure_violation_time));
        if(row[2] == l->pressure_violation_time)
        {
            anomaly_flag |= (1<<2);
            b[2] = true;
            row[2] = 0;
        }
    }
    else
        row[2] = 0;
    if(s.tire_p_rl < l->min_pressure)
    {
        row[3] = std::min(row[3]+1,(int)(l->pressure_violation_time));
        if(row[3] == l->pressure_violation_time)
        {
            anomaly_flag |= (1<<3);
            b[3] = true;
            row[3] = 0;
        }
    }
    else
        row[3] = 0;
    if(s.tire_p_rl < l->min_pressure)
    {
        row[4] = std::min(row[4]+1,(int)(l->pressure_violation_time));
        if(row[4] == l->pressure_violation_time)
        {
            anomaly_flag |= (1<<4);
            b[4] = true;
            row[4] = 0;
        }
    }
    else
        row[4] = 0;
    if(s.batt_volt < l->min_voltage)
    {
        row[5] = std::min(row[5]+1,(int)(l->voltage_violation_time));
        if(row[5] == (int)(l->voltage_violation_time))
        {
            anomaly_flag |= (1<<5);
            b[5] = true;
            row[5] = 0;
        }
    }
    else
        row[5] = 0;
    if(s.fuel_percentage < l->min_fuel_percentage)
    {
        row[6] = std::min(row[6]+1,(int)(l->fuel_violation_time));
        if(row[6] == (int)(l->fuel_violation_time))
        {
            anomaly_flag |= (1<<6);
            b[6] = true;
            row[6] = 0;
        }
    }
    else
        row[6] = 0;
    if(s.hard_brake)
    {
        row[7] = std::min(row[7]+1,(int)(l->brake_violation_time));
        if(row[7] == (int)(l->brake_violation_time))
        {
            anomaly_flag |= (1<<7);
            b[7] = true;
            row[7] = 0;
        }
    }
    else
        row[7] = 0;
    if(s.speed > l->max_speed)
    {
        row[8] = std::min(row[8]+1,(int)l->speed_violation_time);
        if(row[8] == (int)l->speed_violation_time)
        {
            anomaly_flag |= (1 << 8);
            b[8] = true;
            row[8] = 0;
        }
    }
    else 
        row[8] = 0;
    if(!s.door_lock)
    {
        row[9] = std::min(row[9]+1,(int)(l->door_violation_time));
        if(row[9] == (int)(l->door_violation_time))
        {
            anomaly_flag |= (1<<9);
            b[9] = true;
            row[9] = 0;
        }
    }
    else
        row[9] = 0;
    if(s.hard_steer)
    {
        row[10] = std::min(row[10]+1,(int)(l->steer_violation_time));
        if(row[10] == (int)(l->steer_violation_time))
        {
            anomaly_flag |= (1<<10);
            b[10] = true;
            row[10] = 0;
        }
    }
    if(anomaly_flag != 0)
    {
        char c[20];
        // std::cout<<"Anomaly detected for "<<s.vehicle_id<<" and is ";
        // for(int j=0;j<10;j++)
        //     std::cout<<b[j];
        // std::cout<<'\n';
        if(b[0] || b[1] || b[2] || b[3] || b[4] || b[5])
        {
            request_body* rb = new request_body(2,s.vehicle_id,anomaly_flag);
            write(request_file_descriptor[1],rb,sizeof(request_body));//this will take care of signalling and updating the rest.
        }
        if(b[6])
        {
            request_body* rb = new request_body(3,s.vehicle_id,anomaly_flag);
            write(request_file_descriptor[1],rb,sizeof(request_body));//this will take care of signalling and updating the rest.
        }
        if(!b[0] && !b[1] && !b[2] && !b[3] && !b[4] && !b[5] && !b[6])
        {
            sprintf(c,"shm_1_%d",s.vehicle_id);
            int fd = shm_open(c,O_CREAT|O_RDWR,0666);
            ftruncate(fd,sizeof(int));
            int* ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            *ptr = 1;
            close(fd);
            c[4] = '2';
            fd = shm_open(c,O_CREAT|O_RDWR,0666);
            ftruncate(fd,sizeof(int));
            ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            *ptr = anomaly_flag;
            close(fd);//close file descriptor.
            kill(s.vehicle_id,SIGUSR1);//send a signal now itself. 
        }   
    }
}

std::vector<Schema> Table::select(SelectQuery* select_query)
{
    mtx.lock();
    Schema* selectedValues;
    int* endIndexSelectedValues;
    Schema* retArr;
    int size;
    cudaMalloc((void**)&selectedValues, numberOfRows*sizeof(Schema));//row indices that were selected 
    cudaMalloc((void**)&endIndexSelectedValues, sizeof(int));
    cudaMemset(endIndexSelectedValues,0,sizeof(int));//set this value to zero.        
    init_bt(numberOfRows);
    //std::cout<<"Acquired lock, going for kernel launch with "<<nb<<" "<<nt<<" and "<<numberOfRows<<" number of Rows!\n";
    selectKernel<<<nb, nt>>>(
            StateDatabase,
            numberOfRows,
            selectedValues,
            endIndexSelectedValues,
            select_query
        );
    cudaDeviceSynchronize();
    //std::cout<<"Kernel Launch done!\n";
    cudaMemcpy(&size, endIndexSelectedValues, sizeof(int), cudaMemcpyDeviceToHost);
    if(size == 0){
        std::vector<Schema> s1;
        return s1;
    }
    retArr = (Schema*)malloc(sizeof(Schema)*size);
    cudaMemcpy(retArr, selectedValues, size*sizeof(Schema), cudaMemcpyDeviceToHost);
    cudaFree(selectedValues);
    mtx.unlock();
    std::vector<Schema> result;
    //std::cout<<"Got here!"<<std::endl;
    if(select_query->distinct)
    {
        std::set<Schema,distinct_comparator> dis_set(result.begin(),result.end(),distinct_comparator(select_query));
        for(int i=0;i < size;i++)
            dis_set.insert(retArr[i]);
        for(Schema s: dis_set)
            result.push_back(s);
    }
    else
    {
        for(int i =0; i < size;i++)
            result.push_back(retArr[i]);
    }
    //std::cout<<"After distinct!"<<std::endl;
    if(select_query->order_term != NULL && select_query->order_term->size() >= 1)
        std::sort(result.begin(),result.end(),select_comparator(select_query));
    int ms = (select_query->limit_term <= 0)?(result.size()):(std::min((int)(result.size()),(int)select_query->limit_term));
    result.resize(ms);
    return result;
}

std::pair<std::vector<std::vector<std::pair<double,double>>>,std::vector<std::string>> Table::aggregate_select(SelectQuery* select_query)
{
    std::vector<Schema> selected_rows = select(select_query);
    //now proceed to process
    std::vector<std::vector<std::pair<double,double>>> v1;
    std::vector<std::string> s;
    if(selected_rows.size() == 0)
        return std::make_pair(v1,s);
    if(select_query->group_term != NULL)
    {
        bool group_is_double = (select_query->group_term->type_of_expr == 3); 
        for(std::pair<char*,ExpressionNode*> it: *(select_query->aggregate_columns))
        {
            std::vector<std::pair<double,double>> q;
            s.push_back(it.first);
            bool param_is_double;
            if(str_equal(it.first,"max"))
            {   
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,double> max_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    if(max_map.find(t) == max_map.end())
                    {
                        if(!param_is_double)
                            max_map[t] = (double)it.second->evaluate_int_expression(obj);
                        else
                            max_map[t] = it.second->evaluate_double_expression(obj);
                    }
                    else
                    {
                        if(!param_is_double)
                            max_map[t] = std::max(max_map[t],(double)it.second->evaluate_int_expression(obj));
                        else
                            max_map[t] = std::max(max_map[t],it.second->evaluate_double_expression(obj));
                    } 
                }
                for(auto it: max_map)
                    q.push_back(it);
            }
            else if(str_equal(it.first,"min"))
            {
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,double> min_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    if(min_map.find(t) == min_map.end())
                    {
                        if(!param_is_double)
                            min_map[t] = (double)it.second->evaluate_int_expression(obj);
                        else
                            min_map[t] = it.second->evaluate_double_expression(obj);
                    }
                    else
                    {
                        if(!param_is_double)
                            min_map[t] = std::min(min_map[t],(double)it.second->evaluate_int_expression(obj));
                        else
                            min_map[t] = std::min(min_map[t],it.second->evaluate_double_expression(obj));
                    } 
                }
                for(auto it: min_map)
                    q.push_back(it);
            }
            else if(str_equal(it.first,"avg"))
            {
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,double> sum_map;
                std::map<double,int> count_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    if(sum_map.find(t) == sum_map.end())
                    {
                        if(!param_is_double)
                        {
                            sum_map[t] = (double)it.second->evaluate_int_expression(obj);
                            count_map[t]++;
                        }
                        else
                        {
                            sum_map[t] = it.second->evaluate_double_expression(obj);
                            count_map[t]++;
                        }
                    }
                    else
                    {
                        if(!param_is_double)
                        {
                            sum_map[t] += (double)it.second->evaluate_int_expression(obj);
                            count_map[t]++;
                        }
                        else
                        {
                            sum_map[t] += it.second->evaluate_double_expression(obj);
                            count_map[t]++;
                        }
                    } 
                }
                for(auto it: sum_map)
                {
                    q.push_back(std::make_pair(it.first,it.second/count_map[it.first]));
                }
            }
            else if(str_equal(it.first,"std"))
            {
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,std::vector<double>> std_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    
                    if(!param_is_double)
                    {
                        std_map[t].push_back((double)it.second->evaluate_int_expression(obj));
                    }
                    else
                    {
                        std_map[t].push_back(it.second->evaluate_double_expression(obj));
                    }
                }
                for(auto it: std_map)
                {
                    double sum = 0;
                    for(auto it1: it.second)
                        sum += it1;
                    sum /= it.second.size();
                    double std_dev = 0;
                    for(auto it1: it.second)
                        std_dev += pow(it1-sum,2);
                    std_dev = sqrt(std_dev/it.second.size());
                    q.push_back(std::make_pair(it.first,std_dev));
                }
                //edit for standard dev
            }
            else if(str_equal(it.first,"var"))
            {
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,std::vector<double>> std_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    
                    if(!param_is_double)
                    {
                        std_map[t].push_back((double)it.second->evaluate_int_expression(obj));
                    }
                    else
                    {
                        std_map[t].push_back(it.second->evaluate_double_expression(obj));
                    }
                }
                for(auto it: std_map)
                {
                    double sum = 0;
                    for(auto it1: it.second)
                        sum += it1;
                    sum /= it.second.size();
                    double std_dev = 0;
                    for(auto it1: it.second)
                        std_dev += pow(it1-sum,2);
                    std_dev /= it.second.size();
                    q.push_back(std::make_pair(it.first,std_dev));
                }
            }
            else if(str_equal(it.first,"sum"))
            {
                param_is_double = (it.second->type_of_expr == 3);
                std::map<double,double> sum_map;
                for(Schema obj: selected_rows)
                {
                    double t;
                    if(group_is_double)
                        t = select_query->group_term->evaluate_double_expression(obj);
                    else
                        t = (double)select_query->group_term->evaluate_int_expression(obj);
                    
                    if(!param_is_double)
                    {
                        double u = (double)it.second->evaluate_int_expression(obj);
                        sum_map[t] += u;
                    }
                    else
                    {
                        double u = it.second->evaluate_double_expression(obj);
                        sum_map[t] += u;
                    }
                    
                }
                for(auto it: sum_map)
                {
                    q.push_back(it);
                }
            }
            v1.push_back(q);
        }
        return std::make_pair(v1,s);
    }
    for(std::pair<char*,ExpressionNode*> it: *(select_query->aggregate_columns)){
        std::vector<std::pair<double,double>> v_r;
        s.push_back(it.first);
        if(str_equal(it.first,"max"))
        {
            double maxi = -DBL_MAX;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                maxi = std::max(maxi,x);
            }
            v_r.push_back(std::make_pair(0.0,maxi));
        }
        else if(str_equal(it.first,"min"))
        {   
            double mini = DBL_MAX;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                mini = std::min(mini,x);
            }
            v_r.push_back(std::make_pair(0.0,mini));
        }
        else if(str_equal(it.first,"avg"))
        {
            double avg = 0.0;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                avg += x;
            }
            avg /= selected_rows.size();
            v_r.push_back(std::make_pair(0.0,avg));
        }
        else if(str_equal(it.first,"sum"))
        {
            double sum = 0.0;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                sum += x;
            }
            v_r.push_back(std::make_pair(0.0,sum));
        }
        else if(str_equal(it.first,"var"))
        {
            std::vector<double> v;
            double sum = 0;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                sum += x;
                v.push_back(x);
            }
            sum /= selected_rows.size();
            double var = 0;
            for(auto it: v)
                var += pow(it-sum,2);
            var /= selected_rows.size();
            v_r.push_back(std::make_pair(0.0,var));
        }
        else if(str_equal(it.first,"std"))
        {
            std::vector<double> v;
            double sum = 0;
            for(auto schema_object: selected_rows)
            {
                double x;
                if(it.second->type_of_expr == 2)
                    x = it.second->evaluate_int_expression(schema_object);
                else
                    x = it.second->evaluate_double_expression(schema_object);
                sum += x;
                v.push_back(x);
            }
            sum /= selected_rows.size();
            double var = 0;
            for(auto it: v)
                var += pow(it-sum,2);
            var /= selected_rows.size();
            double std_dev = sqrt(var);
            v_r.push_back(std::make_pair(0.0,std_dev));
        }
        v1.push_back(v_r);
    }
    return std::make_pair(v1,s);
}

std::vector<Schema> Table::normal_select(SelectQuery* select_query)//parse the query and filter out what's relevant. 
{
    //acquire a lock before writing? Mostly needed, can be a lock using mutex across both threads. 
    return select(select_query);
}

void Table::PrintDatabase()
{
    //iterate through the table and print out.
    mtx.lock();
    Schema* arr = (Schema*)malloc(sizeof(Schema)*numberOfRows);
    cudaMemcpy(arr,StateDatabase,numberOfRows*sizeof(Schema),cudaMemcpyDeviceToHost);
    for(int i = 0;i < numberOfRows;i++)
    {
        std::cout<<"Vehicle ID = "<<arr[i].vehicle_id<<'\n';
    } 
    mtx.unlock();
}


void GPSSystem::init_bt(int numThreads)
{
    nb = std::ceil(numThreads/1024.0);
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
    int* hostDroppedVertices = (int*)malloc(sizeof(int)*numberOfDroppedVertices);
    int* deviceDroppedVertices;
    int* deviceAdjacencyMatrix;
    int idx = 0;
    for(auto vertex: setDroppedVertices)
    {
        hostDroppedVertices[idx++] = vertex;
    }
    int fd = shm_open("adjacency_matrix",O_RDONLY,0666);
    int* hostAdjacencyMatrix = (int*)mmap(0,numberOfVertices*numberOfVertices*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
    close(fd);
    cudaMalloc((void**)&deviceDroppedVertices, numberOfDroppedVertices*sizeof(int));
    cudaMemcpy(deviceDroppedVertices, hostDroppedVertices, numberOfDroppedVertices*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&deviceAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int));
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
        int* hostMinDistance = (int*)malloc(sizeof(int));
        *hostMinDistance = INT_MAX;
        cudaMalloc(&minDistance, sizeof(int));        
        int* argMinVertex;
        cudaMalloc(&argMinVertex, sizeof(int));
        int* deviceUsed;
        int* hostUsed = (int*)malloc(numberOfVertices*sizeof(int));
        for(int i = 0; i < numberOfVertices; i++){
            hostUsed[i] = 0;
        }
        int* deviceDistance;
        int* hostDistance = (int*)malloc(sizeof(int)*numberOfVertices);
        int* hostParent = (int*)malloc(numberOfVertices*sizeof(int));
        for(int i = 0; i < numberOfVertices; i ++){
            hostDistance[i] = ((i == source)?0:INT_MAX);
            hostParent[i] = -1;
        }
        int* deviceParent;
        cudaMalloc((void**)&deviceUsed, numberOfVertices*sizeof(int));
        cudaMalloc((void**)&deviceDistance, numberOfVertices*sizeof(int));
        cudaMalloc((void**)&deviceParent, numberOfVertices*sizeof(int));
        cudaMemcpy(deviceParent,hostParent,numberOfVertices*sizeof(int),cudaMemcpyHostToDevice);
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
    cudaFree(deviceDistance);
    cudaFree(deviceParent);
    cudaFree(deviceAdjacencyMatrix);
    cudaFree(deviceUsed);
    // for(auto it: setDroppedVertices)
    //     std::cout<<"Dropping "<<it<<'\n';
    return std::make_pair(hostParent,hostDistance);
}

GPSSystem::GPSSystem(int numVert, int* initMat){
    numberOfVertices = numVert;
    hostAdjacencyMatrix = (int*)malloc(numberOfVertices*numberOfVertices*sizeof(int));
    for(int i = 0; i < numberOfVertices*numberOfVertices; i++){
        hostAdjacencyMatrix[i] = initMat[i];
}
}

void GPSSystem::convoyNodeFinder(std::map<int,int>& car_ids)
{
    //djikstras kernel call,and then cumulatively add those distances. Then check the city with least sum of 
    //distance and ask cars to converge there.
    double* sum_array;
    cudaMalloc((void**)&sum_array,numberOfVertices*sizeof(double));
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
            if(y <= 0.2)//randomly dropped, can be adjusted. 
                droppedVertices.insert(i);
        }
    }
    std::vector<int*> parent_array;
    std::pair<int*,int*> p;
    init_bt(numberOfVertices);
    //std::cout<<included_vertices.size()<<"\n";
    int* device_distances;
    cudaMalloc((void**)&device_distances,numberOfVertices*sizeof(int));
    std::map<int,int> vertex_map;
    int count = 0;
    for(int i: included_vertices)
    {   
        vertex_map[i] = count;
        count++;
        p = djikstra_result(i,droppedVertices);
        parent_array.push_back(p.first);
        cudaMemcpy(device_distances,p.second,numberOfVertices*sizeof(int),cudaMemcpyHostToDevice);
        addMatrix<<<nb,nt>>>(sum_array,device_distances);
        cudaDeviceSynchronize();
    }
    cudaFree(device_distances);
    //std::cout<<"Yes, solved it!\n"<<std::endl;
    int min_index = thrust::min_element(thrust::device,sum_array,sum_array+numberOfVertices) - sum_array;
    //now write to shared memory and send a signal to each car.
    cudaFree(sum_array);
    for(std::pair<int,int> p: car_ids)//gives their respective current positions.
    {
        std::vector<int> path;
        int curr = min_index;
        while(curr != -1)
        {
            //std::cout<<curr<<'\n';
            path.push_back(curr);
            curr = parent_array[vertex_map[p.second]][curr];
        }
        std::reverse(path.begin(),path.end());
        std::cout<<"Path for "<<p.first<<": ";
        std::copy(path.begin(),path.end(),std::ostream_iterator<int>(std::cout," "));
        std::cout<<'\n';
        if(path.size() == 1)
            continue;
        char c[20];
        sprintf(c,"shm_1_%d",p.first);
        int fd = shm_open(c,O_CREAT|O_RDWR,0666);
        ftruncate(fd,sizeof(int));
        int* ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = 2;
        close(fd);
        c[4] = '3';
        fd = shm_open(c,O_CREAT|O_RDWR,0666);
        ftruncate(fd,sizeof(int));
        ptr = (int*)mmap(0,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        *ptr = path.size();
        close(fd);
        c[4] = '4';
        fd = shm_open(c,O_CREAT|O_RDWR,0666);
        ftruncate(fd,path.size()*sizeof(int));
        ptr = (int*)mmap(0,path.size()*sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        for(int i = 0;i < path.size();i++)
            ptr[i] = path[i];
        close(fd);
        kill(p.first,SIGUSR1);
        //std::cout<<"Sending to "<<p.first<<std::endl;
        //update the path here by writing to shared memory. 
    }
}

std::vector<int> GPSSystem::findGarageOrBunk(int source,int target_type,std::set<int>& setDroppedVertices){
    std::pair<int*,int*> value = djikstra_result(source, setDroppedVertices);
    int* hostParent = value.first;
    int* hostDistance = value.second;
    int fd = shm_open("vertex_type",O_RDONLY,0666);
    int* arr = (int*)mmap(0,numberOfVertices*sizeof(int),PROT_READ,MAP_SHARED,fd,0);
    int min_dist = INT_MAX;
    int path_v = -1;
    for(int i = 0;i < numberOfVertices;i++)
    {
        if(arr[i] == target_type && hostDistance[i] < min_dist)
        {
            min_dist = hostDistance[i];
            path_v = i;
        }
    }
    std::vector<int> path;
    if(min_dist == INT_MAX)
        return path;
    int currentVertex = path_v;
    while(currentVertex != -1){
        path.push_back(currentVertex);
        currentVertex = hostParent[currentVertex];
    }
    std::reverse(path.begin(), path.end());
    // for(int i = 0;i < numberOfVertices;i++)
    //     std::cout<<hostParent[i]<<' ';
    // std::cout<<"\n";
    // for(int it: path)
    //     std::cout<<it<<" ";
    // std::cout<<'\n';
    close(fd);
    return path;
}

__host__ __device__ bool str_equal(const char* s1, const char* s2)
{
    if(s1 == NULL || s2 == NULL)
        return (s1 == NULL && s2 == NULL);
    int i = 0;
    int j = 0;
    while(s1[i] != '\0')
        i++;
    while(s2[j] != '\0')
        j++;
    if(i != j)
        return false;
    for(j=0;j < i;j++)
    {
        if(s1[j] != s2[j])
            return false;
    }
    return true;
}
request_body::request_body(int type,int s_car,int af)
{
    request_type = type;
    sending_car = s_car;
    anomaly_flag = af;
}