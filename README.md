# Connected-Car-Data-Platform
## What is this project about?
This is the course project for CS6023: GPU Programming done by Arjun Bharat and Vamsi Krishna.
The goal was to design and simulate a connected car data platform consisting of cars that periodically send status updates to a central server. The server consists of the following:
1. A parallelised read and write mechanism on a database consisting of car data.
2. A query parser to allow a user to select data from the database.
3. Anomaly detection and correction in a car's health parameters, such as low fuel, oil levels or battery voltage.
4. Convoy queries to help a set of cars converge at a particular point in the city graph. 

Features that can further be added in the future:
1. OpenGL visualisation of the convoy effect/shortest-path algorithm.
2. Time regulated traffic signals as an alternative to the current randomized mechanism of dropped vertices. 

## Query Syntax:

See the file `query.txt` for example queries.
Here **?** denotes an optionalclause in the query.
1. For Ordinary Select Queries: 
    **SELECT Columns (WHERE EXP1)? (ORDER BY EXP2)? (LIMIT val)?**
    Here Columns can be any comma-separated values among **{vehicle_id,oil_life_pct,tire_p_rl,tire_p_rr,tire_p_fl,tire_p_fr,batt_volt,fuel_percentage;accel,seatbelt,hard_brake,door_lock,gear_toggle,clutch,hard_steer,speed,distance,origin_vertex,destination_vertex}** or just an asterisk '*'(all columns).

    vehicle_id,origin_vertex and destination_vertex are integers. accel,seatbelt,hard_brake,door_lock,gear_toggle,clutch,hard_steer are boolean values. The rest are floating point values.
2. For Aggregate Select Queries:
    **SELECT Aggregate_Columns (WHERE EXP1)? (GROUP BY EXP2)? (ORDER BY EXP3)? (LIMIT val)?**
    Aggregate Columns are many comma separated terms of the form **Aggregate(EXP)** for an expression **EXP**. Allowed aggregates are **{MIN,MAX,SUM,AVERAGE,VARIANCE,STDDEV}**

3. For Convoy Queries:
    **CONVOY n car_1 car_2 ...car_n** 
    Here car_1, car_2 .. car_n are the indices of the cars involved in the query, in the range [1,number_of_cars].

Expressions should be properly bracketed and on the columns of the database, and the allowed operators are +,-,*,/,%,>,<,>=,<=,==,!=,&&,||,! (which have their usual meaning).
## How do you run this project?

1. Simply type `make all` in bash. This builds the repository into a target `app`.
2. The `Tests` folder contains test cases against which the code can be run. The command line argument passed to `app` should be a valid test case file. The format of a test case file is as follows:
- Number_of_rows Max_worklist_size Number_of cars Number_of_vertices(N)
- (N x N adjacency matrix weights, can be asymmetric but must be positive integral)
- (N numbers denoting the type of each vertex: 1 is a an ordinary node, 2 is a garage and 3 is a fuel station. You need at least one garage and one fuel station to service anomaly requests properly.)
3. For example, you can run `./app Tests/test1.txt`.
4. You may enter queries in interactive mode, either to select data or to make a convoy request.