# Connected-Car-Data-Platform
This is Arjun's version of the code.
The current code ahs the following features:
1. A parallelised read mechanism on the database.
2. Query parser to read select queries.
3. Message passing between cars and server, and among cars themselves using shared memory. 
4. Anomaly detection and correction in a cars's health parameters.
5. Handling shortest path queries for a convoy of cars.

Features that can further be added:
1. OpenGL visualisation of the convoy effect/shortest-path algorithm.
2. Time regulated traffic signals as an alternative to the current randomized mechanism. 

## How do you run this project?

1. Simply type `make all` in bash. This builds the repository into a target `app`.
2. The `Tests` folder contains test cases against which the code can be run. The command line argument passed to `app` should be a valid file in the tests folder. 
3. For example, you can run `./app Tests/test1.txt`.