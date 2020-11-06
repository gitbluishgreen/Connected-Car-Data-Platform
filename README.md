# Connected-Car-Data-Platform
This is Arjun's version of the code.
The current code ahs the following features:
1. A parallelised read mechanism on the database.
2. Query parser to read select queries.
3. Message passing between cars and server, and among cars themselves using shared memory. 
4. Anomaly detection and correction in a cars's health parameters.
5. Handling shortest path queries for a convoy of cars.

Features that are being added:
1. Table partitioning to allow storage of messages for longer durations.
2. Overall bug fixes.

Features that can further be added:
1. OpenGL visualisation of the convoy effect/shortest-path algorithm.
2. Time regulated traffic signals as an alternative to the current randomized mechanism. 
