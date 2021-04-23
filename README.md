# Run RK3399 Open

## PREPARATION
1. Please put all the model files under `models` folder
2. Change the path of models of json files under `jsonconfig` folder
3. __Modify the `CMakeLists.txt` file, Line 15, choose the `.cpp` file that you would like to run.
For example, `main.cpp` shows a sample program

## BUILD AND RUN

```bash
mkdir build
cd build
cmake ..
make 
./example_gesture_app
```

## MORE INFO

1. `include/` folder contains 3 header files online, which implements thread pool in 
c++ 11, please feel free to use any of them or you can implement your own multi-thread
pool
2. `lib/` folder contains previous dynamic library files, but you should only use 
`libkneron_gesture_sdk.so`. Replace this file if it is updated in the future.
3. `example/` folder contains all the driver `main.cpp` in which different usages and
sample codes are shown.
4. `jsonconfig/` folder contains multiple necessary json files, please modify the path
of models accordingly.
5. `models/` should contain all the neural network models provided, please put those
right under the folder.
