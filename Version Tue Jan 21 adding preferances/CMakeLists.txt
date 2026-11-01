cmake_minimum_required(VERSION 3.30)
project(PX5net2 C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(PX5net2
    main.c
    px5_neural_network.c
    px5_neural_network.h)
