cmake_minimum_required(VERSION 3.16)
project(optical_flow_prototype C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(bench_main
    src/main.c
    src/io.c
    src/preprocess.c
    src/lk_scalar.c
    src/lk_neon.c
    src/metrics.c
)