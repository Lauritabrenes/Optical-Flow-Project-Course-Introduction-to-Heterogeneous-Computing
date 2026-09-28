cmake_minimum_required(VERSION 3.16)
project(optical_flow_prototype C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

# Agregar después de la línea 6 en CMakeLists.txt
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(-O3 -march=armv8-a+simd)
    add_definitions(-DHAVE_NEON)
endif()

add_executable(test_metrics tests/test_metrics.c src/metrics.c)
if(NOT MSVC)
    target_link_libraries(test_metrics PRIVATE m)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(
    include
    third_party
)

add_executable(bench_main
    src/main.c
    src/io.c
    src/preprocess.c
    src/lk_scalar.c
    src/lk_neon.c
    src/metrics.c
)

if(NOT MSVC)
    target_link_libraries(
        bench_main
        PRIVATE
        m
    )
endif()

add_executable(test_io
    tests/test_io.c
    src/io.c
)

if(NOT MSVC)
    target_link_libraries(
        test_io
        PRIVATE
        m
    )
endif()