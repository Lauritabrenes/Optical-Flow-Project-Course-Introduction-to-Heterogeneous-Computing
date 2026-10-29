/*
 * cuda_check.h - Macros de verificacion de errores CUDA
 */
#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>

/* Verifica una llamada CUDA; si falla imprime archivo/linea y retorna -3. */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s -> %s\n",                  \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err__));   \
            return -3;                                                       \
        }                                                                    \
    } while (0)

#endif
