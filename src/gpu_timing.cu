/*
 * gpu_timing.cu - Variable global de tiempos y warm-up del contexto CUDA
 */
#include <cuda_runtime.h>
#include "gpu_timing.h"
#include "cuda_check.h"

extern "C" {

OFGpuTiming g_gpu_timing = {0};

int of_gpu_warmup(void)
{
    CUDA_CHECK(cudaFree(0));            /* fuerza la creacion del contexto */
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

}
