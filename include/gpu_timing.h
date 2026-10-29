/*
 * gpu_timing.h - Desglose de tiempos de la etapa GPU
 */
#ifndef GPU_TIMING_H
#define GPU_TIMING_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float alloc_ms;    /* cudaMalloc + cudaMemset   */
    float h2d_ms;      /* Host -> Device            */
    float kernel_ms;   /* ejecucion de kernel(s)    */
    float d2h_ms;      /* Device -> Host            */
    float free_ms;     /* cudaFree                  */
} OFGpuTiming;

extern OFGpuTiming g_gpu_timing;

/* Crea el contexto CUDA fuera de la region medida. */
int of_gpu_warmup(void);

#ifdef __cplusplus
}
#endif

#endif
