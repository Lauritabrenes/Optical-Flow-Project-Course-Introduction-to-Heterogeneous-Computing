/*
 * lk_cuda.cu - Etapa 3: Lucas-Kanade acelerado con NVIDIA CUDA
 *
 * Mapea la resolucion de la imagen a un grid 2D de hilos.
 * Cada hilo calcula de forma independiente los gradientes y resuelve 
 * el tensor de estructura para un unico pixel.
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    #include "of.h"
    #include "gpu_timing.h"
}
#include "cuda_check.h"

#define LK_RADIUS 3
#define LK_DET_THRESHOLD 1e-6f
#define LK_MARGIN (LK_RADIUS + 1)

/* 
 * KERNEL 1: Calculo de gradientes espaciales y temporales
 */
__global__ void compute_gradients_kernel(const float* f1, const float* f2, 
                                         float* Ix, float* Iy, float* It, 
                                         int w, int h) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
        int i = y * w + x;
        Ix[i] = (f1[i + 1] - f1[i - 1]) * 0.5f;
        Iy[i] = (f1[i + w] - f1[i - w]) * 0.5f;
        It[i] = f2[i] - f1[i];
    }
}

/* 
 * KERNEL 2: Construccion del Tensor de Estructura y Regla de Cramer
 */
__global__ void solve_flow_kernel(const float* Ix, const float* Iy, const float* It, 
                                  float* u, float* v, 
                                  int w, int h) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= LK_MARGIN && x < w - LK_MARGIN && y >= LK_MARGIN && y < h - LK_MARGIN) {
        float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
        float sxt = 0.0f, syt = 0.0f;

        for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++) {
            for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                int i = (y + wy) * w + (x + wx);
                float gx = Ix[i];
                float gy = Iy[i];
                float gt = It[i];

                sxx += gx * gx;
                syy += gy * gy;
                sxy += gx * gy;
                sxt += gx * gt;
                syt += gy * gt;
            }
        }

        float det = sxx * syy - sxy * sxy;
        int o = y * w + x;

        if (det > LK_DET_THRESHOLD) {
            u[o] = (-syy * sxt + sxy * syt) / det;
            v[o] = ( sxy * sxt - sxx * syt) / det;
        }
    }
}

#define TRY(call) do { e = (call); if (e != cudaSuccess) {                   \
        fprintf(stderr, "CUDA error %s:%d: %s -> %s\n", __FILE__, __LINE__,  \
                #call, cudaGetErrorString(e)); rc = -3; goto fail; } } while (0)

/*
 * FUNCION HOST: llamada desde C (main.c)
 * - Verificacion de errores en todas las llamadas CUDA (TRY).
 * - Desglose de tiempos con cudaEvent: alloc / H2D / kernel / D2H / free
 *   (resultado en g_gpu_timing).
 * - La memoria se libera siempre, incluso tras un error.
 */
extern "C" int of_lk_cuda(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
    if (!frame1 || !frame2 || !flow) return -1;

    const int w = frame1->width;
    const int h = frame1->height;
    const size_t img_size = (size_t)w * (size_t)h * sizeof(float);

    float *d_f1 = NULL, *d_f2 = NULL, *d_Ix = NULL, *d_Iy = NULL, *d_It = NULL, *d_u = NULL, *d_v = NULL;
    int rc = 0;
    cudaError_t e;
    cudaEvent_t ev[6];
    int k;

    for (k = 0; k < 6; k++) CUDA_CHECK(cudaEventCreate(&ev[k]));

    /* ---- [3a] ALLOC ---- */
    TRY(cudaEventRecord(ev[0]));
    TRY(cudaMalloc(&d_f1, img_size));
    TRY(cudaMalloc(&d_f2, img_size));
    TRY(cudaMalloc(&d_Ix, img_size));
    TRY(cudaMalloc(&d_Iy, img_size));
    TRY(cudaMalloc(&d_It, img_size));
    TRY(cudaMalloc(&d_u, img_size));
    TRY(cudaMalloc(&d_v, img_size));
    TRY(cudaMemset(d_u, 0, img_size));
    TRY(cudaMemset(d_v, 0, img_size));
    TRY(cudaMemset(d_Ix, 0, img_size));
    TRY(cudaMemset(d_Iy, 0, img_size));
    TRY(cudaMemset(d_It, 0, img_size));

    /* ---- [3b] H2D ---- */
    TRY(cudaEventRecord(ev[1]));
    TRY(cudaMemcpy(d_f1, frame1->data, img_size, cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(d_f2, frame2->data, img_size, cudaMemcpyHostToDevice));

    /* ---- [3c] KERNEL ---- */
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
        TRY(cudaEventRecord(ev[2]));
        compute_gradients_kernel<<<gridSize, blockSize>>>(d_f1, d_f2, d_Ix, d_Iy, d_It, w, h);
        TRY(cudaGetLastError());
        solve_flow_kernel<<<gridSize, blockSize>>>(d_Ix, d_Iy, d_It, d_u, d_v, w, h);
        TRY(cudaGetLastError());
        TRY(cudaEventRecord(ev[3]));
        TRY(cudaEventSynchronize(ev[3]));   /* errores de ejecucion aparecen aqui */
    }

    /* ---- [3d] D2H ---- */
    TRY(cudaMemcpy(flow->u, d_u, img_size, cudaMemcpyDeviceToHost));
    TRY(cudaMemcpy(flow->v, d_v, img_size, cudaMemcpyDeviceToHost));
    TRY(cudaEventRecord(ev[4]));

fail:
    /* ---- [3e] FREE (siempre) ---- */
    cudaFree(d_f1);
    cudaFree(d_f2);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_It);
    cudaFree(d_u);
    cudaFree(d_v);

    if (rc == 0) {
        cudaEventRecord(ev[5]);
        cudaEventSynchronize(ev[5]);
        cudaEventElapsedTime(&g_gpu_timing.alloc_ms,  ev[0], ev[1]);
        cudaEventElapsedTime(&g_gpu_timing.h2d_ms,    ev[1], ev[2]);
        cudaEventElapsedTime(&g_gpu_timing.kernel_ms, ev[2], ev[3]);
        cudaEventElapsedTime(&g_gpu_timing.d2h_ms,    ev[3], ev[4]);
        cudaEventElapsedTime(&g_gpu_timing.free_ms,   ev[4], ev[5]);
    }
    for (k = 0; k < 6; k++) cudaEventDestroy(ev[k]);
    return rc;
}
#undef TRY
