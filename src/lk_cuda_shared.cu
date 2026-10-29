/*
 * lk_cuda_shared.cu - Etapa 3: Acelerador II (Memoria Compartida Cooperativa)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {
    #include "of.h"
    #include "gpu_timing.h"
}
#include "cuda_check.h"

#define LK_RADIUS 3
#define LK_DET_THRESHOLD 1e-6f
#define LK_MARGIN (LK_RADIUS + 1)

#define TILE_W 16
#define TILE_H 16
/* CORRECCION CRITICA: El halo debe ser igual al MARGEN (4), no al RADIO (3),
   porque calcular el gradiente exige leer f[x+1] y f[x-1] */
#define SHARED_W (TILE_W + 2 * LK_MARGIN)
#define SHARED_H (TILE_H + 2 * LK_MARGIN)

__global__ void solve_flow_shared_kernel(const float* f1, const float* f2, 
                                         float* u, float* v, 
                                         int w, int h) 
{
    __shared__ float s_f1[SHARED_H][SHARED_W];
    __shared__ float s_f2[SHARED_H][SHARED_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    /* Carga Cooperativa a Memoria Compartida */
    int tid = ty * TILE_W + tx; 
    int num_threads = TILE_W * TILE_H; 
    int shared_size = SHARED_W * SHARED_H; 

    for (int i = tid; i < shared_size; i += num_threads) {
        int sy = i / SHARED_W;
        int sx = i % SHARED_W;
        
        int global_y = blockIdx.y * TILE_H + sy - LK_MARGIN;
        int global_x = blockIdx.x * TILE_W + sx - LK_MARGIN;
        
        global_x = min(max(global_x, 0), w - 1);
        global_y = min(max(global_y, 0), h - 1);
        
        s_f1[sy][sx] = f1[global_y * w + global_x];
        s_f2[sy][sx] = f2[global_y * w + global_x];
    }
    
    __syncthreads();

    if (x >= LK_MARGIN && x < w - LK_MARGIN && y >= LK_MARGIN && y < h - LK_MARGIN) {
        float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
        float sxt = 0.0f, syt = 0.0f;

        /* Centro de este hilo compensado por el MARGEN */
        int sx = tx + LK_MARGIN;
        int sy = ty + LK_MARGIN;

        for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++) {
            for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                
                float gx = (s_f1[sy + wy][sx + wx + 1] - s_f1[sy + wy][sx + wx - 1]) * 0.5f;
                float gy = (s_f1[sy + wy + 1][sx + wx] - s_f1[sy + wy - 1][sx + wx]) * 0.5f;
                float gt = s_f2[sy + wy][sx + wx] - s_f1[sy + wy][sx + wx];

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
extern "C" int of_lk_cuda_shared(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
    if (!frame1 || !frame2 || !flow) return -1;

    const int w = frame1->width;
    const int h = frame1->height;
    const size_t img_size = (size_t)w * (size_t)h * sizeof(float);

    float *d_f1 = NULL, *d_f2 = NULL, *d_u = NULL, *d_v = NULL;
    int rc = 0;
    cudaError_t e;
    cudaEvent_t ev[6];
    int k;

    for (k = 0; k < 6; k++) CUDA_CHECK(cudaEventCreate(&ev[k]));

    /* ---- [3a] ALLOC ---- */
    TRY(cudaEventRecord(ev[0]));
    TRY(cudaMalloc(&d_f1, img_size));
    TRY(cudaMalloc(&d_f2, img_size));
    TRY(cudaMalloc(&d_u, img_size));
    TRY(cudaMalloc(&d_v, img_size));
    TRY(cudaMemset(d_u, 0, img_size));
    TRY(cudaMemset(d_v, 0, img_size));

    /* ---- [3b] H2D ---- */
    TRY(cudaEventRecord(ev[1]));
    TRY(cudaMemcpy(d_f1, frame1->data, img_size, cudaMemcpyHostToDevice));
    TRY(cudaMemcpy(d_f2, frame2->data, img_size, cudaMemcpyHostToDevice));

    /* ---- [3c] KERNEL ---- */
    {
        dim3 blockSize(TILE_W, TILE_H);
        dim3 gridSize((w + TILE_W - 1) / TILE_W, (h + TILE_H - 1) / TILE_H);
        TRY(cudaEventRecord(ev[2]));
        solve_flow_shared_kernel<<<gridSize, blockSize>>>(d_f1, d_f2, d_u, d_v, w, h);
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
