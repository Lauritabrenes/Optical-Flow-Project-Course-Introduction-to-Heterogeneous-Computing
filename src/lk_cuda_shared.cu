/*
 * lk_cuda_shared.cu - Etapa 3: Acelerador II (Memoria Compartida Cooperativa)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {
    #include "of.h"
}

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

extern "C" int of_lk_cuda_shared(const OFImage *frame1, const OFImage *frame2, OFFlow *flow) 
{
    if (!frame1 || !frame2 || !flow) return -1;

    int w = frame1->width;
    int h = frame1->height;
    size_t img_size = w * h * sizeof(float);

    float *d_f1, *d_f2, *d_u, *d_v;

    if (cudaMalloc(&d_f1, img_size) != cudaSuccess) return -2;
    if (cudaMalloc(&d_f2, img_size) != cudaSuccess) return -2;
    if (cudaMalloc(&d_u, img_size) != cudaSuccess) return -2;
    if (cudaMalloc(&d_v, img_size) != cudaSuccess) return -2;

    cudaMemcpy(d_f1, frame1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, frame2->data, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_u, 0, img_size);
    cudaMemset(d_v, 0, img_size);

    dim3 blockSize(TILE_W, TILE_H);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    solve_flow_shared_kernel<<<gridSize, blockSize>>>(d_f1, d_f2, d_u, d_v, w, h);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -3;
    }

    cudaMemcpy(flow->u, d_u, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(flow->v, d_v, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_f1); cudaFree(d_f2);
    cudaFree(d_u); cudaFree(d_v);

    return 0;
}