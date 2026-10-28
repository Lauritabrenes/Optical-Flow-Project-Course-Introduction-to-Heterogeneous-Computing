/*
 * lk_cuda_opt.cu - Etapa 3: Lucas-Kanade Optimizado (Acelerador II)
 *
 * Optimizaciones aplicadas:
 * 1. Fusión de Kernels: Gradientes calculados al vuelo para evitar VRAM.
 * 2. Eliminacion de tensores intermedios (Ix, Iy, It) en memoria global.
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    #include "of.h"
}

#define LK_RADIUS 3
#define LK_DET_THRESHOLD 1e-6f
#define LK_MARGIN (LK_RADIUS + 1)

__global__ void solve_flow_opt_kernel(const float* f1, const float* f2, 
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

                /* Calculo de gradientes al vuelo en registros locales */
                float gx = (f1[i + 1] - f1[i - 1]) * 0.5f;
                float gy = (f1[i + w] - f1[i - w]) * 0.5f;
                float gt = f2[i] - f1[i];

                /* Acumulacion inmediata */
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

extern "C" int of_lk_cuda_opt(const OFImage *frame1, const OFImage *frame2, OFFlow *flow) 
{
    if (!frame1 || !frame2 || !flow) return -1;

    int w = frame1->width;
    int h = frame1->height;
    size_t img_size = w * h * sizeof(float);

    float *d_f1, *d_f2, *d_u, *d_v;

    /* 1. Asignacion drásticamente reducida (solo entrada y salida) */
    if (cudaMalloc(&d_f1, img_size) != cudaSuccess) return -2;
    cudaMalloc(&d_f2, img_size);
    cudaMalloc(&d_u, img_size);
    cudaMalloc(&d_v, img_size);

    cudaMemcpy(d_f1, frame1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, frame2->data, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_u, 0, img_size);
    cudaMemset(d_v, 0, img_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    /* 2. Lanzamiento de un único kernel fusionado */
    solve_flow_opt_kernel<<<gridSize, blockSize>>>(d_f1, d_f2, d_u, d_v, w, h);
    cudaDeviceSynchronize();

    cudaMemcpy(flow->u, d_u, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(flow->v, d_v, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_f1); cudaFree(d_f2);
    cudaFree(d_u); cudaFree(d_v);

    return 0;
}