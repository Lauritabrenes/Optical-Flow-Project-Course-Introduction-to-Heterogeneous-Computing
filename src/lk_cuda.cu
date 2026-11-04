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
}

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

/* 
 * FUNCION HOST: Puente llamado desde C (main.c)
 */
extern "C" int of_lk_cuda(const OFImage *frame1, const OFImage *frame2, OFFlow *flow) 
{
    if (!frame1 || !frame2 || !flow) return -1;

    int w = frame1->width;
    int h = frame1->height;
    size_t img_size = w * h * sizeof(float);

    /* Punteros de Device (VRAM) */
    float *d_f1, *d_f2, *d_Ix, *d_Iy, *d_It, *d_u, *d_v;

    /* 1. Asignacion en GPU */
    if (cudaMalloc(&d_f1, img_size) != cudaSuccess) return -2;
    cudaMalloc(&d_f2, img_size);
    cudaMalloc(&d_Ix, img_size);
    cudaMalloc(&d_Iy, img_size);
    cudaMalloc(&d_It, img_size);
    cudaMalloc(&d_u, img_size);
    cudaMalloc(&d_v, img_size);

    /* 2. Transferencia RAM -> VRAM */
    cudaMemcpy(d_f1, frame1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, frame2->data, img_size, cudaMemcpyHostToDevice);
    cudaMemset(d_u, 0, img_size);
    cudaMemset(d_v, 0, img_size);
    cudaMemset(d_Ix, 0, img_size);
    cudaMemset(d_Iy, 0, img_size);
    cudaMemset(d_It, 0, img_size);

    /* 3. Configuracion de Geometria de Hilos (Bloques de 16x16) */
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    /* 4. Lanzamiento de Kernels */
    compute_gradients_kernel<<<gridSize, blockSize>>>(d_f1, d_f2, d_Ix, d_Iy, d_It, w, h);
    cudaDeviceSynchronize();

    solve_flow_kernel<<<gridSize, blockSize>>>(d_Ix, d_Iy, d_It, d_u, d_v, w, h);
    cudaDeviceSynchronize();

    /* 5. Transferencia VRAM -> RAM */
    cudaMemcpy(flow->u, d_u, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(flow->v, d_v, img_size, cudaMemcpyDeviceToHost);

    /* 6. Liberacion de VRAM */
    cudaFree(d_f1); cudaFree(d_f2);
    cudaFree(d_Ix); cudaFree(d_Iy); cudaFree(d_It);
    cudaFree(d_u); cudaFree(d_v);

    return 0;
}