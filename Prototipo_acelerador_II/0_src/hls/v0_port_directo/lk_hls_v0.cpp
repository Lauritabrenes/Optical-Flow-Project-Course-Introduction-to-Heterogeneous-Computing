/*
 * lk_hls_v0.cpp - Etapa 5 (FPGA) / Version 0: port directo de lk_scalar.c
 *
 * Objetivo de esta version:
 *   - Tener una primera sintesis funcional en Vitis HLS (baseline FPGA).
 *   - Mantener el MISMO algoritmo que lk_scalar.c (gradientes centrales,
 *     ventana 7x7, regla de Cramer) para validar contra la CPU.
 *   - No se aplica ninguna optimizacion de microarquitectura: se espera un
 *     II alto en el lazo de la ventana. Sirve para cuantificar la mejora
 *     de v1 en el articulo.
 *
 * Interfaces:
 *   - f1, f2, u, v : m_axi (DDR del KV260), un bundle por puerto.
 *   - w, h y return: s_axilite (control desde el PS con XRT/PYNQ).
 *
 * Convenciones (iguales a of.h / lk_scalar.c):
 *   - Pixeles fuera de [LK_MARGIN, w-LK_MARGIN) x [LK_MARGIN, h-LK_MARGIN)
 *     quedan en 0.
 */

#include "lk_hls.h"

/* Buffers locales: en v0 se copia el frame completo al chip.
 * 1280*720 floats = 3.5 MB por buffer -> NO cabe en BRAM del KV260,
 * por eso se fuerza URAM y se documenta como limitacion de v0
 * (para pruebas de C-sim / cosim se usa una resolucion menor). */
static float Ix[LK_MAX_W * LK_MAX_H];
static float Iy[LK_MAX_W * LK_MAX_H];
static float It[LK_MAX_W * LK_MAX_H];

/* Calcula Ix, Iy, It igual que compute_gradients() de lk_scalar.c */
static void compute_gradients(const float *f1, const float *f2, int w, int h)
{
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1278
            const int i = y * w + x;
            Ix[i] = (f1[i + 1] - f1[i - 1]) * 0.5f;
            Iy[i] = (f1[i + w] - f1[i - w]) * 0.5f;
            It[i] = f2[i] - f1[i];
        }
    }
}

/* Acumula la ventana 7x7 y resuelve el sistema 2x2 */
static void solve_flow(float *u, float *v, int w, int h)
{
    for (int y = LK_MARGIN; y < h - LK_MARGIN; y++) {
        for (int x = LK_MARGIN; x < w - LK_MARGIN; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1272
            float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
            float sxt = 0.0f, syt = 0.0f;

            for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++) {
                for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                    const int i = (y + wy) * w + (x + wx);
                    const float gx = Ix[i];
                    const float gy = Iy[i];
                    const float gt = It[i];

                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                    sxt += gx * gt;
                    syt += gy * gt;
                }
            }

            const float det = sxx * syy - sxy * sxy;
            const int o = y * w + x;

            if (det > LK_DET_THRESHOLD) {
                u[o] = (-syy * sxt + sxy * syt) / det;
                v[o] = ( sxy * sxt - sxx * syt) / det;
            }
        }
    }
}

/* Inicializa la salida en cero (equivalente al memset de lk_scalar.c) */
static void clear_flow(float *u, float *v, int n)
{
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=921600
        u[i] = 0.0f;
        v[i] = 0.0f;
    }
}

void lk_hls_v0(const float *f1, const float *f2,
               float *u, float *v, int w, int h)
{
#pragma HLS INTERFACE m_axi port=f1 offset=slave bundle=gmem0 depth=921600
#pragma HLS INTERFACE m_axi port=f2 offset=slave bundle=gmem1 depth=921600
#pragma HLS INTERFACE m_axi port=u  offset=slave bundle=gmem2 depth=921600
#pragma HLS INTERFACE m_axi port=v  offset=slave bundle=gmem3 depth=921600
#pragma HLS INTERFACE s_axilite port=w
#pragma HLS INTERFACE s_axilite port=h
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS BIND_STORAGE variable=Ix type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=Iy type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=It type=ram_2p impl=uram

    if (w <= 2 * LK_MARGIN || h <= 2 * LK_MARGIN) return;
    if (w > LK_MAX_W || h > LK_MAX_H) return;

    clear_flow(u, v, w * h);
    compute_gradients(f1, f2, w, h);
    solve_flow(u, v, w, h);
}
