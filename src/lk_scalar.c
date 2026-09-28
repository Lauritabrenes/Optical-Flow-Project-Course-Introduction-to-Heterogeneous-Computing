/*
 * lk_scalar.c - Etapa 3: Lucas-Kanade escalar (baseline, sin NEON)
 *
 * Entrada : dos frames OFImage en escala de grises, float, ya preprocesados
 *           (rango [0,1] y suavizados por of_preprocess).
 * Salida  : campo de flujo (u, v) por pixel en OFFlow.
 *
 * Convenciones (las mismas que preprocess.c):
 *   - El llamador reserva flow->u y flow->v (width * height floats) y fija
 *     flow->width y flow->height. of_lk_scalar no los reserva ni los libera.
 *   - Retorno: 0 = OK, -1 = argumentos invalidos, -2 = fallo de memoria.
 *   - Este archivo no lee imagenes, no normaliza, no suaviza y no escribe
 *     archivos .flo ni calcula EPE: eso pertenece a otras etapas.
 *
 * Algoritmo: para cada pixel se acumulan, en una ventana de
 * (2*LK_RADIUS+1)^2, las sumas de Ix^2, Iy^2, IxIy, IxIt e IyIt, y se
 * resuelve el sistema 2x2 (A^T A)[u v]^T = -A^T b por regla de Cramer.
 */

#include <stdlib.h>
#include <string.h>
#include "of.h"

#ifndef LK_RADIUS
#define LK_RADIUS 3              /* ventana de 7 x 7 */
#endif

#ifndef LK_DET_THRESHOLD
#define LK_DET_THRESHOLD 1e-6f   /* por debajo, el sistema se considera singular */
#endif

/* Los gradientes usan vecinos a +-1 pixel, asi que la ventana necesita
 * un pixel adicional de margen respecto al borde de la imagen. */
#define LK_MARGIN (LK_RADIUS + 1)

/* Ix, Iy: diferencias centrales sobre frame1. It: diferencia temporal.
 * Solo se calculan en el interior (1..w-2, 1..h-2); el resto queda en cero
 * y nunca se lee porque la ventana respeta LK_MARGIN. */
static void compute_gradients(const OFImage *f1, const OFImage *f2,
                              float *Ix, float *Iy, float *It)
{
    const int w = f1->width;
    const int h = f1->height;

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            const int i = y * w + x;
            Ix[i] = (f1->data[i + 1] - f1->data[i - 1]) * 0.5f;
            Iy[i] = (f1->data[i + w] - f1->data[i - w]) * 0.5f;
            It[i] = f2->data[i] - f1->data[i];
        }
    }
}

/* Acumula la ventana de cada pixel y resuelve el sistema 2x2. */
static void solve_flow(const float *Ix, const float *Iy, const float *It,
                       OFFlow *flow)
{
    const int w = flow->width;
    const int h = flow->height;

    for (int y = LK_MARGIN; y < h - LK_MARGIN; y++) {
        for (int x = LK_MARGIN; x < w - LK_MARGIN; x++) {
            float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
            float sxt = 0.0f, syt = 0.0f;

            for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++) {
                const int row = (y + wy) * w + x;
                for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                    const int i = row + wx;
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
                flow->u[o] = (-syy * sxt + sxy * syt) / det;
                flow->v[o] = ( sxy * sxt - sxx * syt) / det;
            }
            /* si no, u y v ya estan en cero (memset inicial) */
        }
    }
}

int of_lk_scalar(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
    if (!frame1 || !frame2 || !flow) return -1;
    if (!frame1->data || !frame2->data || !flow->u || !flow->v) return -1;

    const int w = frame1->width;
    const int h = frame1->height;

    if (frame2->width != w || frame2->height != h) return -1;
    if (flow->width != w || flow->height != h) return -1;
    if (w <= 2 * LK_MARGIN || h <= 2 * LK_MARGIN) return -1;

    const size_t n = (size_t)w * (size_t)h;

    /* Todo el campo parte en cero: los bordes y los pixeles sin
     * solucion quedan con flujo nulo. */
    memset(flow->u, 0, n * sizeof(float));
    memset(flow->v, 0, n * sizeof(float));

    /* Un solo bloque para Ix, Iy e It. */
    float *grad = (float *)calloc(3 * n, sizeof(float));
    if (!grad) return -2;

    float *Ix = grad;
    float *Iy = grad + n;
    float *It = grad + 2 * n;

    compute_gradients(frame1, frame2, Ix, Iy, It);
    solve_flow(Ix, Iy, It, flow);

    free(grad);
    return 0;
}