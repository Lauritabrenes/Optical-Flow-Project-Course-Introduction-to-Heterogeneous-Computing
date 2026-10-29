/*
 * lk_neon.c - Etapa 3: Lucas-Kanade acelerado con ARM NEON
 * 
 * Procesa 4 pixeles concurrentemente usando registros float32x4_t de 128 bits.
 */

#include "of.h"
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(HAVE_NEON)
#include <arm_neon.h>

#ifndef LK_RADIUS
#define LK_RADIUS 3
#endif

#ifndef LK_DET_THRESHOLD
#define LK_DET_THRESHOLD 1e-6f
#endif

#define LK_MARGIN (LK_RADIUS + 1)

static void compute_gradients_neon(const OFImage *f1, const OFImage *f2,
                                   float *Ix, float *Iy, float *It)
{
    const int w = f1->width;
    const int h = f1->height;
    
    /* Vector constante de 0.5 para los gradientes espaciales */
    float32x4_t half = vdupq_n_f32(0.5f);

    for (int y = 1; y < h - 1; y++) {
        int x = 1;
        /* Procesar de 4 en 4 */
        for (; x <= w - 1 - 4; x += 4) {
            const int i = y * w + x;
            
            /* Ix = (f1[i+1] - f1[i-1]) * 0.5 */
            float32x4_t f1_next = vld1q_f32(&f1->data[i + 1]);
            float32x4_t f1_prev = vld1q_f32(&f1->data[i - 1]);
            float32x4_t ix_vec = vmulq_f32(vsubq_f32(f1_next, f1_prev), half);
            vst1q_f32(&Ix[i], ix_vec);

            /* Iy = (f1[i+w] - f1[i-w]) * 0.5 */
            float32x4_t f1_down = vld1q_f32(&f1->data[i + w]);
            float32x4_t f1_up   = vld1q_f32(&f1->data[i - w]);
            float32x4_t iy_vec = vmulq_f32(vsubq_f32(f1_down, f1_up), half);
            vst1q_f32(&Iy[i], iy_vec);

            /* It = f2[i] - f1[i] */
            float32x4_t f2_vec = vld1q_f32(&f2->data[i]);
            float32x4_t f1_vec = vld1q_f32(&f1->data[i]);
            float32x4_t it_vec = vsubq_f32(f2_vec, f1_vec);
            vst1q_f32(&It[i], it_vec);
        }
        /* Pixeles residuales en los bordes que no caben en un bloque de 4 */
        for (; x < w - 1; x++) {
            const int i = y * w + x;
            Ix[i] = (f1->data[i + 1] - f1->data[i - 1]) * 0.5f;
            Iy[i] = (f1->data[i + w] - f1->data[i - w]) * 0.5f;
            It[i] = f2->data[i] - f1->data[i];
        }
    }
}

static void solve_flow_neon(const float *Ix, const float *Iy, const float *It,
                            OFFlow *flow)
{
    const int w = flow->width;
    const int h = flow->height;

    for (int y = LK_MARGIN; y < h - LK_MARGIN; y++) {
        int x = LK_MARGIN;
        
        /* Procesar de 4 en 4 */
        for (; x <= w - LK_MARGIN - 4; x += 4) {
            float32x4_t sxx = vdupq_n_f32(0.0f);
            float32x4_t syy = vdupq_n_f32(0.0f);
            float32x4_t sxy = vdupq_n_f32(0.0f);
            float32x4_t sxt = vdupq_n_f32(0.0f);
            float32x4_t syt = vdupq_n_f32(0.0f);

            for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++) {
                const int row = (y + wy) * w + x;
                for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                    const int i = row + wx;
                    
                    /* Cargar gradientes de 4 pixeles simultaneos */
                    float32x4_t gx = vld1q_f32(&Ix[i]);
                    float32x4_t gy = vld1q_f32(&Iy[i]);
                    float32x4_t gt = vld1q_f32(&It[i]);

                    /* Acumulacion de tensores (Fused Multiply-Add) */
                    sxx = vmlaq_f32(sxx, gx, gx);
                    syy = vmlaq_f32(syy, gy, gy);
                    sxy = vmlaq_f32(sxy, gx, gy);
                    sxt = vmlaq_f32(sxt, gx, gt);
                    syt = vmlaq_f32(syt, gy, gt);
                }
            }

            /* det = sxx * syy - sxy * sxy */
            float32x4_t det = vsubq_f32(vmulq_f32(sxx, syy), vmulq_f32(sxy, sxy));

            /* u_num = -syy * sxt + sxy * syt */
            float32x4_t u_num = vaddq_f32(vmulq_f32(vnegq_f32(syy), sxt), vmulq_f32(sxy, syt));
            /* v_num = sxy * sxt - sxx * syt */
            float32x4_t v_num = vsubq_f32(vmulq_f32(sxy, sxt), vmulq_f32(sxx, syt));

            /* 
             * Para maxima portabilidad ARM y evitar dependencias de hardware en la 
             * division vectorial (vdivq_f32 no siempre esta disp.), resolvemos 
             * la division y el threshold de manera escalar en los 4 resultados
             */
            float det_arr[4], u_num_arr[4], v_num_arr[4];
            vst1q_f32(det_arr, det);
            vst1q_f32(u_num_arr, u_num);
            vst1q_f32(v_num_arr, v_num);

            const int o = y * w + x;
            for (int k = 0; k < 4; k++) {
                if (det_arr[k] > LK_DET_THRESHOLD) {
                    flow->u[o + k] = u_num_arr[k] / det_arr[k];
                    flow->v[o + k] = v_num_arr[k] / det_arr[k];
                }
            }
        }
        
        /* Pixeles residuales en los bordes horizontales de la ventana */
        for (; x < w - LK_MARGIN; x++) {
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
        }
    }
}

int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
    if (!frame1 || !frame2 || !flow) return -1;
    if (!frame1->data || !frame2->data || !flow->u || !flow->v) return -1;

    const int w = frame1->width;
    const int h = frame1->height;

    if (frame2->width != w || frame2->height != h) return -1;
    if (flow->width != w || flow->height != h) return -1;
    if (w <= 2 * LK_MARGIN || h <= 2 * LK_MARGIN) return -1;

    const size_t n = (size_t)w * (size_t)h;

    memset(flow->u, 0, n * sizeof(float));
    memset(flow->v, 0, n * sizeof(float));

    float *grad = (float *)calloc(3 * n, sizeof(float));
    if (!grad) return -2;

    float *Ix = grad;
    float *Iy = grad + n;
    float *It = grad + 2 * n;

    compute_gradients_neon(frame1, frame2, Ix, Iy, It);
    solve_flow_neon(Ix, Iy, It, flow);

    free(grad);
    return 0;
}

#else
/* Fallback a escalar si el compilador no detecta ARM NEON */
extern int of_lk_scalar(const OFImage *frame1, const OFImage *frame2, OFFlow *flow);
int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow) {
    return of_lk_scalar(frame1, frame2, flow);
}
#endif