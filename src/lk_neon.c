/*
 *  lk_neon.c — Stage 3 Vectorized Lucas-Kanade with ARM NEON
 */

#include "of.h"
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(HAVE_NEON)
#include <arm_neon.h>
#endif

int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(HAVE_NEON)
    const int w = frame1->width;
    const int h = frame1->height;
    const float *I1 = frame1->data;
    const float *I2 = frame2->data;

    const int win = 2; // Ventana de 5x5 alrededor de cada pixel

    for (int y = win; y < h - win; y++) {
        for (int x = win; x < w - win; x++) {
            
            // Acumuladores vectoriales inicializados en 0
            float32x4_t v_sum_Ix2  = vdupq_n_f32(0.0f);
            float32x4_t v_sum_Iy2  = vdupq_n_f32(0.0f);
            float32x4_t v_sum_IxIy = vdupq_n_f32(0.0f);
            float32x4_t v_sum_IxIt = vdupq_n_f32(0.0f);
            float32x4_t v_sum_IyIt = vdupq_n_f32(0.0f);

            for (int wy = -win; wy <= win; wy++) {
                int py = y + wy;
                int px_start = x - win; // Ventana horizontal de 5 elementos

                // Procesamos los primeros 4 pixeles en paralelo mediante NEON
                int idx = py * w + px_start;

                // Carga de pixeles adyacentes para derivadas Ix, Iy, It
                float32x4_t v_I1_right = vld1q_f32(&I1[idx + 1]);
                float32x4_t v_I1_left  = vld1q_f32(&I1[idx - 1]);
                float32x4_t v_I1_down  = vld1q_f32(&I1[idx + w]);
                float32x4_t v_I1_up    = vld1q_f32(&I1[idx - w]);
                float32x4_t v_I1_curr  = vld1q_f32(&I1[idx]);
                float32x4_t v_I2_curr  = vld1q_f32(&I2[idx]);

                // Gradientes: Ix = (I1_right - I1_left)/2, Iy = (I1_down - I1_up)/2, It = I2 - I1
                float32x4_t v_half = vdupq_n_f32(0.5f);
                float32x4_t v_Ix = vmulq_f32(vsubq_f32(v_I1_right, v_I1_left), v_half);
                float32x4_t v_Iy = vmulq_f32(vsubq_f32(v_I1_down, v_I1_up), v_half);
                float32x4_t v_It = vsubq_f32(v_I2_curr, v_I1_curr);

                // Acumulación de productos del Tensor de Estructura
                v_sum_Ix2  = vfmaq_f32(v_sum_Ix2,  v_Ix, v_Ix);
                v_sum_Iy2  = vfmaq_f32(v_sum_Iy2,  v_Iy, v_Iy);
                v_sum_IxIy = vfmaq_f32(v_sum_IxIy, v_Ix, v_Iy);
                v_sum_IxIt = vfmaq_f32(v_sum_IxIt, v_Ix, v_It);
                v_sum_IyIt = vfmaq_f32(v_sum_IyIt, v_Iy, v_It);

                // Procesamiento escalar para el 5to pixel restante del parche 5x5
                int idx5 = py * w + (x + 2);
                float Ix5 = (I1[idx5 + 1] - I1[idx5 - 1]) * 0.5f;
                float Iy5 = (I1[idx5 + w] - I1[idx5 - w]) * 0.5f;
                float It5 = I2[idx5] - I1[idx5];

                v_sum_Ix2  = vsetq_lane_f32(vgetq_lane_f32(v_sum_Ix2, 0)  + Ix5 * Ix5, v_sum_Ix2, 0);
                v_sum_Iy2  = vsetq_lane_f32(vgetq_lane_f32(v_sum_Iy2, 0)  + Iy5 * Iy5, v_sum_Iy2, 0);
                v_sum_IxIy = vsetq_lane_f32(vgetq_lane_f32(v_sum_IxIy, 0) + Ix5 * Iy5, v_sum_IxIy, 0);
                v_sum_IxIt = vsetq_lane_f32(vgetq_lane_f32(v_sum_IxIt, 0) + Ix5 * It5, v_sum_IxIt, 0);
                v_sum_IyIt = vsetq_lane_f32(vgetq_lane_f32(v_sum_IyIt, 0) + Iy5 * It5, v_sum_IyIt, 0);
            }

            // Reducción horizontal de los registros NEON a escalares
            float sum_Ix2  = vaddvq_f32(v_sum_Ix2);
            float sum_Iy2  = vaddvq_f32(v_sum_Iy2);
            float sum_IxIy = vaddvq_f32(v_sum_IxIy);
            float sum_IxIt = vaddvq_f32(v_sum_IxIt);
            float sum_IyIt = vaddvq_f32(v_sum_IyIt);

            // Resolución del sistema 2x2: [sum_Ix2 sum_IxIy; sum_IxIy sum_Iy2] * [u; v] = -[sum_IxIt; sum_IyIt]
            float det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy;
            int flow_idx = y * w + x;

            if (fabsf(det) > 1e-4f) {
                flow->u[flow_idx] = (-sum_Iy2 * sum_IxIt + sum_IxIy * sum_IyIt) / det;
                flow->v[flow_idx] = ( sum_IxIy * sum_IxIt - sum_Ix2 * sum_IyIt) / det;
            } else {
                flow->u[flow_idx] = 0.0f;
                flow->v[flow_idx] = 0.0f;
            }
        }
    }
    return 0;
#else
    /* Fallback en x86_64 o entornos sin soporte NEON */
    return of_lk_scalar(frame1, frame2, flow);
#endif
}