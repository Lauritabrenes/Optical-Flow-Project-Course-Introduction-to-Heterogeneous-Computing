#ifndef LK_HLS_H
#define LK_HLS_H

/*
 * lk_hls.h - Etapa 5: Prototipo en FPGA (Kria KV260, Vitis HLS)
 *
 * Parametros compartidos por todas las versiones incrementales del kernel.
 * Deben coincidir con src/lk_scalar.c para que la comparacion sea valida.
 */

#define LK_RADIUS        3          /* ventana 7 x 7 */
#define LK_WIN           (2 * LK_RADIUS + 1)
#define LK_MARGIN        (LK_RADIUS + 1)
#define LK_DET_THRESHOLD 1e-6f

/* Resolucion maxima soportada (decisions.md: 720p de trabajo).
 * Se puede sobreescribir desde CFLAGS (p. ej. v0: -DLK_MAX_W=320 -DLK_MAX_H=240) */
#ifndef LK_MAX_W
#define LK_MAX_W 1280
#endif
#ifndef LK_MAX_H
#define LK_MAX_H 720
#endif

/* v0: port directo, buffers internos en BRAM/URAM */
void lk_hls_v0(const float *f1, const float *f2,
               float *u, float *v, int w, int h);

/* v1: port directo con gradientes en DDR (ix, iy, it los reserva el host) */
void lk_hls_v1(const float *f1, const float *f2,
               float *ix, float *iy, float *it,
               float *u, float *v, int w, int h);

/* v2: streaming con line buffers, II = 1 */
void lk_hls_v2(const float *f1, const float *f2,
               float *u, float *v, int w, int h);

#endif
