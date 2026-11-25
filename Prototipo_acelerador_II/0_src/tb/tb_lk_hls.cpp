/*
 * tb_lk_hls.cpp  --  Testbench C-sim / Co-sim para lk_hls_v0 y lk_hls_v1
 *
 *  Referencia dorada: copia literal del algoritmo de src/lk_scalar.c
 *  Frames sinteticos (mismo enfoque que tests/test_cuda_correctness.c):
 *  patron senoidal desplazado dx = 1.0, dy = 0.5.
 *
 *  Uso:  csim  -> retorna 0 si PASS, 1 si FAIL
 *        -DLK_TB_V0 -> prueba v0
 *        -DLK_TB_V1 -> prueba v1 (gradientes en DDR)
 *        (nada)     -> prueba v2 (streaming)
 */
#include <cstdio>
#include <cmath>
#include <vector>
#include "lk_hls.h"

#define TB_W   128          /* resolucion reducida para co-sim rapida */
#define TB_H    96
#define TB_TOL 1e-4f

/* Co-sim: el wrapper copia 'depth' elementos de cada puntero m_axi
 * (depth=921600 = 1280x720 en los pragmas). Los buffers deben tener al
 * menos ese tamano o la co-sim truena con SIGSEGV. */
#define TB_ALLOC (1280 * 720)

static void lk_ref(const float *f1, const float *f2, float *u, float *v, int w, int h)
{
    std::vector<float> Ix(w*h, 0.f), Iy(w*h, 0.f), It(w*h, 0.f);
    for (int y = 1; y < h-1; y++)
        for (int x = 1; x < w-1; x++) {
            int i = y*w + x;
            Ix[i] = (f1[i+1] - f1[i-1]) * 0.5f;
            Iy[i] = (f1[i+w] - f1[i-w]) * 0.5f;
            It[i] = f2[i] - f1[i];
        }
    for (int i = 0; i < w*h; i++) { u[i] = 0.f; v[i] = 0.f; }
    for (int y = LK_MARGIN; y < h-LK_MARGIN; y++)
        for (int x = LK_MARGIN; x < w-LK_MARGIN; x++) {
            float sxx=0, syy=0, sxy=0, sxt=0, syt=0;
            for (int wy = -LK_RADIUS; wy <= LK_RADIUS; wy++)
                for (int wx = -LK_RADIUS; wx <= LK_RADIUS; wx++) {
                    int i = (y+wy)*w + x + wx;
                    sxx += Ix[i]*Ix[i]; syy += Iy[i]*Iy[i]; sxy += Ix[i]*Iy[i];
                    sxt += Ix[i]*It[i]; syt += Iy[i]*It[i];
                }
            float det = sxx*syy - sxy*sxy;
            if (det > LK_DET_THRESHOLD) {
                u[y*w+x] = (-syy*sxt + sxy*syt) / det;
                v[y*w+x] = ( sxy*sxt - sxx*syt) / det;
            }
        }
}

int main()
{
    const int w = TB_W, h = TB_H, n = w*h;
    std::vector<float> f1(TB_ALLOC, 0.f), f2(TB_ALLOC, 0.f);
    std::vector<float> u(TB_ALLOC, 0.f),  v(TB_ALLOC, 0.f);
    std::vector<float> ur(n), vr(n);

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            f1[y*w+x] = 0.5f + 0.25f*sinf(x*0.3f)       *cosf(y*0.2f);
            f2[y*w+x] = 0.5f + 0.25f*sinf((x-1.0f)*0.3f)*cosf((y-0.5f)*0.2f);
        }

    lk_ref(f1.data(), f2.data(), ur.data(), vr.data(), w, h);
#if defined(LK_TB_V0)
    const char *name = "lk_hls_v0";
    lk_hls_v0(f1.data(), f2.data(), u.data(), v.data(), w, h);
#elif defined(LK_TB_V1)
    const char *name = "lk_hls_v1";
    std::vector<float> gix(TB_ALLOC, 0.f), giy(TB_ALLOC, 0.f), git(TB_ALLOC, 0.f);   /* "DDR" */
    lk_hls_v1(f1.data(), f2.data(), gix.data(), giy.data(), git.data(),
              u.data(), v.data(), w, h);
#else
    const char *name = "lk_hls_v2";
    lk_hls_v2(f1.data(), f2.data(), u.data(), v.data(), w, h);
#endif

    float max_err = 0.f; double mean_err = 0.0; int bad = 0;
    for (int i = 0; i < n; i++) {
        float e = fmaxf(fabsf(u[i]-ur[i]), fabsf(v[i]-vr[i]));
        if (e > max_err) max_err = e;
        mean_err += e;
        if (e > TB_TOL) bad++;
    }
    mean_err /= n;

    printf("Frames sinteticos %dx%d (dx=1.0, dy=0.5)\n", w, h);
    printf("[%s] %-10s max_err=%.3e  mean_err=%.3e  pix_fuera_tol=%d (%.4f %%)\n",
           bad ? "FAIL" : "PASS", name, max_err, mean_err, bad, 100.0*bad/n);
    return bad ? 1 : 0;
}
