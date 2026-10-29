/*
 * test_cuda_correctness.c - Verifica que las versiones GPU producen el
 * mismo flujo que la referencia escalar (of_lk_scalar).
 *
 * Criterio: un pixel "difiere" si |u_gpu-u_ref| o |v_gpu-v_ref| supera
 * ABS_TOL + REL_TOL*|ref|. Se aceptan diferencias en a lo sumo
 * MAX_BAD_FRAC de los pixeles (FMA / orden de redondeo en GPU; pixeles con
 * det cercano al umbral pueden cambiar de decision).
 *
 * Uso: ./test_cuda_correctness [frame1 frame2]
 *   sin argumentos usa imagenes sinteticas desplazadas (dx=1, dy=0.5)
 * Codigo de salida: 0 = PASS, 1 = algun FAIL, 2 = error de preparacion
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "of.h"

#define ABS_TOL      1e-3f
#define REL_TOL      1e-3f
#define MAX_BAD_FRAC 0.001   /* 0.1 % */

typedef int (*lk_fn)(const OFImage *, const OFImage *, OFFlow *);

static float pattern(float x, float y)
{
    return 0.5f + 0.25f * sinf(0.15f * x) * cosf(0.11f * y)
                + 0.15f * sinf(0.05f * (x + y));
}

static int make_synthetic(OFImage *a, OFImage *b, int w, int h)
{
    a->width = b->width = w; a->height = b->height = h;
    a->data = malloc((size_t)w * h * sizeof(float));
    b->data = malloc((size_t)w * h * sizeof(float));
    if (!a->data || !b->data) return -1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            a->data[y * w + x] = pattern((float)x, (float)y);
            b->data[y * w + x] = pattern(x - 1.0f, y - 0.5f);
        }
    return 0;
}

static int alloc_flow(OFFlow *f, int w, int h)
{
    f->width = w; f->height = h;
    f->u = calloc((size_t)w * h, sizeof(float));
    f->v = calloc((size_t)w * h, sizeof(float));
    return (f->u && f->v) ? 0 : -1;
}

static int compare(const char *name, lk_fn fn, const OFImage *a,
                   const OFImage *b, const OFFlow *ref)
{
    OFFlow g = {0};
    if (alloc_flow(&g, a->width, a->height) != 0) return 1;

    int rc = fn(a, b, &g);
    if (rc != 0) {
        printf("[FAIL] %-12s retorno %d\n", name, rc);
        free(g.u); free(g.v); return 1;
    }

    size_t n = (size_t)a->width * a->height, bad = 0;
    double max_err = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        float du = fabsf(g.u[i] - ref->u[i]);
        float dv = fabsf(g.v[i] - ref->v[i]);
        float tu = ABS_TOL + REL_TOL * fabsf(ref->u[i]);
        float tv = ABS_TOL + REL_TOL * fabsf(ref->v[i]);
        float e  = du > dv ? du : dv;
        if (du > tu || dv > tv) bad++;
        if (e > max_err) max_err = e;
        sum_err += e;
    }
    double frac = (double)bad / (double)n;
    int ok = frac <= MAX_BAD_FRAC;
    printf("[%s] %-12s max_err=%.3e  mean_err=%.3e  pix_fuera_tol=%zu (%.4f %%)\n",
           ok ? "PASS" : "FAIL", name, max_err, sum_err / n, bad, 100.0 * frac);
    free(g.u); free(g.v);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    OFImage r1 = {0}, r2 = {0}, a = {0}, b = {0};
    OFFlow ref = {0};
    int fails = 0;

    if (argc >= 3) {
        if (of_load_frame(argv[1], &r1) || of_load_frame(argv[2], &r2)) {
            fprintf(stderr, "No se pudieron cargar los frames\n"); return 2;
        }
        a = r1; b = r2;
        a.data = malloc((size_t)r1.width * r1.height * sizeof(float));
        b.data = malloc((size_t)r2.width * r2.height * sizeof(float));
        if (!a.data || !b.data) return 2;
        if (of_preprocess(&r1, &a) || of_preprocess(&r2, &b)) return 2;
        printf("Frames reales %dx%d\n", a.width, a.height);
    } else {
        if (make_synthetic(&a, &b, 640, 480)) return 2;
        printf("Frames sinteticos 640x480 (dx=1.0, dy=0.5)\n");
    }

    if (alloc_flow(&ref, a.width, a.height) || of_lk_scalar(&a, &b, &ref)) {
        fprintf(stderr, "Fallo la referencia escalar\n"); return 2;
    }

    fails += compare("cuda",        of_lk_cuda,        &a, &b, &ref);
    fails += compare("cuda_opt",    of_lk_cuda_opt,    &a, &b, &ref);
    fails += compare("cuda_shared", of_lk_cuda_shared, &a, &b, &ref);
#ifdef HAVE_NEON
    fails += compare("neon",        of_lk_neon,        &a, &b, &ref);
#endif

    if (fails) printf("\nRESULTADO: %d prueba(s) fallaron\n", fails);
    else       printf("\nRESULTADO: todas las pruebas pasaron\n");
    free(r1.data); free(r2.data); free(a.data); free(b.data);
    free(ref.u); free(ref.v);
    return fails ? 1 : 0;
}
