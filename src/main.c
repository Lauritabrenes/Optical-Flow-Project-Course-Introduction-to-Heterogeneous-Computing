/*
 * main.c - Punto de entrada del prototipo (bench_main)
 *
 * Pipeline: carga (io.c) -> preprocesamiento (preprocess.c)
 *           -> Lucas-Kanade escalar (lk_scalar.c) -> resultados (metrics.c)
 *
 * Uso: ./build/bench_main <frame1> <frame2> [salida]
 *
 * Responsabilidad de memoria (contrato del equipo):
 *   - of_load_frame reserva raw.data; el llamador la libera con free().
 *   - of_preprocess y of_lk_scalar NO reservan los buffers de salida;
 *     el llamador (aqui) los reserva y los libera.
 */

#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "of.h"

/* Resolucion de trabajo acordada por el equipo. Si los frames difieren
 * solo se advierte; no se detiene la ejecucion. */
#define EXPECTED_WIDTH  1280
#define EXPECTED_HEIGHT 720

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Reserva un OFImage con las mismas dimensiones que 'src'. */
static int alloc_like(OFImage *dst, const OFImage *src)
{
    dst->width  = src->width;
    dst->height = src->height;
    dst->data   = (float *)malloc((size_t)src->width * (size_t)src->height
                                  * sizeof(float));
    return dst->data ? 0 : -1;
}

/* Resumen rapido del flujo para comprobar que el resultado es razonable. */
static void print_flow_stats(const OFFlow *f)
{
    const size_t n = (size_t)f->width * (size_t)f->height;
    size_t nonzero = 0;
    double sum_u = 0.0, sum_v = 0.0, max_sq = 0.0;

    for (size_t i = 0; i < n; i++) {
        const double u = f->u[i];
        const double v = f->v[i];
        sum_u += u;
        sum_v += v;
        if (u != 0.0 || v != 0.0) nonzero++;
        if (u * u + v * v > max_sq) max_sq = u * u + v * v;
    }

    printf("Flujo: pixeles con solucion = %zu de %zu (%.1f %%)\n",
           nonzero, n, 100.0 * (double)nonzero / (double)n);
    printf("Flujo: promedio (u, v) = (%.4f, %.4f), magnitud maxima = %.4f px\n",
           sum_u / (double)n, sum_v / (double)n, sqrt(max_sq));
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <frame1> <frame2> [salida]\n", argv[0]);
        return 1;
    }

    const char *path1    = argv[1];
    const char *path2    = argv[2];
    const char *out_path = (argc > 3) ? argv[3] : "results/output";

    OFImage raw1 = {0}, raw2 = {0};   /* etapa 1: float [0,255]          */
    OFImage pp1  = {0}, pp2  = {0};   /* etapa 2: float [0,1], suavizado */
    OFFlow  flow = {0};               /* etapa 3: campo (u, v)           */
    int status = 1;
    int rc;
    double t0, t_load, t_pre, t_lk;

    /* Etapa 1: inicializacion y carga */
    t0 = now_ms();
    if (of_load_frame(path1, &raw1) != 0) {
        fprintf(stderr, "Error: no se pudo cargar '%s'\n", path1);
        goto cleanup;
    }
    if (of_load_frame(path2, &raw2) != 0) {
        fprintf(stderr, "Error: no se pudo cargar '%s'\n", path2);
        goto cleanup;
    }
    t_load = now_ms() - t0;

    if (raw1.width != raw2.width || raw1.height != raw2.height) {
        fprintf(stderr, "Error: los frames tienen dimensiones distintas "
                        "(%dx%d vs %dx%d)\n",
                raw1.width, raw1.height, raw2.width, raw2.height);
        goto cleanup;
    }
    if (raw1.width != EXPECTED_WIDTH || raw1.height != EXPECTED_HEIGHT) {
        fprintf(stderr, "Aviso: resolucion %dx%d (se esperaba %dx%d)\n",
                raw1.width, raw1.height, EXPECTED_WIDTH, EXPECTED_HEIGHT);
    }

    /* Buffers de las etapas siguientes (el llamador los reserva) */
    if (alloc_like(&pp1, &raw1) != 0 || alloc_like(&pp2, &raw2) != 0) {
        fprintf(stderr, "Error: sin memoria para los buffers preprocesados\n");
        goto cleanup;
    }
    flow.width  = raw1.width;
    flow.height = raw1.height;
    flow.u = (float *)malloc((size_t)flow.width * (size_t)flow.height * sizeof(float));
    flow.v = (float *)malloc((size_t)flow.width * (size_t)flow.height * sizeof(float));
    if (!flow.u || !flow.v) {
        fprintf(stderr, "Error: sin memoria para el campo de flujo\n");
        goto cleanup;
    }

    /* Etapa 2: preprocesamiento (ambos frames) */
    t0 = now_ms();
    if ((rc = of_preprocess(&raw1, &pp1)) != 0 ||
        (rc = of_preprocess(&raw2, &pp2)) != 0) {
        fprintf(stderr, "Error en of_preprocess (codigo %d)\n", rc);
        goto cleanup;
    }
    t_pre = now_ms() - t0;

    /* Etapa 3: Lucas-Kanade escalar */
    t0 = now_ms();
    rc = of_lk_scalar(&pp1, &pp2, &flow);
    t_lk = now_ms() - t0;
    if (rc != 0) {
        fprintf(stderr, "Error en of_lk_scalar (codigo %d)\n", rc);
        goto cleanup;
    }

    /* Etapa 4: resultados y mediciones */
    printf("Frames: %dx%d\n", raw1.width, raw1.height);
    printf("Tiempo carga         : %8.2f ms (2 frames)\n", t_load);
    printf("Tiempo preprocesado  : %8.2f ms (2 frames)\n", t_pre);
    printf("Tiempo LK escalar    : %8.2f ms\n", t_lk);
    print_flow_stats(&flow);

    /* TODO (Persona 4): cuando metrics.c este implementado, calcular EPE
     * con of_compute_epe() contra el ground truth. */
    of_save_results(out_path, &flow, t_lk);

    status = 0;

cleanup:
    free(raw1.data);
    free(raw2.data);
    free(pp1.data);
    free(pp2.data);
    free(flow.u);
    free(flow.v);
    return status;
}