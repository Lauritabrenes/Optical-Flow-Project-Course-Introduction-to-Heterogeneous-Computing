/*
 * main.c - Punto de entrada del prototipo (bench_main)
 *
 * Pipeline: carga (io.c) -> preprocesamiento (preprocess.c)
 *           -> Lucas-Kanade (lk_scalar.c / lk_neon.c / lk_cuda) -> resultados (metrics.c)
 */

#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "of.h"
#ifdef HAVE_CUDA
#include "gpu_timing.h"
#endif

typedef enum { MODE_SCALAR, MODE_NEON, MODE_CUDA, MODE_CUDA_OPT, MODE_CUDA_SHARED} ComputeMode;

typedef struct {
    const char *path1;
    const char *path2;
    const char *out_path;
    int   has_gt;
    float gt_dx;
    float gt_dy;
    ComputeMode mode;
} Args;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Uso: %s <frame1> <frame2> [salida] [--gt-shift DX DY] [--mode scalar|neon|cuda|cuda_opt|cuda_shared]\n"
            "  salida            CSV de resultados (defecto: results/last_run.csv)\n"
            "  --gt-shift DX DY  ground truth uniforme en pixeles (para EPE)\n"
            "  --mode            Define el acelerador a usar (defecto: scalar)\n",
            prog);
}

static const char *default_out_path(void)
{
    struct stat st;
    if (stat("results", &st) == 0 && S_ISDIR(st.st_mode)) {
        return "results/last_run.csv";
    }
    return "../results/last_run.csv";
}

static int parse_args(int argc, char **argv, Args *a)
{
    int positional = 0;

    a->path1 = NULL;
    a->path2 = NULL;
    a->out_path = default_out_path();
    a->has_gt = 0;
    a->gt_dx = 0.0f;
    a->gt_dy = 0.0f;
    a->mode = MODE_SCALAR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gt-shift") == 0) {
            char *end_x, *end_y;
            if (i + 2 >= argc) return -1;
            a->gt_dx = strtof(argv[i + 1], &end_x);
            a->gt_dy = strtof(argv[i + 2], &end_y);
            if (end_x == argv[i + 1] || *end_x != '\0') return -1;
            if (end_y == argv[i + 2] || *end_y != '\0') return -1;
            a->has_gt = 1;
            i += 2;
        } else if (strcmp(argv[i], "--mode") == 0) {
            if (i + 1 >= argc) return -1;
            if (strcmp(argv[i+1], "neon") == 0) a->mode = MODE_NEON;
            else if (strcmp(argv[i+1], "cuda") == 0) a->mode = MODE_CUDA;
            else if (strcmp(argv[i+1], "cuda_opt") == 0) a->mode = MODE_CUDA_OPT;
            else if (strcmp(argv[i+1], "cuda_shared") == 0) a->mode = MODE_CUDA_SHARED;
            else a->mode = MODE_SCALAR;
            i++;
        } else if (positional == 0) {
            a->path1 = argv[i];
            positional++;
        } else if (positional == 1) {
            a->path2 = argv[i];
            positional++;
        } else if (positional == 2) {
            a->out_path = argv[i];
            positional++;
        } else {
            return -1;
        }
    }
    return (a->path1 && a->path2) ? 0 : -1;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int alloc_like(OFImage *dst, const OFImage *src)
{
    dst->width  = src->width;
    dst->height = src->height;
    dst->data   = (float *)malloc((size_t)src->width * (size_t)src->height * sizeof(float));
    return dst->data ? 0 : -1;
}

static int make_uniform_flow(OFFlow *f, int w, int h, float dx, float dy)
{
    const size_t n = (size_t)w * (size_t)h;
    f->width  = w;
    f->height = h;
    f->u = (float *)malloc(n * sizeof(float));
    f->v = (float *)malloc(n * sizeof(float));
    if (!f->u || !f->v) return -1;
    for (size_t i = 0; i < n; i++) {
        f->u[i] = dx;
        f->v[i] = dy;
    }
    return 0;
}

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
    Args args;
    if (parse_args(argc, argv, &args) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    OFImage raw1 = {0}, raw2 = {0};   
    OFImage pp1  = {0}, pp2  = {0};   
    OFFlow  flow = {0};               
    OFFlow  gt   = {0};               
    int status = 1;
    int rc;
    double t0;
    double t_load = 0.0, t_pre = 0.0, t_lk = 0.0, t_save = 0.0;

    /* [1] Inicializacion y carga */
    t0 = now_ms();
    if (of_load_frame(args.path1, &raw1) != 0) {
        fprintf(stderr, "Error: no se pudo cargar '%s'\n", args.path1);
        goto cleanup;
    }
    if (of_load_frame(args.path2, &raw2) != 0) {
        fprintf(stderr, "Error: no se pudo cargar '%s'\n", args.path2);
        goto cleanup;
    }
    t_load = now_ms() - t0;

    if (raw1.width != raw2.width || raw1.height != raw2.height) {
        fprintf(stderr, "Error: frames difieren en resolucion (%dx%d vs %dx%d)\n",
                raw1.width, raw1.height, raw2.width, raw2.height);
        goto cleanup;
    }

    /* Reserva de memoria intermedios */
    if (alloc_like(&pp1, &raw1) != 0 || alloc_like(&pp2, &raw2) != 0) {
        fprintf(stderr, "Error: sin memoria para preprocesamiento\n");
        goto cleanup;
    }
    flow.width  = raw1.width;
    flow.height = raw1.height;
    flow.u = (float *)malloc((size_t)flow.width * (size_t)flow.height * sizeof(float));
    flow.v = (float *)malloc((size_t)flow.width * (size_t)flow.height * sizeof(float));
    if (!flow.u || !flow.v) {
        fprintf(stderr, "Error: sin memoria para flujo\n");
        goto cleanup;
    }

    /* [2] Preprocesamiento */
    t0 = now_ms();
    if ((rc = of_preprocess(&raw1, &pp1)) != 0 || (rc = of_preprocess(&raw2, &pp2)) != 0) {
        fprintf(stderr, "Error en of_preprocess (%d)\n", rc);
        goto cleanup;
    }
    t_pre = now_ms() - t0;

    /* [2b] Warm-up GPU: crea el contexto CUDA FUERA de la region medida
     *      para que LK_COMPUTE no incluya la inicializacion del driver. */
#ifdef HAVE_CUDA
    if (args.mode == MODE_CUDA || args.mode == MODE_CUDA_OPT ||
        args.mode == MODE_CUDA_SHARED) {
        if (of_gpu_warmup() != 0) {
            fprintf(stderr, "Error: no se pudo inicializar CUDA\n");
            goto cleanup;
        }
    }
#endif

    /* [3] Cómputo LK dirigido por flags */
    t0 = now_ms();
    if (args.mode == MODE_NEON) {
        rc = of_lk_neon(&pp1, &pp2, &flow);
    } else if (args.mode == MODE_CUDA) {
#ifdef HAVE_CUDA
        rc = of_lk_cuda(&pp1, &pp2, &flow);
#else
        rc = of_lk_scalar(&pp1, &pp2, &flow);
#endif

    } else if (args.mode == MODE_CUDA_OPT) {
#ifdef HAVE_CUDA
        rc = of_lk_cuda_opt(&pp1, &pp2, &flow);
#else
        rc = of_lk_scalar(&pp1, &pp2, &flow);
#endif

    } else if (args.mode == MODE_CUDA_SHARED) {
#ifdef HAVE_CUDA
        rc = of_lk_cuda_shared(&pp1, &pp2, &flow);
#else
        rc = of_lk_scalar(&pp1, &pp2, &flow);
#endif

    } else {
        rc = of_lk_scalar(&pp1, &pp2, &flow);
    }
    t_lk = now_ms() - t0;

    if (rc != 0) {
        fprintf(stderr, "Error en calculo de optical flow (%d)\n", rc);
        goto cleanup;
    }

    /* Reporte y Métricas */
    printf("\nResolucion procesada: %dx%d\n", raw1.width, raw1.height);
    print_flow_stats(&flow);

    if (args.has_gt) {
        if (make_uniform_flow(&gt, flow.width, flow.height, args.gt_dx, args.gt_dy) == 0) {
            float epe = of_compute_epe(&flow, &gt);
            if (epe >= 0.0f) printf("EPE vs GT: %.4f px\n", epe);
        }
    }

    /* [4] Guardado (aislado) */
    t0 = now_ms();
    of_save_results(args.out_path, &flow, t_lk);
    t_save = now_ms() - t0;
    
    FILE *chk = fopen(args.out_path, "r");
    if (chk) {
        fclose(chk);
        printf("Guardado exitoso en %s\n\n", args.out_path);
    }

    /* Salida final estándar que captura profile.sh */
    printf("[1] IO          %10.3f ms\n", t_load);
    printf("[2] PREPROCESS  %10.3f ms\n", t_pre);
    printf("[3] LK_COMPUTE  %10.3f ms\n", t_lk);
    printf("[4] SAVE        %10.3f ms\n", t_save);
    printf("TOTAL           %10.3f ms\n", t_load + t_pre + t_lk + t_save);

    /* Desglose GPU (no coincide con el regex de profile.sh: ^\[[1-4]\]) */
#ifdef HAVE_CUDA
    if (args.mode == MODE_CUDA || args.mode == MODE_CUDA_OPT ||
        args.mode == MODE_CUDA_SHARED) {
        printf("[3a] GPU_ALLOC  %10.3f ms\n", g_gpu_timing.alloc_ms);
        printf("[3b] GPU_H2D    %10.3f ms\n", g_gpu_timing.h2d_ms);
        printf("[3c] GPU_KERNEL %10.3f ms\n", g_gpu_timing.kernel_ms);
        printf("[3d] GPU_D2H    %10.3f ms\n", g_gpu_timing.d2h_ms);
        printf("[3e] GPU_FREE   %10.3f ms\n", g_gpu_timing.free_ms);
    }
#endif

    status = 0;

cleanup:
    free(raw1.data); free(raw2.data);
    free(pp1.data);  free(pp2.data);
    free(flow.u);    free(flow.v);
    free(gt.u);      free(gt.v);
    return status;
}