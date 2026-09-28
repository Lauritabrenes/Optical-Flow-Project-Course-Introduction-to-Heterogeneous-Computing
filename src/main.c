/*
 * main.c - Punto de entrada del prototipo (bench_main)
 *
 * Pipeline: carga (io.c) -> preprocesamiento (preprocess.c)
 *           -> Lucas-Kanade escalar (lk_scalar.c) -> resultados (metrics.c)
 *
 * Uso:
 *   ./bench_main <frame1> <frame2> [salida] [--gt-shift DX DY]
 *
 *   salida            CSV de resultados. Por defecto results/last_run.csv si
 *                     existe la carpeta results/ (ejecucion desde la raiz);
 *                     si no, ../results/last_run.csv (ejecucion desde build/
 *                     o scripts/).
 *   --gt-shift DX DY  Ground truth uniforme: el frame 2 es el frame 1
 *                     desplazado (DX, DY) pixeles. Pensado para imagenes
 *                     sinteticas; con esto se calcula el EPE.
 *
 * Formato de la salida que consume scripts/profile.sh. Estas cinco lineas
 * deben EMPEZAR con el marcador, en este orden, y terminar con "<valor> ms".
 * No cambiar los nombres sin actualizar profile.sh:
 *   [1] IO          tiempo de carga de los 2 frames
 *   [2] PREPROCESS  tiempo de preprocesamiento de los 2 frames
 *   [3] LK_SCALAR   tiempo de of_lk_scalar
 *   [4] SAVE        tiempo de of_save_results
 *   TOTAL           suma de las cuatro etapas
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
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "of.h"

/* Resolucion de trabajo acordada por el equipo. Si los frames difieren
 * solo se advierte; no se detiene la ejecucion. */
#define EXPECTED_WIDTH  1280
#define EXPECTED_HEIGHT 720

typedef struct {
    const char *path1;
    const char *path2;
    const char *out_path;
    int   has_gt;
    float gt_dx;
    float gt_dy;
} Args;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Uso: %s <frame1> <frame2> [salida] [--gt-shift DX DY]\n"
            "  salida            CSV de resultados (defecto: results/last_run.csv\n"
            "                    o ../results/last_run.csv si no existe results/)\n"
            "  --gt-shift DX DY  ground truth uniforme en pixeles (para EPE)\n",
            prog);
}

/* Se ejecuta desde la raiz (results/ existe) o desde build/ o scripts/. */
static const char *default_out_path(void)
{
    struct stat st;
    if (stat("results", &st) == 0 && S_ISDIR(st.st_mode)) {
        return "results/last_run.csv";
    }
    return "../results/last_run.csv";
}

/* Devuelve 0 si los argumentos son validos, -1 en caso contrario. */
static int parse_args(int argc, char **argv, Args *a)
{
    int positional = 0;

    a->path1 = NULL;
    a->path2 = NULL;
    a->out_path = default_out_path();
    a->has_gt = 0;
    a->gt_dx = 0.0f;
    a->gt_dy = 0.0f;

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

/* Reserva un OFImage con las mismas dimensiones que 'src'. */
static int alloc_like(OFImage *dst, const OFImage *src)
{
    dst->width  = src->width;
    dst->height = src->height;
    dst->data   = (float *)malloc((size_t)src->width * (size_t)src->height
                                  * sizeof(float));
    return dst->data ? 0 : -1;
}

/* Campo de flujo constante (dx, dy) en todos los pixeles. */
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
    Args args;
    if (parse_args(argc, argv, &args) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    OFImage raw1 = {0}, raw2 = {0};   /* etapa 1: float [0,255]          */
    OFImage pp1  = {0}, pp2  = {0};   /* etapa 2: float [0,1], suavizado */
    OFFlow  flow = {0};               /* etapa 3: campo (u, v)           */
    OFFlow  gt   = {0};               /* ground truth uniforme (opcional) */
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

    /* [2] Preprocesamiento (ambos frames) */
    t0 = now_ms();
    if ((rc = of_preprocess(&raw1, &pp1)) != 0 ||
        (rc = of_preprocess(&raw2, &pp2)) != 0) {
        fprintf(stderr, "Error en of_preprocess (codigo %d)\n", rc);
        goto cleanup;
    }
    t_pre = now_ms() - t0;

/* [3] Lucas-Kanade Scalar vs NEON */
    t0 = now_ms();
    rc = of_lk_scalar(&pp1, &pp2, &flow);
    t_lk = now_ms() - t0;
    if (rc != 0) {
        fprintf(stderr, "Error en of_lk_scalar (codigo %d)\n", rc);
        goto cleanup;
    }

    /* Ejecución y perfilado NEON */
    double t0_neon = now_ms();
    rc = of_lk_neon(&pp1, &pp2, &flow);
    double t_neon = now_ms() - t0_neon;
    if (rc != 0) {
        fprintf(stderr, "Error en of_lk_neon (codigo %d)\n", rc);
        goto cleanup;
    }

    /* Imprimir métricas de ejecución */
    printf("[1] IO          %10.3f ms\n", t_load);
    printf("[2] PREPROCESS  %10.3f ms\n", t_pre);
    printf("[3] LK_SCALAR   %10.3f ms\n", t_lk);
    printf("[4] LK_NEON     %10.3f ms\n", t_neon);
    printf("[5] SAVE        %10.3f ms\n", t_save);
    printf("TOTAL           %10.3f ms\n", t_load + t_pre + t_lk + t_save);

    /* Diagnostico del resultado (no forma parte de los tiempos) */
    printf("Frames: %dx%d\n", raw1.width, raw1.height);
    print_flow_stats(&flow);

    if (args.has_gt) {
        if (make_uniform_flow(&gt, flow.width, flow.height,
                              args.gt_dx, args.gt_dy) != 0) {
            fprintf(stderr, "Error: sin memoria para el ground truth\n");
            goto cleanup;
        }
        const float epe = of_compute_epe(&flow, &gt);
        if (epe < 0.0f) {
            fprintf(stderr, "Error: of_compute_epe devolvio un valor invalido\n");
        } else {
            printf("EPE vs GT uniforme (%.3f, %.3f): %.4f px\n",
                   args.gt_dx, args.gt_dy, epe);
        }
    }

    /* [4] Guardado de resultados */
    t0 = now_ms();
    of_save_results(args.out_path, &flow, t_lk);
    t_save = now_ms() - t0;
    {
        /* of_save_results no informa errores: se comprueba que el archivo exista */
        FILE *chk = fopen(args.out_path, "r");
        if (chk) {
            fclose(chk);
            printf("Resultado guardado en %s\n", args.out_path);
        } else {
            fprintf(stderr, "Aviso: no se pudo escribir '%s' (existe la carpeta?)\n",
                    args.out_path);
        }
    }

    /* Resumen de tiempos: formato fijo que lee scripts/profile.sh */
    printf("[1] IO          %10.3f ms\n", t_load);
    printf("[2] PREPROCESS  %10.3f ms\n", t_pre);
    printf("[3] LK_SCALAR   %10.3f ms\n", t_lk);
    printf("[4] SAVE        %10.3f ms\n", t_save);
    printf("TOTAL           %10.3f ms\n", t_load + t_pre + t_lk + t_save);

    status = 0;

cleanup:
    free(raw1.data);
    free(raw2.data);
    free(pp1.data);
    free(pp2.data);
    free(flow.u);
    free(flow.v);
    free(gt.u);
    free(gt.v);
    return status;
}
