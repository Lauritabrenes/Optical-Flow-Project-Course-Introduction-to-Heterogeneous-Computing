/*
 * ============================================================================
 *  src/main.c
 * ----------------------------------------------------------------------------
 *  Punto de entrada del benchmark de Optical Flow.
 *
 *  Ejecuta el pipeline completo de 4 etapas:
 *      [1] IO          -> carga de dos frames desde disco
 *      [2] PREPROCESS  -> normalizacion + desenfoque gaussiano
 *      [3] LK_SCALAR   -> Lucas-Kanade piramidal (o LK_NEON en ARM)
 *      [4] SAVE        -> persistencia de resultados en CSV
 *      TOTAL           -> tiempo total del pipeline
 *
 *  Imprime los tiempos por etapa con un formato que scripts/profile.sh
 *  puede parsear sin ambiguedad:
 *      [1] IO          <tiempo> ms
 *      [2] PREPROCESS  <tiempo> ms
 *      [3] LK_SCALAR   <tiempo> ms
 *      [4] SAVE        <tiempo> ms
 *      TOTAL           <tiempo> ms
 *
 *  IMPORTANTE: NO imprimir lineas adicionales que contengan la palabra
 *  "TOTAL" o "[N]" para evitar que el grep del script de perfilado las
 *  capture por error. En particular, evitar mensajes del tipo
 *  "Pipeline TOTAL ejecutado en X ms".
 *
 *  Uso:
 *      ./bench_main <frame1> <frame2>
 *
 *  Autor: Grupo 3
 *  Curso: EL5859 Computacion Heterogenea (TEC)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "of.h"

/* Codigo de retorno que indica exito en las funciones del modulo of */
#define OF_OK 0

/* ============================================================================
 *  Utilidad de medicion de tiempo
 * ============================================================================
 */

/*
 * now_ms
 * ------
 * Devuelve el tiempo actual del reloj monotónico del sistema en milisegundos.
 *
 * Usa CLOCK_MONOTONIC porque no se ve afectado por ajustes del reloj del
 * sistema (NTP, cambios manuales), lo que lo hace ideal para medir duraciones.
 *
 * Retorna: marca de tiempo en milisegundos como double.
 */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 *  Punto de entrada
 * ============================================================================
 */

int main(int argc, char **argv)
{
    /* --- 0. Validacion de argumentos --- */
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <frame1> <frame2>\n", argv[0]);
        return 1;
    }

    /* --- Estructuras de datos --- */
    OFImage f1 = {0}, f2 = {0};     /* Frames originales (gris, [0,255])    */
    OFImage p1 = {0}, p2 = {0};     /* Frames preprocesados (gris, [0,1])   */
    OFFlow  flow = {0};             /* Flujo optico resultante              */

    /* Variables de tiempo por etapa */
    double t_start, t_after_io, t_after_pre, t_after_lk, t_after_save;

    /* ========================================================================
     *  Inicio de la medicion total
     * ======================================================================== */
    t_start = now_ms();

    /* ========================================================================
     *  ETAPA 1 — IO: carga de frames
     * ======================================================================== */
    if (of_load_frame(argv[1], &f1) != OF_OK) {
        fprintf(stderr, "ERROR: fallo al cargar frame1 (%s)\n", argv[1]);
        return 1;
    }
    if (of_load_frame(argv[2], &f2) != OF_OK) {
        fprintf(stderr, "ERROR: fallo al cargar frame2 (%s)\n", argv[2]);
        free(f1.data);
        return 1;
    }

    t_after_io = now_ms();

    /* ========================================================================
     *  ETAPA 2 — PREPROCESS: normalizacion + Gaussiano
     * ======================================================================== */

    /* Reserva memoria para los frames preprocesados (responsabilidad del
     * llamador, segun el contrato definido en docs/decisions.md). */
    p1.width  = f1.width;
    p1.height = f1.height;
    p1.data   = (float *)malloc((size_t)p1.width * p1.height * sizeof(float));

    p2.width  = f2.width;
    p2.height = f2.height;
    p2.data   = (float *)malloc((size_t)p2.width * p2.height * sizeof(float));

    if (!p1.data || !p2.data) {
        fprintf(stderr, "ERROR: fallo al reservar memoria para preprocesamiento\n");
        free(f1.data); free(f2.data);
        free(p1.data); free(p2.data);
        return 1;
    }

    if (of_preprocess(&f1, &p1) != OF_OK) {
        fprintf(stderr, "ERROR: fallo en preprocesamiento de frame1\n");
        free(f1.data); free(f2.data);
        free(p1.data); free(p2.data);
        return 1;
    }
    if (of_preprocess(&f2, &p2) != OF_OK) {
        fprintf(stderr, "ERROR: fallo en preprocesamiento de frame2\n");
        free(f1.data); free(f2.data);
        free(p1.data); free(p2.data);
        return 1;
    }

    t_after_pre = now_ms();

    /* ========================================================================
     *  ETAPA 3 — LK: Lucas-Kanade piramidal
     * ======================================================================== */

    /* Dimensiona el flujo de salida igual que el frame preprocesado */
    flow.width  = p1.width;
    flow.height = p1.height;
    const size_t n = (size_t)flow.width * flow.height;

    /* calloc: inicializa a cero (util si LK no cubre todos los pixeles) */
    flow.u = (float *)calloc(n, sizeof(float));
    flow.v = (float *)calloc(n, sizeof(float));

    if (!flow.u || !flow.v) {
        fprintf(stderr, "ERROR: fallo al reservar memoria para el flujo optico\n");
        free(f1.data); free(f2.data);
        free(p1.data); free(p2.data);
        free(flow.u);  free(flow.v);
        return 1;
    }

    if (of_lk_scalar(&p1, &p2, &flow) != OF_OK) {
        fprintf(stderr, "ERROR: fallo en Lucas-Kanade escalar\n");
        free(f1.data); free(f2.data);
        free(p1.data); free(p2.data);
        free(flow.u);  free(flow.v);
        return 1;
    }

    t_after_lk = now_ms();

    /* ========================================================================
     *  ETAPA 4 — SAVE: persistencia de resultados
     * ======================================================================== */

    /* Calcula el tiempo acumulado hasta ahora para pasarlo al CSV */
    const double lk_elapsed_ms = t_after_lk - t_start;

    /* Persiste el flujo optico y el tiempo en formato CSV */
    of_save_results("../results/last_run.csv", &flow, lk_elapsed_ms);

    t_after_save = now_ms();

    /* ========================================================================
     *  Reporte de tiempos por etapa
     * ========================================================================
     *
     *  CRITICO: el formato debe ser exactamente:
     *      [N] NAME <tiempo> ms
     *      TOTAL  <tiempo> ms
     *
     *  No agregar lineas adicionales que contengan "TOTAL" o "[N]".
     *  No cambiar el orden de los campos ni la posicion del valor.
     */
    printf("[1] IO          %.3f ms\n", t_after_io   - t_start);
    printf("[2] PREPROCESS  %.3f ms\n", t_after_pre  - t_after_io);
    printf("[3] LK_SCALAR   %.3f ms\n", t_after_lk   - t_after_pre);
    printf("[4] SAVE        %.3f ms\n", t_after_save - t_after_lk);
    printf("TOTAL           %.3f ms\n", t_after_save - t_start);

    /* ========================================================================
     *  Liberacion de memoria
     * ======================================================================== */
    free(f1.data);
    free(f2.data);
    free(p1.data);
    free(p2.data);
    free(flow.u);
    free(flow.v);

    return 0;
}