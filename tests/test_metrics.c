/**
 * @file    test_metrics.c
 * @brief   Suite de pruebas unitarias para métricas de flujo óptico.
 *
 * Este archivo valida el correcto funcionamiento de las funciones
 * definidas en "of.h", específicamente:
 *   - of_compute_epe()      -> Cálculo del End-Point Error (EPE)
 *   - of_save_results()     -> Persistencia de resultados en CSV
 *
 * @author  (tu nombre)
 * @date    (fecha)
 */

#include "of.h"      /* Define OFFlow y las funciones a probar */
#include <math.h>    /* fabsf() para comparaciones con tolerancia */
#include <stdio.h>   /* printf, fopen, fclose */
#include <stdlib.h>  /* utilidades generales */

/* ------------------------------------------------------------------ */
/*  Macro de tolerancia para comparaciones con números flotantes      */
/* ------------------------------------------------------------------ */
#define TOL 1e-4f


/* ================================================================== */
/*  TEST 1: EPE con flujos idénticos                                  */
/* ================================================================== */
/**
 * @brief  Verifica que el EPE entre dos flujos idénticos sea 0.
 *
 * Se construyen dos flujos 4x4 con los mismos valores (u=1.0, v=2.0)
 * en todos los píxeles. El resultado esperado es 0.0.
 *
 * @return 1 si la prueba pasa, 0 en caso contrario.
 */
static int test_epe_zero(void)
{
    const int w = 4, h = 4;
    const size_t n = (size_t)w * h;
    float u[16], v[16];

    /* Inicializa ambos flujos con el mismo vector constante */
    for (size_t i = 0; i < n; ++i) {
        u[i] = 1.0f;
        v[i] = 2.0f;
    }

    OFFlow a = { u, v, w, h };
    OFFlow b = { u, v, w, h };

    float epe = of_compute_epe(&a, &b);

    printf("[TEST] EPE identico: %.6f (esperado 0)\n", epe);

    return fabsf(epe) < TOL;
}


/* ================================================================== */
/*  TEST 2: EPE con desplazamiento conocido (3,4) -> 5.0              */
/* ================================================================== */
/**
 * @brief  Verifica el cálculo del EPE con un desplazamiento conocido.
 *
 * Flujo a = (0,0) en todos los píxeles.
 * Flujo b = (3,4) en todos los píxeles.
 * La distancia euclídea esperada es sqrt(3² + 4²) = 5.0
 *
 * @return 1 si la prueba pasa, 0 en caso contrario.
 */
static int test_epe_offset(void)
{
    const int w = 4, h = 4;
    const size_t n = (size_t)w * h;
    float u1[16], v1[16], u2[16], v2[16];

    for (size_t i = 0; i < n; ++i) {
        u1[i] = 0.0f;  v1[i] = 0.0f;
        u2[i] = 3.0f;  v2[i] = 4.0f;
    }

    OFFlow a = { u1, v1, w, h };
    OFFlow b = { u2, v2, w, h };

    float epe = of_compute_epe(&a, &b);

    printf("[TEST] EPE offset (3,4): %.6f (esperado 5)\n", epe);

    return fabsf(epe - 5.0f) < TOL;
}


/* ================================================================== */
/*  TEST 3: EPE con punteros NULL (manejo de error)                   */
/* ================================================================== */
/**
 * @brief  Verifica el manejo de errores cuando los flujos son NULL.
 *
 * Se construyen dos OFFlow con punteros u y v = NULL pero dimensiones
 * válidas. La función debe devolver -1.0 como convención de error.
 *
 * @return 1 si la prueba pasa, 0 en caso contrario.
 */
static int test_epe_invalid(void)
{
    OFFlow a = { NULL, NULL, 4, 4 };
    OFFlow b = { NULL, NULL, 4, 4 };

    float epe = of_compute_epe(&a, &b);

    printf("[TEST] EPE con NULL: %.6f (esperado -1)\n", epe);

    return fabsf(epe - (-1.0f)) < TOL;
}


/* ================================================================== */
/*  TEST 4: Persistencia de resultados en CSV                         */
/* ================================================================== */
/**
 * @brief  Verifica que of_save_results() crea correctamente el CSV.
 *
 * Se genera un flujo 2x2 con valores definidos y se intenta guardar
 * en "../results/test_metrics.csv". El test verifica que el archivo
 * pueda abrirse posteriormente en modo lectura.
 *
 * @return 1 si el archivo existe y es legible, 0 en caso contrario.
 */
static int test_save_results(void)
{
    const int w = 2, h = 2;
    float u[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
    float v[4] = { 1.1f, 1.2f, 1.3f, 1.4f };

    OFFlow flow = { u, v, w, h };

    /* 12.34f se pasa como métrica adicional (probablemente el EPE) */
    of_save_results("../results/test_metrics.csv", &flow, 12.34);

    FILE *fp = fopen("../results/test_metrics.csv", "r");
    if (!fp) {
        printf("[TEST] save_results: archivo no creado\n");
        return 0;
    }
    fclose(fp);

    printf("[TEST] save_results: OK\n");
    return 1;
}


/* ================================================================== */
/*  FUNCIÓN PRINCIPAL                                                 */
/* ================================================================== */
/**
 * @brief  Ejecuta todas las pruebas y reporta el resumen final.
 *
 * @return 0 si todas las pruebas pasan, 1 si al menos una falla.
 */
int main(void)
{
    int pass = 0, total = 0;

    total++; pass += test_epe_zero();
    total++; pass += test_epe_offset();
    total++; pass += test_epe_invalid();
    total++; pass += test_save_results();

    printf("\n%d/%d tests passed\n", pass, total);

    return (pass == total) ? 0 : 1;
}