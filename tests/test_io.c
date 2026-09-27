#include "of.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define OF_OK 0
#define TOLERANCE 0.01f

static int almost_equal(float a, float b)
{
    return fabsf(a - b) < TOLERANCE;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Uso: %s <imagen>\n", argv[0]);
        return 1;
    }

    OFImage image;

    int rc = of_load_frame(argv[1], &image);

    if (rc != OF_OK) {
        printf("[FAIL] No se pudo cargar la imagen\n");
        return 1;
    }

    printf("[PASS] Imagen cargada correctamente\n");
    printf("Width  : %d\n", image.width);
    printf("Height : %d\n", image.height);
    printf("Data   : %s\n", image.data != NULL ? "OK" : "NULL");

    if (image.width != 2 || image.height != 2) {
        printf("[FAIL] Dimensiones incorrectas\n");
        free(image.data);
        return 1;
    }

    const float expected[4] = {
        76.245f,   /* rojo */
        149.685f,  /* verde */
        29.070f,   /* azul */
        255.000f   /* blanco */
    };

    for (int i = 0; i < 4; ++i) {

        printf(
            "Pixel %d: obtenido = %.3f, esperado = %.3f\n",
            i,
            image.data[i],
            expected[i]
        );

        if (!almost_equal(image.data[i], expected[i])) {
            printf("[FAIL] Conversion RGB -> gris incorrecta\n");
            free(image.data);
            return 1;
        }
    }

    printf("[PASS] Conversion BT.601 correcta\n");

    free(image.data);

    printf("[PASS] Test de Inicializacion y Carga completado\n");

    return 0;
}