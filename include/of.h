#ifndef OF_H
#define OF_H

// Imagen en escala de grises, formato punto flotante
typedef struct {
    float *data;
    int width;
    int height;
} OFImage;

// Campo de flujo optico (u, v) por pixel
typedef struct {
    float *u;
    float *v;
    int width;
    int height;
} OFFlow;

// Etapa 1: inicializacion y carga
int of_load_frame(const char *path, OFImage *out);

// Etapa 2: preprocesamiento
int of_preprocess(const OFImage *in, OFImage *out);

// Etapa 3: optical flow
int of_lk_scalar(const OFImage *frame1, const OFImage *frame2, OFFlow *flow);
int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow);

// Etapa 4: resultados y metricas
float of_compute_epe(const OFFlow *estimated, const OFFlow *ground_truth);
void of_save_results(const char *path, const OFFlow *flow, double elapsed_ms);

#endif
