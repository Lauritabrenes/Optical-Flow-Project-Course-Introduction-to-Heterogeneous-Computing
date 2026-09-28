#include "of.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define OF_EPE_INVALID (-1.0f)

float of_compute_epe(const OFFlow *estimated, const OFFlow *ground_truth)
{
    if (estimated == NULL || ground_truth == NULL) return OF_EPE_INVALID;
    if (estimated->u == NULL || estimated->v == NULL) return OF_EPE_INVALID;
    if (ground_truth->u == NULL || ground_truth->v == NULL) return OF_EPE_INVALID;
    if (estimated->width  != ground_truth->width)  return OF_EPE_INVALID;
    if (estimated->height != ground_truth->height) return OF_EPE_INVALID;
    if (estimated->width <= 0 || estimated->height <= 0) return OF_EPE_INVALID;

    const size_t n = (size_t)estimated->width * (size_t)estimated->height;
    double sum = 0.0;
    size_t valid = 0;

    for (size_t i = 0; i < n; ++i) {
        const float ue = estimated->u[i];
        const float ve = estimated->v[i];
        const float ug = ground_truth->u[i];
        const float vg = ground_truth->v[i];

        if (isnan(ug) || isnan(vg) || isinf(ug) || isinf(vg)) continue;
        if (isnan(ue) || isnan(ve) || isinf(ue) || isinf(ve)) continue;

        const double du = (double)ue - (double)ug;
        const double dv = (double)ve - (double)vg;
        sum += sqrt(du * du + dv * dv);
        ++valid;
    }

    if (valid == 0) return OF_EPE_INVALID;
    return (float)(sum / (double)valid);
}

void of_save_results(const char *path, const OFFlow *flow, double elapsed_ms)
{
    if (path == NULL || flow == NULL) return;
    if (flow->u == NULL || flow->v == NULL) return;

    FILE *fp = fopen(path, "w");
    if (fp == NULL) return;

    fprintf(fp, "width,height,elapsed_ms\n");
    fprintf(fp, "%d,%d,%.6f\n", flow->width, flow->height, elapsed_ms);

    fprintf(fp, "u,v\n");
    const size_t n = (size_t)flow->width * (size_t)flow->height;
    for (size_t i = 0; i < n; ++i) {
        fprintf(fp, "%.6f,%.6f\n", flow->u[i], flow->v[i]);
    }

    fclose(fp);
}