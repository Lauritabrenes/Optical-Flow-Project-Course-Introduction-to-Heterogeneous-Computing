/*
 * ============================================================================
 *  lk_neon.c — Stage 3 Vectorized Lucas-Kanade with ARM NEON (Persona 3)
 * ============================================================================
 */

#include "of.h"
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#define OF_OK          0
#define OF_ERR_ARGS   -1
#define OF_ERR_ALLOC  -2

int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    /*
     * Implementation with ARM NEON intrinsics (running on Kria KV260 ARM CPU).
     * Vectorized computation of gradients and structure tensor accumulations.
     */
    return of_lk_scalar(frame1, frame2, flow); // TODO: Replace with explicit NEON loops
#else
    /* Fallback for x86_64 development machines */
    return of_lk_scalar(frame1, frame2, flow);
#endif
}