/*
 *  lk_neon.c — Stage 3 Vectorized Lucas-Kanade with ARM NEON
 */

#include "of.h"
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(HAVE_NEON)
#include <arm_neon.h>
#endif

#define OF_OK          0
#define OF_ERR_ARGS   -1
#define OF_ERR_ALLOC  -2

int of_lk_neon(const OFImage *frame1, const OFImage *frame2, OFFlow *flow) {
    // TODO: implementacion real (Persona 3, version NEON)
    (void)frame1;
    (void)frame2;
    (void)flow;
    return 0;
}