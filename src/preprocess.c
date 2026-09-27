/*
 * ============================================================================
 *  preprocess.c  —  Stage 2 of the Optical Flow C pipeline (Persona 2)
 * ============================================================================
 *
 *  WHAT THIS STAGE DOES
 *  --------------------
 *  This file takes a grayscale image and prepares it so the optical-flow
 *  algorithm (Stage 3, lk_scalar.c / lk_neon.c) can work with it reliably.
 *  "Preparing" here means exactly two things, in this order:
 *
 *      1. NORMALIZE intensity from the [0, 255] range to the [0, 1] range.
 *         Optical flow math (gradients, least squares) is numerically nicer
 *         when pixel values are small floats around 0..1 instead of 0..255.
 *
 *      2. SMOOTH the image with a Gaussian blur (sigma = 1.0).
 *         Lucas-Kanade estimates motion from image gradients. Raw images
 *         contain high-frequency noise that produces noisy gradients and
 *         unstable flow. A light Gaussian blur removes that noise while
 *         keeping the overall structure, which makes the gradients stable.
 *
 *  WHAT THIS STAGE DOES *NOT* DO (on purpose)
 *  ------------------------------------------
 *    - It does NOT read image files            -> that is io.c (Persona 1).
 *    - It does NOT convert RGB to grayscale     -> that is io.c (BT.601 luma).
 *    - It does NOT build the image pyramid      -> that is lk_*.c (Persona 3).
 *    - It does NOT compute gradients Ix/Iy/It   -> that is lk_*.c (Persona 3).
 *  Keeping the stage boundaries clean is what lets four people work in
 *  parallel without stepping on each other's code.
 *
 *  TEAM CONTRACT (agreed with the group, see docs/decisions.md)
 *  ------------------------------------------------------------
 *    - INPUT  : OFImage, grayscale, float32, values in [0, 255].
 *    - OUTPUT : OFImage, grayscale, float32, values in [0, 1], smoothed.
 *    - MEMORY : the CALLER allocates both in->data and out->data
 *               (width * height floats each). This stage never allocates
 *               or frees those two buffers. (It only uses a small internal
 *               scratch buffer for the blur, which it frees itself.)
 *    - RETURN : 0 on success, negative on error:
 *                   OF_OK        (0)   success
 *                   OF_ERR_ARGS (-1)   invalid arguments
 *                   OF_ERR_ALLOC(-2)   internal allocation failed
 *    - of.h is the shared contract and is NEVER modified here.
 *
 *  DESIGN NOTE — WHY IT IS SPLIT INTO SMALL FUNCTIONS
 *  --------------------------------------------------
 *  Each helper does exactly ONE job and is documented on its own. This makes
 *  the file easy to read top-to-bottom, easy to unit-test piece by piece,
 *  and easy to optimize later (e.g. replacing the blur inner loops with NEON
 *  in a future revision) without touching the rest of the logic.
 * ============================================================================
 */

#include "of.h"

#include <math.h>     /* expf, ceilf                     */
#include <stdlib.h>   /* malloc, free                    */

/* ----------------------------------------------------------------------------
 *  CONFIGURATION CONSTANTS
 *  These are the only "magic numbers" in the file. They are named and grouped
 *  here so a reader can see every tunable in one place, and so a teammate can
 *  change behavior (e.g. blur strength) without hunting through the code.
 * -------------------------------------------------------------------------- */

/* Return codes — must match the convention used by io.c and the rest. */
#define OF_OK           0
#define OF_ERR_ARGS    -1
#define OF_ERR_ALLOC   -2

/* Intensity normalization: multiplying by 1/255 turns [0,255] into [0,1].
 * We precompute the reciprocal because a multiply is cheaper than a divide
 * and we do it once per pixel. */
#define OF_INV_U8_MAX   (1.0f / 255.0f)

/* Gaussian blur strength. sigma controls how much smoothing is applied.
 * Larger sigma = blurrier image = more noise removed but more detail lost.
 * 1.0 was agreed with Persona 3 as the default for this pipeline. */
#define OF_DEFAULT_SIGMA  1.0f

/* A Gaussian kernel technically extends to infinity, but its weights become
 * negligible past ~3*sigma. So we cut the kernel off at radius = ceil(3*sigma).
 * OF_MAX_KERNEL caps the number of taps so we can use a fixed-size stack array
 * for the kernel instead of a heap allocation. (2*radius + 1) must fit here. */
#define OF_MAX_KERNEL   64


/* ============================================================================
 *  SECTION 1 — GAUSSIAN KERNEL CONSTRUCTION
 * ============================================================================
 */

/*
 * of_compute_kernel_radius
 * ------------------------
 * Decide how many samples on each side of the center the Gaussian kernel needs.
 *
 * A Gaussian's weights are effectively zero beyond ~3 standard deviations, so
 * radius = ceil(3 * sigma) captures essentially all of the useful weight.
 *
 * We also clamp the result so the full kernel length (2*radius + 1) never
 * exceeds OF_MAX_KERNEL, which keeps the fixed-size kernel array safe.
 *
 * Returns: the kernel radius (>= 1).
 */
static int of_compute_kernel_radius(float sigma)
{
    int radius = (int)ceilf(3.0f * sigma);

    if (radius < 1)
        radius = 1;                         /* always at least a 3-tap kernel */

    if (2 * radius + 1 > OF_MAX_KERNEL)
        radius = (OF_MAX_KERNEL - 1) / 2;   /* clamp to the array capacity    */

    return radius;
}

/*
 * of_build_gaussian_kernel_1d
 * ---------------------------
 * Fill 'kernel' with a 1-D Gaussian, then normalize it so all weights sum to 1.
 *
 * WHY 1-D FOR A 2-D BLUR?
 *   A 2-D Gaussian blur is "separable": blurring horizontally with a 1-D
 *   kernel and then blurring that result vertically with the SAME 1-D kernel
 *   gives the exact same output as a full 2-D kernel, but far cheaper
 *   (O(2*k) work per pixel instead of O(k*k)). So we only ever build a 1-D
 *   kernel and apply it twice.
 *
 * WHY NORMALIZE?
 *   Each output pixel is a weighted average of its neighbors. If the weights
 *   summed to more than 1 the image would get brighter; less than 1 and it
 *   would get darker. Dividing by the sum forces the weights to sum to 1.0,
 *   so average brightness is preserved.
 *
 * Layout: kernel[0 .. 2*radius] where index (i + radius) holds the weight for
 * offset i, with i ranging from -radius to +radius (0 is the center tap).
 *
 * 'kernel' must have room for at least (2*radius + 1) floats.
 */
static void of_build_gaussian_kernel_1d(float *kernel, int radius, float sigma)
{
    const float inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;

    /* Step 1: evaluate the (unnormalized) Gaussian at each offset. */
    for (int i = -radius; i <= radius; ++i) {
        float weight = expf(-(float)(i * i) * inv_two_sigma_sq);
        kernel[i + radius] = weight;
        sum += weight;
    }

    /* Step 2: normalize so the weights sum to exactly 1.0. */
    const int length = 2 * radius + 1;
    for (int j = 0; j < length; ++j)
        kernel[j] /= sum;
}


/* ============================================================================
 *  SECTION 2 — INTENSITY NORMALIZATION
 * ============================================================================
 */

/*
 * of_normalize_u8_to_unit
 * ------------------------
 * Convert every pixel from the [0, 255] range to the [0, 1] range by
 * multiplying by 1/255.
 *
 * 'src' and 'dst' may point to the SAME buffer (in-place is fine) or to
 * different buffers; the loop reads each element before writing it, so
 * overlap is safe. 'count' is the total number of pixels (width * height).
 */
static void of_normalize_u8_to_unit(const float *src, float *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = src[i] * OF_INV_U8_MAX;
}


/* ============================================================================
 *  SECTION 3 — SEPARABLE GAUSSIAN BLUR
 * ============================================================================
 */

/*
 * of_clamp_index
 * --------------
 * Clamp a coordinate to the valid range [0, size-1].
 *
 * This implements "replicate" (a.k.a. "clamp-to-edge") border handling: when
 * the kernel reaches past the image edge, we reuse the nearest edge pixel
 * instead of reading out of bounds. This avoids dark halos at the borders
 * that a "treat outside as zero" policy would create.
 */
static inline int of_clamp_index(int coord, int size)
{
    if (coord < 0)      return 0;
    if (coord >= size)  return size - 1;
    return coord;
}

/*
 * of_blur_horizontal
 * -------------------
 * First of the two separable passes: blur ALONG each row (the x direction).
 *
 * For every output pixel we take a weighted sum of its horizontal neighbors
 * using the 1-D kernel. Reads come from 'src', results go to 'dst'. Rows are
 * processed independently, so this pass is trivially parallelizable later.
 */
static void of_blur_horizontal(const float *src, float *dst,
                               int width, int height,
                               const float *kernel, int radius)
{
    for (int y = 0; y < height; ++y) {
        const float *src_row = src + (size_t)y * width;
        float       *dst_row = dst + (size_t)y * width;

        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;
            for (int t = -radius; t <= radius; ++t) {
                int xx = of_clamp_index(x + t, width);
                acc += src_row[xx] * kernel[t + radius];
            }
            dst_row[x] = acc;
        }
    }
}

/*
 * of_blur_vertical
 * ----------------
 * Second separable pass: blur DOWN each column (the y direction).
 *
 * Same idea as the horizontal pass but the neighbors are stacked vertically,
 * so we step by 'width' elements to move one row up/down. Feeding this the
 * output of the horizontal pass produces the final 2-D Gaussian blur.
 */
static void of_blur_vertical(const float *src, float *dst,
                             int width, int height,
                             const float *kernel, int radius)
{
    for (int y = 0; y < height; ++y) {
        float *dst_row = dst + (size_t)y * width;

        for (int x = 0; x < width; ++x) {
            float acc = 0.0f;
            for (int t = -radius; t <= radius; ++t) {
                int yy = of_clamp_index(y + t, height);
                acc += src[(size_t)yy * width + x] * kernel[t + radius];
            }
            dst_row[x] = acc;
        }
    }
}

/*
 * of_gaussian_blur_separable
 * --------------------------
 * Apply a full 2-D Gaussian blur by running the horizontal pass then the
 * vertical pass. 'scratch' is a caller-provided temporary buffer of
 * width*height floats used to hold the intermediate (horizontally-blurred)
 * image between the two passes.
 *
 * Data flow:  src --(horizontal)--> scratch --(vertical)--> dst
 *
 * 'src' and 'dst' may be the same buffer; 'scratch' must be distinct.
 */
static void of_gaussian_blur_separable(const float *src, float *dst,
                                       float *scratch,
                                       int width, int height,
                                       const float *kernel, int radius)
{
    of_blur_horizontal(src, scratch, width, height, kernel, radius);
    of_blur_vertical(scratch, dst, width, height, kernel, radius);
}


/* ============================================================================
 *  SECTION 4 — VALIDATION HELPER
 * ============================================================================
 */

/*
 * of_images_valid
 * ----------------
 * Sanity-check the input and output images before doing any work.
 *
 * We require: non-NULL structs, non-NULL data pointers (remember, the CALLER
 * is responsible for allocating them), and positive dimensions. Returning a
 * clear boolean here keeps the main function readable.
 */
static int of_images_valid(const OFImage *in, const OFImage *out)
{
    if (in == NULL || out == NULL)          return 0;
    if (in->data == NULL || out->data == NULL) return 0;
    if (in->width <= 0 || in->height <= 0)  return 0;
    return 1;
}


/* ============================================================================
 *  SECTION 5 — PUBLIC ENTRY POINT (declared in of.h)
 * ============================================================================
 */

/*
 * of_preprocess
 * -------------
 * Stage 2 entry point called by bench_main.
 *
 * Pipeline performed here:
 *      in (grayscale, [0,255])
 *        -> normalize to [0,1]        (written into out->data)
 *        -> Gaussian blur sigma=1.0   (in place on out->data)
 *      out (grayscale, [0,1], smoothed)
 *
 * The input buffer 'in' is treated as read-only. All results are written into
 * the caller-allocated 'out->data'. Only a small internal scratch buffer is
 * allocated (and freed) here for the blur's intermediate result.
 *
 * Returns OF_OK on success, or a negative OF_ERR_* code on failure.
 */
int of_preprocess(const OFImage *in, OFImage *out)
{
    /* --- 1. Validate arguments before touching any memory. --- */
    if (!of_images_valid(in, out))
        return OF_ERR_ARGS;

    const int    width  = in->width;
    const int    height = in->height;
    const size_t count  = (size_t)width * (size_t)height;

    /* Propagate dimensions to the output image. */
    out->width  = width;
    out->height = height;

    /* --- 2. Normalize intensity [0,255] -> [0,1] into the output buffer. --- */
    of_normalize_u8_to_unit(in->data, out->data, count);

    /* --- 3. Build the 1-D Gaussian kernel for the chosen sigma. --- */
    const int radius = of_compute_kernel_radius(OF_DEFAULT_SIGMA);
    float     kernel[OF_MAX_KERNEL];
    of_build_gaussian_kernel_1d(kernel, radius, OF_DEFAULT_SIGMA);

    /* --- 4. Allocate the scratch buffer the separable blur needs. ---
     * This is the ONLY heap allocation in this stage, and it is fully owned
     * and freed here — it does not violate the "caller allocates out->data"
     * contract, because out->data itself is never allocated or freed here. */
    float *scratch = (float *)malloc(count * sizeof(float));
    if (scratch == NULL)
        return OF_ERR_ALLOC;

    /* --- 5. Smooth the normalized image in place (out -> out via scratch). --- */
    of_gaussian_blur_separable(out->data, out->data, scratch,
                               width, height, kernel, radius);

    /* --- 6. Release the scratch buffer and report success. --- */
    free(scratch);
    return OF_OK;
}
