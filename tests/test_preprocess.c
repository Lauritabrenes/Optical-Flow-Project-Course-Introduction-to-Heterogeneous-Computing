/*
 * ============================================================================
 *  test_preprocess.c  —  Unit tests for Stage 2 (of_preprocess)
 * ============================================================================
 *
 *  PURPOSE
 *  -------
 *  A tiny, dependency-free test harness that verifies the behavior guaranteed
 *  by the team contract (see docs/decisions.md) WITHOUT needing a real image
 *  file. Each test builds a small synthetic OFImage in memory, runs it through
 *  of_preprocess, and checks the result.
 *
 *  WHAT WE VERIFY
 *  --------------
 *    1. Argument validation      -> invalid inputs return OF_ERR_ARGS (-1).
 *    2. Normalization            -> [0,255] is mapped to [0,1].
 *    3. Dimension preservation   -> out keeps the input width/height.
 *    4. Blur behavior            -> a constant image stays constant; a single
 *                                   bright pixel (impulse) spreads to its
 *                                   neighbors and its peak is attenuated.
 *
 *  HOW TO BUILD & RUN
 *  ------------------
 *    Via CMake (from build/):   ctest --output-on-failure
 *    Direct with gcc:
 *        gcc -Iinclude -o test_preprocess \
 *            src/preprocess.c tests/test_preprocess.c -lm
 *        ./test_preprocess
 *
 *  Exit code is 0 if all tests pass, non-zero otherwise (so CI / ctest can
 *  detect failures automatically).
 * ============================================================================
 */

#include "of.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Return codes — must mirror the ones used inside preprocess.c. */
#define OF_OK          0
#define OF_ERR_ARGS   -1

/* ----------------------------------------------------------------------------
 *  Minimal test framework
 *  A single CHECK macro records one assertion: it bumps the run counter, and
 *  the pass counter only when the condition holds, printing a PASS/FAIL line.
 *  Keeping this tiny avoids pulling in any external testing dependency.
 * -------------------------------------------------------------------------- */
static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define CHECK(condition, description) do {                         \
    g_tests_run++;                                                 \
    if (condition) {                                               \
        g_tests_passed++;                                          \
        printf("  [PASS] %s\n", (description));                    \
    } else {                                                       \
        printf("  [FAIL] %s\n", (description));                    \
    }                                                              \
} while (0)


/* ============================================================================
 *  Small helpers to build synthetic images
 * ============================================================================
 */

/*
 * fill_constant
 * -------------
 * Set every pixel of a buffer to the same value. Used to build a flat image.
 */
static void fill_constant(float *data, size_t count, float value)
{
    for (size_t i = 0; i < count; ++i)
        data[i] = value;
}

/*
 * make_impulse
 * ------------
 * Build an image that is zero everywhere except one bright pixel in the
 * center. This is the classic way to observe a blur kernel's shape: after
 * blurring, the single spike should spread out into its neighbors.
 */
static void make_impulse(float *data, int width, int height, float peak)
{
    fill_constant(data, (size_t)width * height, 0.0f);
    data[(height / 2) * width + (width / 2)] = peak;
}


/* ============================================================================
 *  Individual test cases
 *  Each returns nothing; it just performs CHECKs. Splitting them keeps main()
 *  readable and lets a reader see exactly what each scenario proves.
 * ============================================================================
 */

/*
 * test_argument_validation
 * -------------------------
 * of_preprocess must reject NULL structs and NULL data pointers with
 * OF_ERR_ARGS, before doing any work. This protects against caller mistakes.
 */
static void test_argument_validation(void)
{
    printf("Test 1: argument validation\n");

    OFImage null_data_in  = { NULL, 8, 8 };
    OFImage null_data_out = { NULL, 8, 8 };

    CHECK(of_preprocess(NULL, NULL) == OF_ERR_ARGS,
          "NULL, NULL -> OF_ERR_ARGS");
    CHECK(of_preprocess(&null_data_in, &null_data_out) == OF_ERR_ARGS,
          "NULL data pointers -> OF_ERR_ARGS");
}

/*
 * test_normalization_and_constant
 * --------------------------------
 * Feed a fully white image (all pixels = 255). We expect:
 *   - return value OF_OK,
 *   - width/height preserved on the output,
 *   - every output pixel ~= 1.0 (255 normalized to 1.0, and blurring a
 *     constant image leaves it unchanged because the kernel weights sum to 1).
 */
static void test_normalization_and_constant(float *in_data, float *out_data,
                                            int width, int height)
{
    printf("Test 2: normalization [0,255]->[0,1] on a constant image\n");

    const size_t count = (size_t)width * height;
    fill_constant(in_data, count, 255.0f);

    OFImage in  = { in_data,  width, height };
    OFImage out = { out_data, width, height };

    int rc = of_preprocess(&in, &out);

    CHECK(rc == OF_OK, "returns OF_OK");
    CHECK(out.width == width && out.height == height,
          "width/height preserved");

    int all_close_to_one = 1;
    for (size_t i = 0; i < count; ++i) {
        if (fabsf(out_data[i] - 1.0f) > 1e-4f) { all_close_to_one = 0; break; }
    }
    CHECK(all_close_to_one,
          "255 normalizes to ~1.0 and blur leaves a constant unchanged");
}

/*
 * test_impulse_spread
 * -------------------
 * Feed an impulse (single bright center pixel). After normalize + blur:
 *   - the center value drops below 1.0 (energy is spread out),
 *   - an adjacent pixel becomes > 0 (energy leaked into the neighborhood),
 *   - the center is still greater than the neighbor (peak stays in the middle).
 * Together these confirm the Gaussian blur is actually running and is centered.
 */
static void test_impulse_spread(float *in_data, float *out_data,
                                int width, int height)
{
    printf("Test 3: Gaussian blur spreads a central impulse\n");

    make_impulse(in_data, width, height, 255.0f);

    OFImage in  = { in_data,  width, height };
    OFImage out = { out_data, width, height };

    int rc = of_preprocess(&in, &out);
    CHECK(rc == OF_OK, "returns OF_OK");

    const int   ci = (height / 2) * width + (width / 2);   /* center index   */
    const float center   = out_data[ci];
    const float neighbor = out_data[ci + 1];               /* pixel to the right */

    CHECK(center < 1.0f,   "impulse peak is attenuated (< 1.0 after norm+blur)");
    CHECK(neighbor > 0.0f, "energy spreads to neighbors (> 0)");
    CHECK(center > neighbor, "center remains greater than neighbor");
}


/* ============================================================================
 *  Entry point
 * ============================================================================
 */
int main(void)
{
    const int    width  = 8;
    const int    height = 8;
    const size_t count  = (size_t)width * height;

    /* The CALLER allocates both buffers — matches the memory contract that
     * of_preprocess relies on (it never allocates in/out data itself). */
    float *in_data  = (float *)malloc(count * sizeof(float));
    float *out_data = (float *)malloc(count * sizeof(float));
    if (in_data == NULL || out_data == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(in_data);
        free(out_data);
        return 2;
    }

    /* Run the three scenarios. */
    test_argument_validation();
    test_normalization_and_constant(in_data, out_data, width, height);
    test_impulse_spread(in_data, out_data, width, height);

    printf("\nSummary: %d/%d tests passed\n", g_tests_passed, g_tests_run);

    free(in_data);
    free(out_data);

    /* Non-zero exit if any test failed, so ctest / CI can detect it. */
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
