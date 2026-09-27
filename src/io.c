#include "of.h"

#include <stddef.h>
#include <stdlib.h>

/*
 * stb_image must have STB_IMAGE_IMPLEMENTATION defined
 * in exactly one source file.
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


/* Same return-code convention used by the rest of the project. */
#define OF_OK          0
#define OF_ERR_ARGS   -1
#define OF_ERR_ALLOC  -2


/*
 * Stage 1: initialization and image loading.
 *
 * Input:
 *      path -> path to the image file.
 *
 * Output:
 *      out -> grayscale OFImage, float values in [0,255].
 *
 * RGB to grayscale conversion:
 *
 *      Y = 0.299 R + 0.587 G + 0.114 B
 *
 * This follows the BT.601 luma conversion agreed by the team.
 *
 * IMPORTANT:
 *      - Normalization to [0,1] is NOT done here.
 *      - Gaussian blur is NOT done here.
 *      - Those operations belong to preprocess.c.
 */
int of_load_frame(const char *path, OFImage *out)
{
    /* Validate parameters. */
    if (path == NULL || out == NULL)
        return OF_ERR_ARGS;


    /*
     * Initialize the structure so that it remains in a safe state
     * if an error occurs.
     */
    out->data   = NULL;
    out->width  = 0;
    out->height = 0;


    /*
     * Image information returned by stb_image.
     */
    int width;
    int height;
    int channels;


    /*
     * Load the file and force the decoded image to RGB.
     *
     * The final argument (3) means:
     *
     *      R G B
     *
     * regardless of whether the original image was RGB or RGBA.
     */
    unsigned char *rgb = stbi_load(
        path,
        &width,
        &height,
        &channels,
        3
    );


    /* Image could not be opened or decoded. */
    if (rgb == NULL)
        return OF_ERR_ARGS;


    /* Validate image dimensions. */
    if (width <= 0 || height <= 0) {
        stbi_image_free(rgb);
        return OF_ERR_ARGS;
    }


    /*
     * Total pixels.
     *
     * The project uses dense packing:
     *
     *      width * height
     *
     * without padding or stride.
     */
    const size_t count =
        (size_t)width * (size_t)height;


    /*
     * Allocate one float for every grayscale pixel.
     */
    float *gray =
        (float *)malloc(count * sizeof(float));


    if (gray == NULL) {
        stbi_image_free(rgb);
        return OF_ERR_ALLOC;
    }


    /*
     * Convert RGB -> grayscale.
     *
     * stb_image stores the pixels as:
     *
     *      R G B | R G B | R G B | ...
     *
     * The grayscale result remains in [0,255].
     */
    for (size_t i = 0; i < count; ++i) {

        const float r =
            (float)rgb[3 * i + 0];

        const float g =
            (float)rgb[3 * i + 1];

        const float b =
            (float)rgb[3 * i + 2];


        gray[i] =
            0.299f * r +
            0.587f * g +
            0.114f * b;
    }


    /*
     * RGB is temporary and is no longer needed.
     */
    stbi_image_free(rgb);


    /*
     * Deliver the result using the OFImage structure that
     * already exists in of.h.
     */
    out->data   = gray;
    out->width  = width;
    out->height = height;


    return OF_OK;
}
