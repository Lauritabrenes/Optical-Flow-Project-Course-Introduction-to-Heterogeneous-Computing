# Team Decisions — Optical Flow C Prototype

Record of agreements that affect the shared interface (`include/of.h`) and the
boundaries between pipeline stages. **`of.h` is never modified without team
agreement.**

Pipeline: `io.c → preprocess.c → lk_scalar.c / lk_neon.c → metrics.c`

---

## Stage 2 Contract — Preprocessing (`preprocess.c`, Persona 2)

| # | Item | Confirmed decision |
|---|------|--------------------|
| 1 | RGB → grayscale conversion | **Persona 1 in `io.c`**, using **BT.601 luma** (`0.299·R + 0.587·G + 0.114·B`). Color never leaves `io.c`, so the single-channel `OFImage` in `of.h` is never broken. |
| 2 | Input intensity range | `of_load_frame` delivers grayscale in **`[0,255]`** (`float32`). Normalization to **`[0,1]` happens in `preprocess.c`**. |
| 3 | Memory ownership | The **caller allocates** `out->data` (`width*height` floats). `of_preprocess` neither allocates nor frees it. |
| 4 | Return codes | **`0 = OK`**, negative on error: **`-1` = invalid arguments**, **`-2` = allocation failure**. |
| 5 | Working resolution | **720p (1280×720)**. Fixed resolution per run → buffers allocated once and reused. |
| 6 | Gaussian pre-blur | **Yes**, separable, with **`sigma = 1.0`**. |
| 7 | Memory layout | Dense packing: row = `width` floats, total = `width*height`, no stride/padding (per `of.h`). |

### `of_preprocess` internal flow
`OFImage` grayscale `[0,255]` → **normalize to `[0,1]`** → **smooth (Gaussian σ=1.0)** → write to `out` (allocated by the caller) → hand off to Persona 3.

### Persona 2 guarantees to the rest of the team
- Does not modify `of.h`.
- Does not touch the input buffer (`in` is `const`); only writes to `out`.
- Delivers to Persona 3 a grayscale `float32` `OFImage`, same `width/height`, ready for pyramid/gradients.

---

## Responsibility boundaries (reminder)

- **Persona 1 (`io.c`)** — load, RGB→grayscale (BT.601), deliver `OFImage` in `[0,255]`.
- **Persona 2 (`preprocess.c`)** — normalize to `[0,1]` + Gaussian blur (σ=1.0).
- **Persona 3 (`lk_*.c`)** — **pyramid, gradients (Ix/Iy/It), and windowing** (confirmed as theirs).
- **Persona 4 (`metrics.c`)** — EPE, timing, benchmarks.

## Open items / to revisit later
- NEON layout requirement (16-byte alignment / width padding): Persona 3 has not requested anything special → dense packing for now. Revisit when `lk_neon.c` is implemented.
