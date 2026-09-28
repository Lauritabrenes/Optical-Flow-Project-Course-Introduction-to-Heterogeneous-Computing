# AI Interaction Log — Stage 2 (Preprocessing) Development

**Project:** Optical Flow C Prototype (EL5859 Computación Heterogénea, TEC)
**Stage:** Persona 2 — Preprocessing (`src/preprocess.c`)
**User:** Marlon Mendez N. (mewmarlo)
**AI assistant:** Amazon Quick
**Session date:** Saturday, 2026-09-26
**Purpose:** Record of AI-assisted interactions for the project's "uso adecuado de la IA" documentation.

---

## How to read this log
Each entry lists what was asked (prompt), what files were provided as context, and a
summary of the AI's response. Full outputs (code, guides) were delivered in-chat and,
where noted, saved to the workspace.

---

### Interaction 1 — Scope of Persona 2 for the C prototype
- **Prompt:** "give me a quick guide on what does persona 2 need to implement for the C prototype without interfering the other people"
- **Context provided:** `Proyecto___Computación_Heterogénea-1.pdf` (course project brief), `onboarding.txt` (team/repo onboarding)
- **AI response summary:** Produced a focused guide for Persona 2 (Preprocessing / `preprocess.c`):
  - Scope: `uint8→float32` conversion, buffer allocation/layout, normalization — NOT pyramid or gradients (those are Persona 3, Stage 3).
  - Non-interference rules: work only in `preprocess.c`; never modify `include/of.h`; implement only the declared signatures; honor caller-allocates-memory convention and shared return codes (0/-1/-2).
  - Git workflow: branch `feature/preprocess`, small PRs, merge second in pipeline order; avoid the `Developers` branch until clarified.
  - Noted preprocessing is plain C (no NEON) so it can be developed/tested on x86.

### Interaction 2 — How to build `of_blur_horizontal`
- **Prompt:** "this is the of.h file as reference how can i build the of_blur_horizontal function on this project"
- **Context provided:** `of.h` (shared contract header)
- **AI response summary:**
  - Clarified `of_blur_horizontal` is NOT declared in `of.h` and must NOT be added there — it belongs inside `preprocess.c` as a `static` helper (implementation detail of Stage 2).
  - Provided a full `static of_blur_horizontal` implementation: separable Gaussian, 1-D kernel applied along rows, clamp-to-edge borders, separate src/dst buffers.
  - Showed how to wire it into `of_preprocess` while keeping the public `of.h` signature unchanged, and reiterated the shared return-code / memory conventions.

### Interaction 3 — Verify code for syntax/errors
- **Prompt:** "verify any syntax or error on this code"
- **Context provided:** `preprocess.c`
- **AI response summary:** Reviewed the file and found **3 compile-blocking typos**, all in `of_preprocess` — the variable *declaration* was misspelled while the *usage* was correct:
  - `heigt` → should be `height`
  - `radis` → should be `radius`
  - `scrach` → should be `scratch`
  Provided the corrected declaration block. Confirmed the rest of the file (kernel construction, normalization, separable blur, clamp borders, memory handling) was correct.

### Interaction 4 — Usage guide for other team members
- **Prompt:** "give me a quick guide for the other people in the team on how to use the preprocess"
- **Context provided:** (uses `of.h` + `preprocess.c` from prior turns)
- **AI response summary:** Produced an integration guide for Personas 1/3/4:
  - Single public entry point `of_preprocess(const OFImage *in, OFImage *out)`; all helpers are `static`/internal.
  - Caller allocates BOTH buffers; example call sequence (io → preprocess → lk).
  - I/O contract table: input `float32 [0,255]`, output `float32 [0,1]` smoothed, dimensions preserved.
  - Return codes and per-consumer notes (P1 don't pre-normalize; P3 don't blur again; P4 range is [0,1]; σ fixed at 1.0).

### Interaction 5 — Guide for the unit test
- **Prompt:** "give me a guide for this test"
- **Context provided:** `test_preprocess.c`
- **AI response summary:** Explained the test harness:
  - Dependency-free, synthetic in-memory images, exit code 0 = all pass (CI/ctest friendly).
  - Build/run commands (gcc direct + ctest).
  - `CHECK` macro framework; `fill_constant` / `make_impulse` helpers.
  - Three test cases: (1) argument validation → `OF_ERR_ARGS`; (2) normalization + constant image → all ≈ 1.0; (3) impulse spread → peak attenuated, energy leaks to neighbors, center stays highest.
  - Suggested gaps to add: `OF_ERR_ALLOC` path, odd/1-px widths, vertical spread & symmetry, a `CHECK_NEAR` macro; plus CMake `add_test` registration.

### Interaction 6 — Session log for project documentation
- **Prompt:** "give me a log of this chat to be added to the project documentation"
- **AI response summary:** Compiled a structured development log (scope, `of.h` contract, blur implementation, bug fixes, usage guide, testing) and saved it to the workspace as `artifacts/preprocess_session_log.md`, opened for review. (User subsequently edited that file.)

### Interaction 7 — This AI interaction log
- **Prompt:** "give me a log of this chat because i need to compile of the AI interactions"
- **AI response summary:** Produced this chronological AI-interaction transcript (prompts, context files, response summaries) for the project's AI-usage documentation, saved as `artifacts/ai_interaction_log.md`.

---

## Summary
Across the session, AI assistance was used to: clarify Persona 2's scope and non-interference
boundaries, implement a `static` separable-Gaussian blur helper without touching the shared
`of.h` contract, catch three compile-blocking declaration typos, produce a teammate-facing
usage guide, and explain/extend the unit test suite. All architectural decisions (keeping
helpers `static`, not modifying `of.h`, caller-allocates-memory, fixed σ = 1.0) were reviewed
against the team contract before adoption.
