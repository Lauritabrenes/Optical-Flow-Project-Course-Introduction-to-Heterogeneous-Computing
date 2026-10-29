#!/bin/bash
# profile_gpu.sh - Perfilado con desglose GPU (alloc/h2d/kernel/d2h/free)
# Uso: ./profile_gpu.sh [BIN] [FRAME1] [FRAME2] [MODE] [N] [OUT]
set -e
BIN="${1:-../build/bench_main}"
FRAME1="${2:-../data/frames/frame_000001.png}"
FRAME2="${3:-../data/frames/frame_000002.png}"
MODE="${4:-cuda}"
N="${5:-100}"
OUT="${6:-../results/profiling/times_${MODE}_breakdown.csv}"

[[ -x "$BIN" ]] || { echo "ERROR: binario no encontrado: $BIN" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "run,io_ms,preprocess_ms,lk_compute_ms,save_ms,total_ms,gpu_alloc_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_free_ms" > "$OUT"

echo "Perfilando (desglose GPU) modo: $MODE ($N corridas)..."
for i in $(seq 1 "$N"); do
    LINE=$("$BIN" "$FRAME1" "$FRAME2" --mode "$MODE" 2>/dev/null | \
        grep -E "^(\[[1-4]\]|TOTAL|\[3[a-e]\])" | \
        awk '{v=$(NF-1); printf "%s,", v}')
    NF_=$(echo "$LINE" | awk -F',' '{print NF-1}')
    if [[ "$NF_" -ne 10 ]]; then echo "WARN: run $i con $NF_ campos (esperado 10), se omite" >&2; continue; fi
    echo "$i,${LINE%,}" >> "$OUT"
done
echo "Guardado: $OUT ($(( $(wc -l < "$OUT") - 1 ))/$N muestras)"
python3 "$(dirname "$0")/analyze.py" "$OUT"
