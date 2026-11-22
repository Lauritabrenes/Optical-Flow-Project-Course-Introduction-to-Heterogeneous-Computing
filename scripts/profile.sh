#!/bin/bash
# ============================================================================
#  scripts/profile.sh
# ----------------------------------------------------------------------------
#  Perfilado del pipeline de Optical Flow.
#
#  Uso:
#      ./profile.sh [BIN] [FRAME1] [FRAME2] [MODE] [N] [OUT]
# ============================================================================

set -e

BIN="${1:-../build/bench_main}"
FRAME1="${2:-../data/frames/frame_000001.png}"
FRAME2="${3:-../data/frames/frame_000002.png}"
MODE="${4:-scalar}"
N="${5:-100}"
OUT="${6:-../results/profiling/times_${MODE}.csv}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binario no encontrado o no ejecutable: $BIN" >&2
    exit 1
fi
[[ -f "$FRAME1" ]] || { echo "ERROR: frame1 no encontrado: $FRAME1" >&2; exit 1; }
[[ -f "$FRAME2" ]] || { echo "ERROR: frame2 no encontrado: $FRAME2" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
echo "run,io_ms,preprocess_ms,lk_compute_ms,save_ms,total_ms" > "$OUT"

echo "Perfilando en modo: $MODE ($N corridas)..."
echo "Ejecutando muestras..."

for i in $(seq 1 "$N"); do
    LINE=$("$BIN" "$FRAME1" "$FRAME2" --mode "$MODE" 2>/dev/null | \
        grep -E "^(\[[1-4]\]|TOTAL)" | \
        awk '{for (j=1; j<=NF; j++) if ($j ~ /^-?[0-9]+(\.[0-9]+)?$/) v=$j; printf "%s,", v}')

    if [[ -z "$LINE" ]]; then
        echo "WARN: run $i sin salida valida del binario (se omite)" >&2
        continue
    fi

    NFIELDS=$(echo "$LINE" | awk -F',' '{print NF-1}')
    if [[ "$NFIELDS" -ne 5 ]]; then
        echo "WARN: run $i con $NFIELDS campos (esperado 5), se omite" >&2
        continue
    fi

    echo "$i,${LINE%,}" >> "$OUT"
done

ROWS=$(( $(wc -l < "$OUT") - 1 ))
echo "Perfilado completo. Archivo guardado en: $OUT ($ROWS/$N muestras validas)"
echo ""
echo "================ ANALISIS ESTADISTICO ================"
# Llamada automática al script de Python asumiendo que está en la misma carpeta
python3 "$(dirname "$0")/analyze.py" "$OUT"
echo "======================================================"